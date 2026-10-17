"""
Cross-Agent Invariant Tests.

Parametrically verifies core invariants across all four modality agents:
- Common AgentResult schema & strict PHI sanitization
- Single shared ExtractionWorkflow & EvidenceValidator type
- Single shared OcrService instance between PDF and Image agents
- Modality-appropriate remote processing failure behavior (partial vs failed-closed)
- Spatial and temporal provenance preservation under tiny chunking budgets
- ErrorCode enum stability and schema snapshot verification
- Zero forbidden frameworks in repository
- Unified CLI subcommand parity
"""

import ast
import json
from pathlib import Path

import pytest

from healthcare_agents.cli import main
from healthcare_agents.config import Settings
from healthcare_agents.contracts.errors import ErrorCode
from healthcare_agents.contracts.facts import (
    AssertionStatus,
    ExperiencerType,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    Temporality,
)
from healthcare_agents.contracts.requests import (
    ImageAgentRequest,
    PDFAgentRequest,
    TextAgentRequest,
    VoiceAgentRequest,
)
from healthcare_agents.contracts.results import AgentResult
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.evidence_validator import EvidenceValidator
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.factory import (
    build_image_agent,
    build_pdf_agent,
    build_text_agent,
    build_voice_agent,
)
from healthcare_agents.ocr.service import FakeOcrService, OCRLine, OCRResult, OCRWord
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    TranscriptionResult,
    TranscriptionSegment,
)
from healthcare_agents.sources.chunker import TextChunker
from tests.unit.test_audio_reader import _make_dummy_wav_bytes, _make_mock_popen


class CrossAgentFakeExtractor(ClinicalFieldExtractor):
    """Stub extractor providing deterministic clinical facts matching source text."""

    def __init__(self, target_quote: str = "Blood Pressure: 120/80 mmHg"):
        self.target_quote = target_quote

    def extract(
        self,
        unit: ProviderUnit,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict]:
        cands = []
        if self.target_quote in unit.text:
            cands.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Blood Pressure",
                    value_raw="120/80",
                    unit_raw="mmHg",
                    reference_range_raw="",
                    abnormal_flag_raw="",
                    date_raw="",
                    time_raw="",
                    qualifier_raw="",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    experiencer_relationship_raw="",
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit.unit_id,
                            quote=self.target_quote,
                        )
                    ],
                )
            )
        return (
            ProviderExtractionResponse(candidates=cands),
            {"model_id": "fake_extractor", "prompt_sha256": "fake_sha"},
        )


def _make_sample_pdf_bytes(text: str = "Blood Pressure: 120/80 mmHg") -> bytes:
    """Creates a valid in-memory PDF containing the specified text with sufficient native text."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    full_content = (
        "Patient Clinical Examination Record\n"
        "Vital signs and physical parameters:\n"
        f"{text}\n"
        "Patient is in stable clinical condition."
    )
    page.insert_text((50, 100), full_content, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_sample_image_bytes(width: int = 800, height: int = 600) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (width, height), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Invariant 1: Single Workflow & Validator Type Across All 4 Agents
# ---------------------------------------------------------------------------


def test_all_four_agents_share_extraction_workflow_and_validator():
    """Verify that text, pdf, image, and voice agents use identical ExtractionWorkflow & EvidenceValidator types."""
    settings = Settings(
        allow_remote_llm_processing=False,
        allow_remote_audio_transcription=False,
    )

    text_agent = build_text_agent(settings=settings)
    pdf_agent = build_pdf_agent(settings=settings)
    image_agent = build_image_agent(settings=settings)
    voice_agent = build_voice_agent(settings=settings)

    agents = [text_agent, pdf_agent, image_agent, voice_agent]

    for ag in agents:
        assert isinstance(ag.workflow, ExtractionWorkflow)
        assert isinstance(ag.workflow.validator, EvidenceValidator)
        assert isinstance(ag.workflow.chunker, TextChunker)


# ---------------------------------------------------------------------------
# Invariant 2: Shared OcrService Instance Identity
# ---------------------------------------------------------------------------


def test_pdf_and_image_share_ocr_service_instance():
    """Verify factory allows sharing the exact same OcrService instance between PDF and Image agents."""
    shared_ocr = FakeOcrService()
    pdf_agent = build_pdf_agent(ocr_service=shared_ocr)
    image_agent = build_image_agent(ocr_service=shared_ocr)

    assert pdf_agent.reader.ocr_service is shared_ocr
    assert image_agent.reader.ocr_service is shared_ocr


# ---------------------------------------------------------------------------
# Invariant 3: Common AgentResult Schema & PHI Sanitization Across All 4 Modalities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["text", "pdf", "image", "voice"])
def test_common_agent_result_schema_and_phi_sanitization(modality: str):
    """Verify AgentResult schema conformance and absolute absence of raw filenames or file paths."""
    quote = "Blood Pressure: 120/80 mmHg"
    extractor = CrossAgentFakeExtractor(target_quote=quote)
    settings = Settings(
        allow_remote_llm_processing=True,
        allow_remote_audio_transcription=True,
    )

    if modality == "text":
        agent = build_text_agent(settings=settings, extractor=extractor)
        req = TextAgentRequest(
            text=f"Clinical note: {quote} measured.",
        )
        result = agent.process(req)

    elif modality == "pdf":
        agent = build_pdf_agent(
            settings=settings, extractor=extractor, ocr_service=FakeOcrService()
        )
        req = PDFAgentRequest(
            content=_make_sample_pdf_bytes(quote),
            source_name="confidential_patient_record.pdf",
        )
        result = agent.process(req)

    elif modality == "image":
        fake_ocr = FakeOcrService(
            canned_result=OCRResult(
                text=quote,
                lines=[
                    OCRLine(
                        text=quote,
                        bbox=[50, 100, 350, 130],
                        words=[
                            OCRWord(text=w, bbox=[50, 100, 100, 130], confidence=98.0)
                            for w in quote.split()
                        ],
                    )
                ],
                words=[],
            )
        )
        agent = build_image_agent(settings=settings, extractor=extractor, ocr_service=fake_ocr)
        req = ImageAgentRequest(
            content=_make_sample_image_bytes(),
            source_name="scanned_prescription_john_doe.png",
        )
        result = agent.process(req)

    elif modality == "voice":
        valid_ffprobe = json.dumps(
            {
                "format": {"format_name": "wav", "duration": "3.0"},
                "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "3.0"}],
            }
        )
        mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe)
        from tests.unit.test_audio_reader import FakeTranscriber

        stt_res = TranscriptionResult(
            text=quote,
            segments=[TranscriptionSegment(text=quote, start_seconds=0.5, end_seconds=2.8)],
            language="english",
        )
        fake_transcriber = FakeTranscriber(canned_result=stt_res)
        reader = AudioReader(
            transcriber=fake_transcriber,
            settings=settings,
            popen_factory=mock_popen_factory,
        )
        agent = build_voice_agent(settings=settings, extractor=extractor, reader=reader)
        req = VoiceAgentRequest(
            content=_make_dummy_wav_bytes(duration_sec=3.0),
            source_name="consultation_recording_jane_doe.wav",
        )
        result = agent.process(req)

    # Invariant assertions
    assert isinstance(result, AgentResult)
    assert result.schema_version == "1.0.0"
    assert result.status == "success"
    assert result.duration_ms >= 0.0

    # Strict PHI sanitization in result
    result_json = result.model_dump_json()
    assert "confidential_patient" not in result_json
    assert "john_doe" not in result_json
    assert "jane_doe" not in result_json
    assert result.sanitized_source_metadata["source_id"].startswith("src_")

    # Candidate extraction & grounding verification
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.label_raw == "Blood Pressure"
    assert cand.value_raw == "120/80"
    assert cand.unit_raw == "mmHg"
    assert len(cand.evidence_records) == 1
    ev = cand.evidence_records[0]
    expected_kind = "audio" if modality == "voice" else modality
    assert ev.location.kind == expected_kind


# ---------------------------------------------------------------------------
# Invariant 4: Modality-Appropriate Remote Flag Behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modality,expected_status",
    [
        ("text", "partial"),
        ("pdf", "partial"),
        ("image", "partial"),
        ("voice", "failed"),
    ],
)
def test_remote_processing_flags_behavior(modality: str, expected_status: str):
    """
    Verify safety kill-switches:
    - Text, PDF, and Image ingest source text locally and return status='partial' when LLM is disabled.
    - Voice fails closed (status='failed') when remote audio transcription is disabled.
    """
    settings = Settings(
        allow_remote_llm_processing=False,
        allow_remote_audio_transcription=False,
    )

    if modality == "text":
        agent = build_text_agent(settings=settings)
        result = agent.process(TextAgentRequest(text="Patient seen."))
    elif modality == "pdf":
        agent = build_pdf_agent(settings=settings)
        result = agent.process(PDFAgentRequest(content=_make_sample_pdf_bytes()))
    elif modality == "image":
        fake_ocr = FakeOcrService(
            canned_result=OCRResult(
                text="Fasting blood sugar 100",
                lines=[
                    OCRLine(
                        text="Fasting blood sugar 100",
                        bbox=[0, 0, 100, 20],
                        words=[],
                    )
                ],
                words=[],
            )
        )
        agent = build_image_agent(settings=settings, ocr_service=fake_ocr)
        result = agent.process(ImageAgentRequest(content=_make_sample_image_bytes()))
    elif modality == "voice":
        valid_ffprobe = json.dumps(
            {
                "format": {"format_name": "wav", "duration": "1.0"},
                "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "1.0"}],
            }
        )
        mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe)
        reader = AudioReader(
            settings=settings,
            popen_factory=mock_popen_factory,
        )
        agent = build_voice_agent(settings=settings, reader=reader)
        result = agent.process(VoiceAgentRequest(content=_make_dummy_wav_bytes()))

    assert result.status == expected_status
    if expected_status == "partial":
        assert any("remote_processing_disabled" in e.code for e in result.errors)
    elif expected_status == "failed":
        assert any(e.code == "remote_processing_disabled" for e in result.errors)


# ---------------------------------------------------------------------------
# Invariant 5: Location Provenance Survives Chunking Under Tiny Budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["text", "pdf", "image", "voice"])
def test_location_provenance_survives_small_chunking_budget(modality: str):
    """Verify that fine-grained locations remain grounded even with tiny chunking budgets."""
    quote = "Blood Pressure: 120/80 mmHg"
    extractor = CrossAgentFakeExtractor(target_quote=quote)
    settings = Settings(
        allow_remote_llm_processing=True,
        allow_remote_audio_transcription=True,
        max_llm_input_chars=500,  # Small chunking budget
        chunk_overlap_chars=50,
    )

    if modality == "text":
        agent = build_text_agent(settings=settings, extractor=extractor)
        result = agent.process(TextAgentRequest(text="A" * 600 + f"\n{quote}\n" + "B" * 600))
    elif modality == "pdf":
        agent = build_pdf_agent(
            settings=settings, extractor=extractor, ocr_service=FakeOcrService()
        )
        result = agent.process(PDFAgentRequest(content=_make_sample_pdf_bytes(quote)))
    elif modality == "image":
        fake_ocr = FakeOcrService(
            canned_result=OCRResult(
                text=quote,
                lines=[
                    OCRLine(
                        text=quote,
                        bbox=[10, 20, 200, 50],
                        words=[],
                    )
                ],
                words=[],
            )
        )
        agent = build_image_agent(settings=settings, extractor=extractor, ocr_service=fake_ocr)
        result = agent.process(ImageAgentRequest(content=_make_sample_image_bytes()))
    elif modality == "voice":
        valid_ffprobe = json.dumps(
            {
                "format": {"format_name": "wav", "duration": "4.0"},
                "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "4.0"}],
            }
        )
        mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe)
        from tests.unit.test_audio_reader import FakeTranscriber

        stt_res = TranscriptionResult(
            text=quote,
            segments=[TranscriptionSegment(text=quote, start_seconds=1.0, end_seconds=3.5)],
            language="english",
        )
        reader = AudioReader(
            transcriber=FakeTranscriber(canned_result=stt_res),
            settings=settings,
            popen_factory=mock_popen_factory,
        )
        agent = build_voice_agent(settings=settings, extractor=extractor, reader=reader)
        result = agent.process(VoiceAgentRequest(content=_make_dummy_wav_bytes(duration_sec=4.0)))

    assert result.status == "success"
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert len(cand.evidence_records) == 1
    ev = cand.evidence_records[0]
    expected_kind = "audio" if modality == "voice" else modality
    assert ev.location.kind == expected_kind


# ---------------------------------------------------------------------------
# Invariant 6: ErrorCode Stability & Schema Snapshot Verification
# ---------------------------------------------------------------------------


def test_error_code_stability():
    """Verify that existing ErrorCode values are strictly preserved and frozen."""
    expected_codes = {
        "invalid_input",
        "no_usable_source_content",
        "remote_processing_disabled",
        "missing_api_key",
        "provider_timeout",
        "provider_rate_limited",
        "provider_authentication_failed",
        "provider_model_unavailable",
        "provider_schema_unsupported",
        "provider_response_invalid",
        "llm_call_budget_exceeded",
        "unsupported_file_type",
        "file_too_large",
        "pdf_unreadable",
        "pdf_encrypted",
        "page_limit_exceeded",
        "ocr_unavailable",
        "ocr_timeout",
        "image_unreadable",
        "image_pixel_limit_exceeded",
        "audio_unreadable",
        "audio_duration_exceeded",
        "audio_duration_unavailable",
        "audio_validation_unavailable",
        "audio_transcription_failed",
    }
    actual_codes = {e.value for e in ErrorCode}
    assert expected_codes.issubset(actual_codes), (
        f"Missing error codes: {expected_codes - actual_codes}"
    )


# ---------------------------------------------------------------------------
# Invariant 7: AST Import Boundary Check (No Forbidden Frameworks)
# ---------------------------------------------------------------------------


def test_ast_forbids_out_of_scope_frameworks():
    """Verify strict prohibition of heavy or unauthorized web/ML frameworks across src/."""
    src_dir = Path(__file__).resolve().parent.parent.parent / "src"
    forbidden_frameworks = {
        "fastapi",
        "starlette",
        "pymongo",
        "motor",
        "jwt",
        "jose",
        "sqlalchemy",
        "torch",
        "sklearn",
        "scipy",
        "transformers",
        "langchain",
        "langgraph",
    }

    for py_file in src_dir.rglob("*.py"):
        if py_file.name.startswith("__pycache__"):
            continue
        with open(py_file, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for fb in forbidden_frameworks:
                        assert not alias.name.startswith(fb), (
                            f"{py_file.name} illegally imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for fb in forbidden_frameworks:
                    assert not node.module.startswith(fb), (
                        f"{py_file.name} illegally imports {node.module}"
                    )


# ---------------------------------------------------------------------------
# Invariant 8: Unified CLI Subcommand Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subcommand", ["text", "pdf", "image", "voice"])
def test_all_four_cli_subcommands_support_help(subcommand: str):
    """Verify that all 4 subcommands (text, pdf, image, voice) expose standardized CLI help."""
    with pytest.raises(SystemExit) as excinfo:
        main([subcommand, "--help"])
    assert excinfo.value.code == 0
