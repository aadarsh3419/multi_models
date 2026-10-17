"""
Unit tests for PDFAgent: end-to-end orchestration, spatial provenance integration, and error handling.
Offline only: all test documents generated dynamically in-memory via PyMuPDF.
"""

import io

import pymupdf
from PIL import Image

from healthcare_agents.agents.pdf_agent import PDFAgent
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
from healthcare_agents.contracts.locations import PdfLocation
from healthcare_agents.contracts.requests import PDFAgentRequest
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.factory import build_pdf_agent
from healthcare_agents.ocr.service import FakeOcrService, OCRLine, OCRResult, OCRWord


class FakeClinicalExtractor(ClinicalFieldExtractor):
    """Offline stub extractor emitting verified clinical facts on matching substrings."""

    def extract(self, unit: ProviderUnit, language_hint: str | None = None):
        candidates = []
        if "Heart rate 72 bpm" in unit.text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Heart rate",
                    value_raw="72",
                    unit_raw="bpm",
                    reference_range_raw="",
                    abnormal_flag_raw="",
                    date_raw="",
                    time_raw="",
                    qualifier_raw="",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    experiencer_relationship_raw="",
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(segment_id=unit.unit_id, quote="Heart rate 72 bpm")
                    ],
                )
            )

        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "fake_extractor",
            "model_id": "test_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "dummy_sha",
            "attempt_count": 1,
            "token_usage": {"total_tokens": 50},
        }


def make_native_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 24
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_scanned_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    img = Image.new("RGB", (400, 600), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(pymupdf.Rect(50, 50, 550, 750), stream=buf.getvalue())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_pdf_agent_offline_native_pdf():
    pdf_bytes = make_native_pdf(
        [
            "Patient Clinical Summary",
            "Vital signs recorded at triage:",
            "Heart rate 72 bpm regular rhythm.",
            "Patient is alert and oriented.",
        ]
    )

    fake_extractor = FakeClinicalExtractor()
    workflow = ExtractionWorkflow(extractor=fake_extractor)
    agent = PDFAgent(workflow=workflow, ocr_service=FakeOcrService())

    req = PDFAgentRequest(content=pdf_bytes, source_name="triage.pdf")
    result = agent.process(req)

    assert result.status == "success"
    assert result.agent_name == "pdf_agent"
    assert result.source_type == "pdf"
    assert len(result.candidates) == 1

    candidate = result.candidates[0]
    assert candidate.label_raw == "Heart rate"
    assert candidate.value_raw == "72"
    assert candidate.unit_raw == "bpm"
    assert len(candidate.evidence_records) == 1

    # Verify spatial provenance on candidate evidence
    ev = candidate.evidence_records[0]
    assert isinstance(ev.location, PdfLocation)
    assert ev.location.page_number == 1
    assert ev.location.coordinate_system == "pdf_points_top_left"
    assert "native" in ev.location.extraction_method
    assert ev.quote_char_span is not None

    # Verify audit trail completeness
    assert result.audit.table_api_available is True
    assert "ocr_mode" in result.audit.ocr_settings
    assert "pymupdf" in result.audit.installed_versions
    assert "pillow" in result.audit.installed_versions
    assert "pytesseract" in result.audit.installed_versions
    assert len(result.audit.page_plans) == 1


def test_pdf_agent_offline_scanned_pdf():
    pdf_bytes = make_scanned_pdf()

    canned_ocr = OCRResult(
        text="Heart rate 72 bpm regular rhythm.",
        lines=[
            OCRLine(
                text="Heart rate 72 bpm regular rhythm.",
                bbox=[100, 100, 500, 150],
                words=[
                    OCRWord(text="Heart", bbox=[100, 100, 160, 150], confidence=98.0),
                    OCRWord(text="rate", bbox=[165, 100, 210, 150], confidence=98.0),
                    OCRWord(text="72", bbox=[215, 100, 245, 150], confidence=99.0),
                    OCRWord(text="bpm", bbox=[250, 100, 290, 150], confidence=97.0),
                    OCRWord(text="regular", bbox=[295, 100, 370, 150], confidence=95.0),
                    OCRWord(text="rhythm.", bbox=[375, 100, 500, 150], confidence=94.0),
                ],
            )
        ],
    )

    fake_ocr = FakeOcrService(canned_result=canned_ocr)
    fake_extractor = FakeClinicalExtractor()
    workflow = ExtractionWorkflow(extractor=fake_extractor)
    agent = PDFAgent(workflow=workflow, ocr_service=fake_ocr)

    req = PDFAgentRequest(content=pdf_bytes)
    result = agent.process(req)

    assert result.status == "success"
    assert len(result.candidates) == 1
    ev = result.candidates[0].evidence_records[0]
    assert isinstance(ev.location, PdfLocation)
    assert "ocr" in ev.location.extraction_method


def test_pdf_agent_remote_processing_disabled_status():
    pdf_bytes = make_native_pdf(
        ["Clinical Notes: Patient stable.", "Heart rate 72 bpm.", "Blood pressure normal."]
    )

    settings = Settings(allow_remote_llm_processing=False)
    agent = build_pdf_agent(settings=settings, ocr_service=FakeOcrService())

    req = PDFAgentRequest(content=pdf_bytes)
    result = agent.process(req)

    # Remote processing is disabled -> status partial with error recorded
    assert result.status == "partial"
    assert result.stage_results["source_reading"].status == "success"
    assert any(e.code == ErrorCode.REMOTE_PROCESSING_DISABLED.value for e in result.errors)


def test_pdf_agent_corrupt_pdf_failure():
    corrupt_bytes = b"%PDF-1.4\nGarbage data not a valid PDF"
    agent = build_pdf_agent(ocr_service=FakeOcrService())

    req = PDFAgentRequest(content=corrupt_bytes)
    result = agent.process(req)

    assert result.status == "failed"
    assert len(result.candidates) == 0
    assert any(e.code == ErrorCode.PDF_UNREADABLE.value for e in result.errors)


def test_pdf_agent_file_too_large_failure():
    pdf_bytes = make_native_pdf(["Line of text"])
    settings = Settings(max_pdf_bytes=50)  # low threshold
    agent = build_pdf_agent(settings=settings, ocr_service=FakeOcrService())

    req = PDFAgentRequest(content=pdf_bytes)
    result = agent.process(req)

    assert result.status == "failed"
    assert any(e.code == ErrorCode.FILE_TOO_LARGE.value for e in result.errors)


def test_pdf_agent_privacy_and_phi_redaction():
    pdf_bytes = make_native_pdf(["Patient: Jane Doe, DOB: 1980-01-01"])
    req = PDFAgentRequest(content=pdf_bytes, source_name="jane_doe_record.pdf")

    # __repr__ and __str__ must strictly redact raw bytes and source_name
    repr_str = repr(req)
    assert "jane_doe_record.pdf" not in repr_str
    assert "Jane Doe" not in repr_str
    assert "content" not in repr_str or "REDACTED" in repr_str

    # Process without debug source text
    agent = build_pdf_agent(ocr_service=FakeOcrService())
    result = agent.process(req)

    assert result.contains_source_text is False
    assert result.debug_source_text is None
