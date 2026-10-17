"""
Unit tests for VoiceAgent orchestration, PHI redaction, and status semantics.
"""

import json

from healthcare_agents.agents.voice_agent import VoiceAgent
from healthcare_agents.config import Settings
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
from healthcare_agents.contracts.requests import VoiceAgentRequest
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.factory import _build_workflow
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    TranscriptionResult,
    TranscriptionSegment,
)
from tests.unit.test_audio_reader import (
    FakeTranscriber,
    _make_dummy_wav_bytes,
    _make_mock_popen,
)


class FakeExtractor(ClinicalFieldExtractor):
    """Deterministic in-memory field extractor."""

    def __init__(self, candidates: list[ProviderCandidate] | None = None):
        self.candidates = candidates or []

    def extract(
        self,
        unit: ProviderUnit,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict]:
        adjusted = []
        for c in self.candidates:
            c_copy = c.model_copy(deep=True)
            for r in c_copy.evidence_refs:
                r.segment_id = unit.unit_id
            adjusted.append(c_copy)
        return (
            ProviderExtractionResponse(candidates=adjusted),
            {"model_id": "fake_model", "prompt_sha256": "fake_sha"},
        )


def _build_test_voice_agent(
    transcriber_result: TranscriptionResult | None = None,
    extractor_candidates: list[ProviderCandidate] | None = None,
    ffprobe_duration: float = 5.0,
    allow_remote_audio: bool = True,
    allow_remote_llm: bool = True,
) -> VoiceAgent:
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": str(ffprobe_duration)},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "duration": str(ffprobe_duration),
                }
            ],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber(canned_result=transcriber_result)

    settings = Settings(
        allow_remote_audio_transcription=allow_remote_audio,
        allow_remote_llm_processing=allow_remote_llm,
    )

    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    fake_extractor = FakeExtractor(candidates=extractor_candidates) if allow_remote_llm else None
    workflow = _build_workflow(cfg=settings, extractor=fake_extractor)

    return VoiceAgent(
        workflow=workflow,
        reader=reader,
        settings=settings,
    )


def test_voice_agent_orchestration_success():
    """Verify end-to-end VoiceAgent extraction with temporal grounding."""
    text = "Patient was diagnosed with Hypertension on 12/05/2023."
    stt_res = TranscriptionResult(
        text=text,
        segments=[
            TranscriptionSegment(
                text=text,
                start_seconds=1.2,
                end_seconds=4.8,
            )
        ],
        language="english",
    )
    cands = [
        ProviderCandidate(
            fact_type=FactType.CONDITION,
            label_raw="Hypertension",
            value_raw="",
            unit_raw="",
            reference_range_raw="",
            abnormal_flag_raw="",
            date_raw="12/05/2023",
            time_raw="",
            qualifier_raw="",
            assertion_status=AssertionStatus.AFFIRMED,
            temporality=Temporality.PAST,
            experiencer_type=ExperiencerType.PATIENT,
            experiencer_relationship_raw="",
            reporting_source_role=ReportingSourceRole.CLINICIAN,
            evidence_refs=[
                ProviderEvidenceRef(
                    segment_id="placeholder",
                    quote=text,
                )
            ],
        )
    ]

    agent = _build_test_voice_agent(
        transcriber_result=stt_res,
        extractor_candidates=cands,
    )

    req = VoiceAgentRequest(
        content=_make_dummy_wav_bytes(duration_sec=5.0),
        source_name="consultation_audio.wav",
        language_hint="en",
    )

    result = agent.process(req)

    assert result.status == "success"
    assert len(result.candidates) == 1
    cand = result.candidates[0]
    assert cand.label_raw == "Hypertension"
    # Verify temporal location preserved
    assert len(cand.evidence_records) == 1
    ev = cand.evidence_records[0]
    assert ev.location is not None
    assert ev.location.kind == "audio"
    assert ev.location.timestamps_available is True
    assert ev.location.start_seconds == 1.2
    assert ev.location.end_seconds == 4.8


def test_voice_agent_phi_redaction():
    """Verify VoiceAgentRequest and AgentResult strictly redact PHI and source filenames."""
    content = _make_dummy_wav_bytes(duration_sec=2.0)
    req = VoiceAgentRequest(
        content=content,
        source_name="secret_patient_john_doe_voice.wav",
        language_hint="en",
    )

    repr_str = repr(req)
    assert "secret_patient" not in repr_str
    assert "[REDACTED]" in repr_str
    assert str(content) not in repr_str

    agent = _build_test_voice_agent(
        transcriber_result=TranscriptionResult(
            text="Fever noted.",
            segments=[
                TranscriptionSegment(text="Fever noted.", start_seconds=0.0, end_seconds=1.0)
            ],
        )
    )

    result = agent.process(req)
    result_json = result.model_dump_json()

    # Ensure raw filename never leaks into output JSON
    assert "secret_patient_john_doe_voice" not in result_json
    assert result.sanitized_source_metadata["source_id"].startswith("src_aud_")


def test_voice_agent_status_partial_when_remote_llm_disabled():
    """Verify status is 'partial' when audio reading succeeds but LLM processing is disabled."""
    stt_res = TranscriptionResult(
        text="Blood Pressure 120/80 mmHg recorded.",
        segments=[
            TranscriptionSegment(
                text="Blood Pressure 120/80 mmHg recorded.",
                start_seconds=0.0,
                end_seconds=2.0,
            )
        ],
    )
    agent = _build_test_voice_agent(
        transcriber_result=stt_res,
        allow_remote_llm=False,  # LLM disabled
    )

    req = VoiceAgentRequest(
        content=_make_dummy_wav_bytes(duration_sec=2.0),
        source_name="bp_reading.wav",
    )

    result = agent.process(req)
    assert result.status == "partial"
    assert len(result.candidates) == 0
    assert any("remote_processing_disabled" in e.code for e in result.errors)


def test_voice_agent_status_failed_when_audio_invalid():
    """Verify status is 'failed' when audio bytes are corrupt or unsupported format."""
    agent = _build_test_voice_agent()

    req = VoiceAgentRequest(
        content=b"CORRUPT_NOT_AUDIO_BYTES_HEADER",
        source_name="bad.wav",
    )

    result = agent.process(req)
    assert result.status == "failed"
    assert len(result.candidates) == 0
    assert any(e.code == "unsupported_file_type" for e in result.errors)
