"""
Unit tests for ImageAgent modality orchestration, provenance, safety, and PHI protection.
"""

import io
from typing import Any

from PIL import Image

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
from healthcare_agents.contracts.locations import ImageLocation
from healthcare_agents.contracts.requests import ImageAgentOptions, ImageAgentRequest
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.factory import build_image_agent
from healthcare_agents.ocr.service import FakeOcrService, OCRLine, OCRResult, OCRWord


class FakeExtractor(ClinicalFieldExtractor):
    """Synthetic extractor for deterministic testing without external API calls."""

    def __init__(self, candidates: list[ProviderCandidate] | None = None):
        self.candidates = candidates or []
        self.call_count = 0

    def extract(
        self,
        unit: ProviderUnit,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        self.call_count += 1
        adjusted_candidates = []
        for c in self.candidates:
            c_copy = c.model_copy(deep=True)
            for ref in c_copy.evidence_refs:
                ref.segment_id = unit.unit_id
            adjusted_candidates.append(c_copy)

        return (
            ProviderExtractionResponse(candidates=adjusted_candidates),
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "model_id": "fake-extractor-v1",
                "provider": "fake",
            },
        )


def _make_image_bytes(width: int = 800, height: int = 600, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_image_agent_success_orchestration():
    """Verify ImageAgent coordinates reader and ExtractionWorkflow end-to-end."""
    quote_text = "Fasting Blood Glucose 110 mg/dL"
    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text=quote_text,
            lines=[
                OCRLine(
                    text=quote_text,
                    bbox=[50, 100, 450, 140],
                    words=[
                        OCRWord(text="Fasting", bbox=[50, 100, 120, 140], confidence=98.0),
                        OCRWord(text="Blood", bbox=[125, 100, 180, 140], confidence=97.0),
                        OCRWord(text="Glucose", bbox=[185, 100, 260, 140], confidence=99.0),
                        OCRWord(text="110", bbox=[270, 100, 310, 140], confidence=96.0),
                        OCRWord(text="mg/dL", bbox=[315, 100, 380, 140], confidence=95.0),
                    ],
                )
            ],
            words=[],
        )
    )

    fake_extractor = FakeExtractor(
        candidates=[
            ProviderCandidate(
                fact_type=FactType.MEASUREMENT,
                label_raw="Fasting Blood Glucose",
                value_raw="110",
                unit_raw="mg/dL",
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
                        segment_id="placeholder",
                        quote=quote_text,
                    )
                ],
            )
        ]
    )

    agent = build_image_agent(
        extractor=fake_extractor,
        ocr_service=fake_ocr,
    )

    image_bytes = _make_image_bytes(width=1200, height=800, fmt="PNG")
    request = ImageAgentRequest(
        content=image_bytes,
        source_name="lab_slip_scan.png",
        options=ImageAgentOptions(ocr_languages="eng"),
    )

    result = agent.process(request)

    assert result.status == "success"
    assert result.source_type == "image"
    assert result.agent_name == "image_agent"
    assert len(result.candidates) == 1

    candidate = result.candidates[0]
    assert candidate.fact_type == "measurement"
    assert candidate.value_raw == "110"
    assert candidate.unit_raw == "mg/dL"
    assert candidate.value_parsed is not None
    assert candidate.value_parsed.numeric_value == 110.0

    # Spatial provenance verification (scaled coordinates [50, 100, 450, 140] inverted by scale 1.25)
    assert len(candidate.evidence_records) == 1
    ev = candidate.evidence_records[0]
    assert ev.quote == quote_text
    assert isinstance(ev.location, ImageLocation)
    assert ev.location.bbox == [40.0, 80.0, 360.0, 112.0]
    assert ev.location.original_width == 1200
    assert ev.location.original_height == 800

    # Audit metadata verification
    assert "pillow" in result.audit.installed_versions
    assert result.audit.ocr_settings["ocr_languages"] == "eng"
    assert result.audit.image_details["format"] == "PNG"
    assert result.audit.image_details["raw_width"] == 1200
    assert result.audit.image_details["raw_height"] == 800
    assert result.audit.image_details["upscale_factor"] == 1.25


def test_image_agent_phi_protection_and_repr():
    """Verify raw bytes and source filename are strictly redacted from request repr and string representations."""
    raw_content = b"TEST_IMAGE_BYTES_WITH_POTENTIAL_PHI"
    req = ImageAgentRequest(
        content=raw_content,
        source_name="patient_john_doe_rx.png",
        options=ImageAgentOptions(),
    )

    repr_str = repr(req)
    str_str = str(req)

    assert "john_doe" not in repr_str
    assert "john_doe" not in str_str
    assert "POTENTIAL_PHI" not in repr_str
    assert "POTENTIAL_PHI" not in str_str
    assert "[REDACTED]" in repr_str


def test_image_agent_offline_partial_status():
    """Verify ImageAgent returns status=partial when remote processing is disabled by default."""
    image_bytes = _make_image_bytes(width=600, height=400, fmt="PNG")
    settings = Settings(allow_remote_llm_processing=False)
    agent = build_image_agent(settings=settings, ocr_service=FakeOcrService())

    request = ImageAgentRequest(
        content=image_bytes,
        source_name="report.png",
        options=ImageAgentOptions(),
    )

    result = agent.process(request)

    # Ingestion succeeds, but LLM extraction cannot proceed offline without remote credentials
    assert result.status == "partial"
    assert result.source_type == "image"
    assert result.agent_name == "image_agent"
    assert any(e.code == ErrorCode.REMOTE_PROCESSING_DISABLED.value for e in result.errors)
    assert result.stage_results["source_reading"].status == "success"


def test_image_agent_corrupt_content_failure():
    """Verify corrupt image bytes produce status=failed with structured diagnostic error."""
    agent = build_image_agent(ocr_service=FakeOcrService())
    request = ImageAgentRequest(
        content=b"CORRUPT_NON_IMAGE_DATA_12345",
        source_name="corrupt.png",
        options=ImageAgentOptions(),
    )

    result = agent.process(request)

    assert result.status == "failed"
    assert result.source_type == "image"
    assert len(result.errors) >= 1
    assert result.errors[0].code == ErrorCode.IMAGE_UNREADABLE.value
    assert result.stage_results["source_reading"].status == "failed"
