"""
Unit tests for Contracts: requests, locations (all 4 modalities), facts, results, and errors.
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from healthcare_agents.contracts import (
    AgentResult,
    AssertionStatus,
    AudioLocation,
    AuditMetadata,
    DateParseStatus,
    EvidenceStatus,
    ExperiencerType,
    FactType,
    ImageLocation,
    ParseStatus,
    PdfLocation,
    ProviderCandidate,
    ReportingSourceRole,
    ReviewStatus,
    SourceLocation,
    StageResult,
    Temporality,
    TextAgentRequest,
    TextLocation,
    compute_candidate_id,
)

# ---------------------------------------------------------------------------
# 1. Request Contracts & PHI Redaction
# ---------------------------------------------------------------------------


def test_text_agent_request_valid():
    req = TextAgentRequest(text="Patient presents with mild fever.", language_hint="en")
    assert req.text == "Patient presents with mild fever."
    assert req.language_hint == "en"
    assert req.request_id is not None
    assert req.options.include_source_text_in_result is False


def test_text_agent_request_empty_rejection():
    with pytest.raises(ValidationError):
        TextAgentRequest(text="")

    with pytest.raises(ValidationError):
        TextAgentRequest(text="   \n\t  ")


def test_text_agent_request_phi_safe_repr():
    secret_text = "Patient John Doe SSN 000-11-2222 diagnosed with Stage 4 Glioblastoma."
    req = TextAgentRequest(text=secret_text)
    repr_str = repr(req)
    str_str = str(req)

    assert "John Doe" not in repr_str
    assert "Glioblastoma" not in repr_str
    assert "000-11-2222" not in repr_str
    assert "[REDACTED" in repr_str
    assert str_str == repr_str


# ---------------------------------------------------------------------------
# 2. Multi-Modal SourceLocation Round-Trips
# ---------------------------------------------------------------------------


def test_text_location_valid_and_invalid():
    loc = TextLocation(start=10, end=25)
    assert loc.kind == "text"
    assert loc.start == 10
    assert loc.end == 25

    with pytest.raises(ValidationError):
        TextLocation(start=10, end=10)  # end must be > start

    with pytest.raises(ValidationError):
        TextLocation(start=20, end=10)  # end must be > start


def test_pdf_location_round_trip():
    loc_data = {
        "kind": "pdf",
        "page_number": 2,
        "bbox": [72.0, 100.0, 200.0, 150.0],
        "coordinate_system": "pdf_points_top_left",
        "page_width": 612.0,
        "page_height": 792.0,
        "rotation": 0,
        "block_id": "b001",
    }
    loc = PdfLocation.model_validate(loc_data)
    assert loc.page_number == 2
    assert loc.coordinate_system == "pdf_points_top_left"

    # Discriminated union round trip
    adapter = TypeAdapter(SourceLocation)
    union_loc = adapter.validate_python(loc_data)
    dumped = json.loads(adapter.dump_json(union_loc))
    assert dumped["kind"] == "pdf"
    assert dumped["page_number"] == 2

    # Validation: bbox present without coordinate_system must fail
    with pytest.raises(ValidationError):
        PdfLocation(page_number=1, bbox=[10.0, 20.0, 30.0, 40.0], coordinate_system=None)


def test_image_location_round_trip():
    loc_data = {
        "kind": "image",
        "bbox": [100.0, 150.0, 400.0, 300.0],
        "original_width": 1920,
        "original_height": 1080,
        "coordinate_system": "pixel_top_left",
    }
    loc = ImageLocation.model_validate(loc_data)
    assert loc.kind == "image"
    assert loc.original_width == 1920

    dumped = loc.model_dump()
    assert dumped["coordinate_system"] == "pixel_top_left"


def test_audio_location_round_trip():
    # Timestamps available
    loc_timed = AudioLocation(
        timestamps_available=True,
        start_seconds=12.5,
        end_seconds=18.0,
    )
    assert loc_timed.kind == "audio"
    assert loc_timed.end_seconds == 18.0

    # End before start must fail
    with pytest.raises(ValidationError):
        AudioLocation(timestamps_available=True, start_seconds=20.0, end_seconds=10.0)

    # Timestamps unavailable
    loc_untimed = AudioLocation(
        timestamps_available=False,
        start_seconds=None,
        end_seconds=None,
    )
    assert loc_untimed.timestamps_available is False

    # Providing timestamps when timestamps_available is False must fail
    with pytest.raises(ValidationError):
        AudioLocation(timestamps_available=False, start_seconds=1.0, end_seconds=2.0)


# ---------------------------------------------------------------------------
# 3. Facts, Enums, and Deterministic Candidate ID
# ---------------------------------------------------------------------------


def test_enum_exact_values():
    assert AssertionStatus.AFFIRMED.value == "affirmed"
    assert AssertionStatus.NEGATED.value == "negated"
    assert AssertionStatus.UNCERTAIN.value == "uncertain"

    assert Temporality.CURRENT.value == "current"
    assert Temporality.NOT_APPLICABLE.value == "not_applicable"

    assert ExperiencerType.PATIENT.value == "patient"
    assert ExperiencerType.FAMILY_MEMBER.value == "family_member"

    assert ReportingSourceRole.CLINICIAN.value == "clinician"
    assert ReportingSourceRole.DEVICE.value == "device"

    assert EvidenceStatus.MATCHED.value == "matched"
    assert EvidenceStatus.AMBIGUOUS_MATCH.value == "ambiguous_match"

    assert DateParseStatus.PARSED.value == "parsed"
    assert ParseStatus.PARSED.value == "parsed"
    assert ReviewStatus.UNREVIEWED.value == "unreviewed"


def test_candidate_id_deterministic():
    content_sha = "abcdef0123456789"
    raw_fields = {
        "value_raw": "120/80",
        "unit_raw": "mmHg",
        "reference_range_raw": "",
        "abnormal_flag_raw": "",
        "date_raw": "2026-03-15",
        "time_raw": "",
        "qualifier_raw": "",
        "relationship_raw": "",
    }
    spans = [("s0001", 10, 25)]

    id1 = compute_candidate_id(
        content_sha256=content_sha,
        fact_type=FactType.MEASUREMENT,
        normalized_label="blood pressure",
        raw_fields=raw_fields,
        assertion=AssertionStatus.AFFIRMED,
        temporality=Temporality.NOT_APPLICABLE,
        experiencer=ExperiencerType.PATIENT,
        role=ReportingSourceRole.CLINICIAN,
        evidence_spans=spans,
    )
    id2 = compute_candidate_id(
        content_sha256=content_sha,
        fact_type=FactType.MEASUREMENT,
        normalized_label="blood pressure",
        raw_fields=raw_fields,
        assertion=AssertionStatus.AFFIRMED,
        temporality=Temporality.NOT_APPLICABLE,
        experiencer=ExperiencerType.PATIENT,
        role=ReportingSourceRole.CLINICIAN,
        evidence_spans=spans,
    )

    assert id1.startswith("fc_")
    assert len(id1) == 19  # "fc_" (3) + 16 hex chars
    assert id1 == id2


def test_provider_schema_extra_fields_forbidden():
    valid_data = {
        "fact_type": "symptom",
        "label_raw": "headache",
        "value_raw": "",
        "unit_raw": "",
        "reference_range_raw": "",
        "abnormal_flag_raw": "",
        "date_raw": "",
        "time_raw": "",
        "qualifier_raw": "",
        "assertion_status": "affirmed",
        "temporality": "current",
        "experiencer_type": "patient",
        "experiencer_relationship_raw": "",
        "reporting_source_role": "patient",
        "evidence_refs": [{"segment_id": "s0001", "quote": "headache"}],
    }
    cand = ProviderCandidate.model_validate(valid_data)
    assert cand.label_raw == "headache"

    # Inject forbidden extra field (e.g. coordinates or confidence)
    invalid_data = dict(valid_data)
    invalid_data["confidence"] = 0.95
    with pytest.raises(ValidationError):
        ProviderCandidate.model_validate(invalid_data)


# ---------------------------------------------------------------------------
# 4. AgentResult Round-Trip
# ---------------------------------------------------------------------------


def test_agent_result_round_trip():
    req_id = uuid4()
    audit = AuditMetadata(
        provider="unavailable",
        model_id="unavailable",
        schema_enforcement="none",
        prompt_version="1.0.0",
        prompt_sha256="dummy_sha",
        schema_version="1.0.0",
    )
    result = AgentResult(
        request_id=req_id,
        agent_name="text_agent",
        source_type="text",
        status="success",
        sanitized_source_metadata={"source_id": "src_1234"},
        content_sha256="dummy_content_sha",
        candidates=[],
        stage_results={
            "source_reading": StageResult(
                stage_name="source_reading", status="success", counts={"segments": 1}
            )
        },
        warnings=[],
        errors=[],
        audit=audit,
        created_at_utc=datetime.now(UTC),
        duration_ms=1.5,
    )

    json_str = result.model_dump_json()
    reloaded = AgentResult.model_validate_json(json_str)
    assert reloaded.request_id == req_id
    assert reloaded.status == "success"
    assert reloaded.stage_results["source_reading"].counts["segments"] == 1
