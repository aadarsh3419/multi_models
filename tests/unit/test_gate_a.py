"""Unit tests for Gate A — Contracts, segmentation, parsers, and evidence validation."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from healthcare_agents.contracts import (
    AssertionStatus,
    EvidenceReference,
    EvidenceStatus,
    ExperiencerType,
    FactCandidate,
    FactType,
    ParseStatus,
    ReportingSourceRole,
    Temporality,
    TextAgentOptions,
    TextAgentRequest,
)
from healthcare_agents.deduplicator import (
    CandidateDeduplicator,
    generate_candidate_id,
)
from healthcare_agents.evidence_validator import EvidenceValidator
from healthcare_agents.source_text import (
    create_source_text_bundle,
)
from healthcare_agents.value_parser import DateParser, ValueParser

# ==============================================================================
# 1. Request Contracts and PHI Safety Tests
# ==============================================================================


def test_text_agent_request_valid():
    """Verify standard valid request initialization."""
    req = TextAgentRequest(text="Patient presents with mild fever.")
    assert req.text == "Patient presents with mild fever."
    assert req.request_id is not None
    assert req.options.max_text_chars == 36000


def test_text_agent_request_size_limit():
    """Verify that text exceeding configured max size raises a clean ValueError."""
    options = TextAgentOptions(max_text_chars=100)
    huge_text = "A" * 101
    with pytest.raises(ValidationError) as exc_info:
        TextAgentRequest(text=huge_text, options=options)
    err_str = str(exc_info.value)
    assert "Input text exceeds maximum allowed size" in err_str
    # Verify that the entire 101 characters are not dumped repeatedly in error message
    assert len(err_str) < 500


def test_text_agent_request_empty_rejection():
    """Verify empty and whitespace-only text is rejected."""
    with pytest.raises(ValidationError):
        TextAgentRequest(text="")

    with pytest.raises(ValidationError):
        TextAgentRequest(text="   \n\t  ")


def test_text_agent_request_phi_safe_repr():
    """Verify that __repr__ and __str__ do not expose raw clinical text."""
    secret_text = "Patient John Doe SSN 000-11-2222 diagnosed with Stage 4 Cancer."
    req = TextAgentRequest(text=secret_text)
    repr_str = repr(req)
    str_str = str(req)

    assert "John Doe" not in repr_str
    assert "Cancer" not in repr_str
    assert "chars=" in repr_str
    assert "John Doe" not in str_str


# ==============================================================================
# 2. SourceTextBundle and Lossless Offset Tests
# ==============================================================================


def test_source_text_bundle_exact_offsets():
    """Verify that text[seg.start_offset:seg.end_offset] matches seg.exact_text exactly."""
    raw_text = (
        "Patient admitted on 2025-04-03.\n\n"
        "Vitals: BP 120/80 mmHg, HR 72 bpm.\n\n"
        "Assessment: Type 2 Diabetes Mellitus."
    )
    req_id = uuid4()
    bundle = create_source_text_bundle(raw_text, req_id)

    assert len(bundle.segments) == 3
    for seg in bundle.segments:
        assert raw_text[seg.start_offset : seg.end_offset] == seg.exact_text
        assert seg.segment_id.startswith("txt_")


def test_source_text_bundle_crlf_and_lf():
    """Verify handling of CRLF Windows line endings and LF Unix endings without offset drift."""
    crlf_text = "Paragraph One.\r\n\r\nParagraph Two.\r\n\r\nParagraph Three."
    req_id = uuid4()
    bundle = create_source_text_bundle(crlf_text, req_id)

    assert len(bundle.segments) == 3
    for seg in bundle.segments:
        assert crlf_text[seg.start_offset : seg.end_offset] == seg.exact_text


def test_source_text_bundle_hindi_unicode_offsets():
    """Verify Unicode code point counting for Hindi text with combining matras."""
    hindi_text = "रोगी को तेज बुखार है।\n\nरक्तचाप 120/80 mmHg मापा गया।"
    req_id = uuid4()
    bundle = create_source_text_bundle(hindi_text, req_id)

    assert len(bundle.segments) == 2
    for seg in bundle.segments:
        assert hindi_text[seg.start_offset : seg.end_offset] == seg.exact_text


# ==============================================================================
# 3. ValueParser Unit Tests
# ==============================================================================


def test_value_parser_scalars():
    """Verify scalar number parsing."""
    res = ValueParser.parse("140.5")
    assert res.parse_status == ParseStatus.parsed
    assert res.value_kind == "scalar"
    assert res.parsed_scalar == 140.5

    res_neg = ValueParser.parse("-2.5")
    assert res_neg.parse_status == ParseStatus.parsed
    assert res_neg.parsed_scalar == -2.5


def test_value_parser_comparators():
    """Verify comparison operator values (<, <=, >, >=)."""
    res = ValueParser.parse("< 0.05")
    assert res.parse_status == ParseStatus.parsed
    assert res.comparator == "<"
    assert res.parsed_scalar == 0.05

    res_gte = ValueParser.parse("≥ 10.0")
    assert res_gte.parse_status == ParseStatus.parsed
    assert res_gte.comparator == ">="
    assert res_gte.parsed_scalar == 10.0


def test_value_parser_ranges():
    """Verify range parsing with various dashes."""
    for dash in ["-", "to", "–", "—"]:
        raw = f"4.0 {dash} 5.5"
        res = ValueParser.parse(raw)
        assert res.parse_status == ParseStatus.parsed
        assert res.value_kind == "range"
        assert res.range_lower == 4.0
        assert res.range_upper == 5.5


def test_value_parser_blood_pressure_allowlist():
    """Verify blood pressure parsing occurs only when context is allowlisted."""
    # With valid BP context
    res_bp = ValueParser.parse("120/80", field_context="Blood Pressure")
    assert res_bp.parse_status == ParseStatus.parsed
    assert res_bp.value_kind == "ratio"
    assert res_bp.ratio_numerator == 120.0
    assert res_bp.ratio_denominator == 80.0
    assert res_bp.rule_identifier == "blood_pressure_ratio"

    # Without BP context: MUST NOT be inferred as BP
    res_generic = ValueParser.parse("120/80", field_context="Albumin/Globulin")
    assert res_generic.parse_status == ParseStatus.ambiguous
    assert res_generic.rule_identifier == "non_bp_slash_ratio"
    assert "without blood pressure context allowlist" in (res_generic.safe_warning or "")


def test_value_parser_ambiguous_commas():
    """Verify numbers with commas are flagged ambiguous without trusted locale rule."""
    res1 = ValueParser.parse("1,5")
    assert res1.parse_status == ParseStatus.ambiguous
    assert res1.rule_identifier == "ambiguous_comma_format"

    res2 = ValueParser.parse("1,000")
    assert res2.parse_status == ParseStatus.ambiguous


def test_value_parser_empty_and_text():
    """Verify empty values and non-numeric text results."""
    res_empty = ValueParser.parse(None)
    assert res_empty.parse_status == ParseStatus.not_stated

    res_text = ValueParser.parse("positive")
    assert res_text.parse_status == ParseStatus.not_applicable
    assert res_text.value_kind == "text"


# ==============================================================================
# 4. DateParser Unit Tests
# ==============================================================================


def test_date_parser_iso_standard():
    """Verify ISO 8601 YYYY-MM-DD parsing."""
    res = DateParser.parse("2025-04-03")
    assert res.parse_status == ParseStatus.parsed
    assert res.parsed_iso_date == "2025-04-03"
    assert res.applied_date_format_rule == "iso_ymd"


def test_date_parser_invalid_calendar_date():
    """Verify invalid calendar dates are marked invalid rather than silently corrected."""
    res = DateParser.parse("2025-02-31")
    assert res.parse_status == ParseStatus.invalid
    assert "Invalid ISO calendar date" in (res.safe_warning or "")


def test_date_parser_unambiguous_day_gt_12():
    """Verify day > 12 is unambiguously resolved even with DATE_ORDER=UNKNOWN."""
    res = DateParser.parse("25/04/2025", date_order="UNKNOWN")
    assert res.parse_status == ParseStatus.parsed
    assert res.parsed_iso_date == "2025-04-25"
    assert res.applied_date_format_rule == "dmy_unambiguous_day_gt_12"


def test_date_parser_ambiguous_date_under_unknown():
    """Verify ambiguous date 03/04/2025 remains ambiguous when order is UNKNOWN."""
    res = DateParser.parse("03/04/2025", date_order="UNKNOWN")
    assert res.parse_status == ParseStatus.ambiguous
    assert "is ambiguous between DMY and MDY" in (res.safe_warning or "")


def test_date_parser_configured_order():
    """Verify resolution of ambiguous date when DATE_ORDER is configured."""
    res_dmy = DateParser.parse("03/04/2025", date_order="DMY")
    assert res_dmy.parse_status == ParseStatus.parsed
    assert res_dmy.parsed_iso_date == "2025-04-03"

    res_mdy = DateParser.parse("03/04/2025", date_order="MDY")
    assert res_mdy.parse_status == ParseStatus.parsed
    assert res_mdy.parsed_iso_date == "2025-03-04"


def test_date_parser_missing_year():
    """Verify missing years are never guessed."""
    res = DateParser.parse("04/03")
    assert res.parse_status == ParseStatus.invalid
    assert "missing year" in (res.safe_warning or "")


# ==============================================================================
# 5. EvidenceValidator Unit Tests
# ==============================================================================


def test_evidence_validator_exact_match():
    """Verify exact quote match maps to precise original source offsets."""
    text = "Initial report: Patient HbA1c is 7.2% recorded today."
    bundle = create_source_text_bundle(text, uuid4())
    seg = bundle.segments[0]

    ref = EvidenceReference(segment_id=seg.segment_id, quote="HbA1c is 7.2%")
    rec = EvidenceValidator.validate_single_reference(ref, bundle)

    assert rec.status == EvidenceStatus.matched
    assert rec.start_offset is not None
    assert rec.end_offset is not None
    assert text[rec.start_offset : rec.end_offset] == "HbA1c is 7.2%"


def test_evidence_validator_duplicate_quote_marked_ambiguous():
    """Verify duplicate quote occurrences in same segment are marked ambiguous (not arbitrarily chosen)."""
    text = "Vitals: BP was 120/80 in morning. Later BP was 120/80 in evening."
    bundle = create_source_text_bundle(text, uuid4())
    seg = bundle.segments[0]

    ref = EvidenceReference(segment_id=seg.segment_id, quote="120/80")
    rec = EvidenceValidator.validate_single_reference(ref, bundle)

    assert rec.status == EvidenceStatus.ambiguous_match
    assert rec.start_offset is None
    assert "duplicate_quote_occurrences" in (rec.safe_warning or "")


def test_evidence_validator_whitespace_normalized_fallback():
    """Verify whitespace fallback maps correctly to original untouched offsets."""
    text = "Lab test:    Fasting    Glucose    110 mg/dL   measured."
    bundle = create_source_text_bundle(text, uuid4())
    seg = bundle.segments[0]

    # Model returned single-spaced quote
    ref = EvidenceReference(segment_id=seg.segment_id, quote="Fasting Glucose 110 mg/dL")
    rec = EvidenceValidator.validate_single_reference(ref, bundle)

    assert rec.status == EvidenceStatus.matched
    assert rec.start_offset is not None
    assert rec.end_offset is not None
    # Slicing original text gives the exact multi-spaced span
    assert text[rec.start_offset : rec.end_offset] == "Fasting    Glucose    110 mg/dL"


def test_evidence_validator_unknown_segment_and_not_found():
    """Verify unknown segment ID and missing quote handling."""
    text = "Routine examination."
    bundle = create_source_text_bundle(text, uuid4())

    ref_bad_seg = EvidenceReference(segment_id="txt_999999", quote="Routine")
    rec1 = EvidenceValidator.validate_single_reference(ref_bad_seg, bundle)
    assert rec1.status == EvidenceStatus.unmatched
    assert "evidence_reference_unknown" in (rec1.safe_warning or "")

    ref_bad_quote = EvidenceReference(
        segment_id=bundle.segments[0].segment_id, quote="Hypertension"
    )
    rec2 = EvidenceValidator.validate_single_reference(ref_bad_quote, bundle)
    assert rec2.status == EvidenceStatus.unmatched
    assert "evidence_not_found" in (rec2.safe_warning or "")


# ==============================================================================
# 6. Candidate Deduplication and Conflict Detection Tests
# ==============================================================================


def test_candidate_id_deterministic_and_unique():
    """Verify deterministic candidate IDs: same coordinates -> same ID, different coordinates -> different ID."""
    source_hash = "abc123hash"
    spans_a = [("txt_000001", 10, 25)]
    spans_b = [("txt_000001", 50, 65)]

    id_1 = generate_candidate_id(
        source_hash, "measurement", "HbA1c", "7.2", "%", "affirmed", "current", "patient", spans_a
    )
    id_1_again = generate_candidate_id(
        source_hash, "measurement", "HbA1c", "7.2", "%", "affirmed", "current", "patient", spans_a
    )
    id_2 = generate_candidate_id(
        source_hash, "measurement", "HbA1c", "7.2", "%", "affirmed", "current", "patient", spans_b
    )

    assert id_1 == id_1_again
    assert id_1 != id_2
    assert id_1.startswith("cnd_")


def test_candidate_deduplication_exact_match():
    """Verify exact duplicate candidate from overlapping chunks is deduplicated."""
    cand1 = FactCandidate(
        candidate_id="cnd_1",
        fact_type=FactType.measurement,
        label_raw="Blood Pressure",
        value_raw="120/80",
        unit_raw="mmHg",
        assertion_status=AssertionStatus.affirmed,
        temporality=Temporality.current,
        experiencer_type=ExperiencerType.patient,
        reporting_source_role=ReportingSourceRole.clinician,
    )
    # Identical candidate
    cand2 = cand1.model_copy()

    deduped, warnings = CandidateDeduplicator.deduplicate_and_detect_conflicts([cand1, cand2])
    assert len(deduped) == 1
    assert len(warnings) == 0


def test_candidate_conflict_detection_same_date_differing_values():
    """Verify conflicting values on same explicit date emit a conflict warning and retain both."""
    cand1 = FactCandidate(
        candidate_id="cnd_1",
        fact_type=FactType.measurement,
        label_raw="Blood Pressure",
        value_raw="120/80",
        unit_raw="mmHg",
        date_raw="2025-04-03",
        assertion_status=AssertionStatus.affirmed,
        temporality=Temporality.current,
        experiencer_type=ExperiencerType.patient,
        reporting_source_role=ReportingSourceRole.clinician,
    )
    cand2 = FactCandidate(
        candidate_id="cnd_2",
        fact_type=FactType.measurement,
        label_raw="Blood Pressure",
        value_raw="150/95",
        unit_raw="mmHg",
        date_raw="2025-04-03",
        assertion_status=AssertionStatus.affirmed,
        temporality=Temporality.current,
        experiencer_type=ExperiencerType.patient,
        reporting_source_role=ReportingSourceRole.clinician,
    )

    deduped, warnings = CandidateDeduplicator.deduplicate_and_detect_conflicts([cand1, cand2])
    assert len(deduped) == 2
    assert len(warnings) == 1
    assert "Conflict detected for 'Blood Pressure'" in warnings[0]
    assert "cnd_2" in deduped[0].conflict_warning_refs
    assert "cnd_1" in deduped[1].conflict_warning_refs
