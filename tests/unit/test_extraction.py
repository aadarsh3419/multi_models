"""
Unit tests for Checkpoint B: EvidenceValidator, ValueParser, DateParser,
CandidateDeduplicator, and ExtractionWorkflow with fake extractors.
"""

from decimal import Decimal
from typing import Any
from uuid import uuid4

from healthcare_agents.contracts.errors import ErrorCode, WarningCode
from healthcare_agents.contracts.facts import (
    AssertionStatus,
    DateParseStatus,
    EvidenceStatus,
    ExperiencerType,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    Temporality,
)
from healthcare_agents.contracts.locations import TextLocation
from healthcare_agents.contracts.parsing import ParseStatus
from healthcare_agents.contracts.source_text import ProviderUnit, SourceSegment
from healthcare_agents.extraction.deduplication import CandidateDeduplicator
from healthcare_agents.extraction.evidence_validator import EvidenceValidator
from healthcare_agents.extraction.field_extractor import (
    ClinicalFieldExtractor,
    ExtractorUnavailable,
)
from healthcare_agents.extraction.value_parser import DateParser, ValueParser
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.sources.text_reader import TextReader

# ---------------------------------------------------------------------------
# 1. ValueParser Tests
# ---------------------------------------------------------------------------


def test_value_parser_scalars():
    parser = ValueParser()
    res = parser.parse_value("72", "Heart rate")
    assert res.parse_status == ParseStatus.PARSED
    assert res.numeric_value == Decimal("72")
    assert res.value_kind == "number"

    res_dec = parser.parse_value("-5.4", "Temperature delta")
    assert res_dec.parse_status == ParseStatus.PARSED
    assert res_dec.numeric_value == Decimal("-5.4")


def test_value_parser_comma_ambiguity():
    parser = ValueParser()
    res = parser.parse_value("1,50,000", "Platelets")
    assert res.parse_status == ParseStatus.AMBIGUOUS
    assert res.rule_id == "comma_ambiguity"

    res_euro = parser.parse_value("138,5", "Sodium")
    assert res_euro.parse_status == ParseStatus.AMBIGUOUS


def test_value_parser_bp_allowlist_gating():
    parser = ValueParser(bp_labels=["bp", "blood pressure"])

    # Matches BP label
    res_bp = parser.parse_value("120/80", "Blood Pressure")
    assert res_bp.parse_status == ParseStatus.PARSED
    assert res_bp.systolic == 120
    assert res_bp.diastolic == 80
    assert res_bp.value_kind == "blood_pressure"

    # Not in BP label list (e.g. Vision ratio 20/20)
    res_other = parser.parse_value("20/20", "Visual Acuity")
    assert res_other.parse_status == ParseStatus.AMBIGUOUS
    assert res_other.rule_id == "bp_label_not_allowlisted"


def test_value_parser_ranges():
    parser = ValueParser()

    # Two-sided
    res = parser.parse_reference_range("3.5-5.0")
    assert res.parse_status == ParseStatus.PARSED
    assert res.range_low == Decimal("3.5")
    assert res.range_high == Decimal("5.0")

    # Inverted range (A > B) -> Invalid
    res_inv = parser.parse_reference_range("10.0-5.0")
    assert res_inv.parse_status == ParseStatus.INVALID

    # One-sided
    res_lt = parser.parse_reference_range("<100")
    assert res_lt.parse_status == ParseStatus.PARSED
    assert res_lt.range_high == Decimal("100")
    assert res_lt.range_high_inclusive is False


# ---------------------------------------------------------------------------
# 2. DateParser Tests
# ---------------------------------------------------------------------------


def test_date_parser_iso():
    date_val, status, rule = DateParser.parse_date("2026-03-15", "UNKNOWN")
    assert status == DateParseStatus.PARSED
    assert date_val == "2026-03-15"

    # Invalid calendar date (Feb 30)
    _, status_inv, _ = DateParser.parse_date("2025-02-30", "UNKNOWN")
    assert status_inv == DateParseStatus.INVALID


def test_date_parser_date_orders():
    # Ambiguous date 05/06/2025
    _, status_unk, _ = DateParser.parse_date("05/06/2025", "UNKNOWN")
    assert status_unk == DateParseStatus.AMBIGUOUS

    # DMY: 05/06/2025 -> June 5, 2025
    date_dmy, status_dmy, _ = DateParser.parse_date("05/06/2025", "DMY")
    assert status_dmy == DateParseStatus.PARSED
    assert date_dmy == "2025-06-05"

    # MDY: 05/06/2025 -> May 6, 2025
    date_mdy, status_mdy, _ = DateParser.parse_date("05/06/2025", "MDY")
    assert status_mdy == DateParseStatus.PARSED
    assert date_mdy == "2025-05-06"


# ---------------------------------------------------------------------------
# 3. EvidenceValidator Tests
# ---------------------------------------------------------------------------


def test_evidence_validator_exact_and_field_containment():
    text = "Vitals: Heart rate 72 bpm sitting."
    unit = ProviderUnit(
        unit_id="s0001", text=text, parent_segment_id="s0001", parent_start=0, parent_end=len(text)
    )
    seg = SourceSegment(
        segment_id="s0001",
        text=text,
        extraction_method="direct",
        source_location=TextLocation(start=0, end=len(text)),
    )
    validator = EvidenceValidator()

    # Valid candidate
    cand = ProviderCandidate(
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
        evidence_refs=[ProviderEvidenceRef(segment_id="s0001", quote="Heart rate 72 bpm")],
    )

    accepted, warnings = validator.validate_candidates([cand], unit, seg, "dummy_hash")
    assert len(accepted) == 1
    assert len(warnings) == 0
    assert accepted[0].evidence_status == EvidenceStatus.MATCHED
    assert isinstance(accepted[0].evidence_records[0].location, TextLocation)
    assert accepted[0].evidence_records[0].location.start == 8
    assert accepted[0].evidence_records[0].location.end == 25

    # Field mismatch: label fabricated that does not exist in quote
    cand_mismatch = cand.model_copy(update={"label_raw": "Blood Pressure"})
    rejected, warn_mismatch = validator.validate_candidates(
        [cand_mismatch], unit, seg, "dummy_hash"
    )
    assert len(rejected) == 0
    assert any(w.code == WarningCode.EVIDENCE_FIELD_MISMATCH.value for w in warn_mismatch)


def test_evidence_validator_repeated_quote_marked_ambiguous():
    text = "Vitals check 1: BP 120/80 mmHg. Vitals check 2: BP 120/80 mmHg."
    unit = ProviderUnit(
        unit_id="s0001", text=text, parent_segment_id="s0001", parent_start=0, parent_end=len(text)
    )
    seg = SourceSegment(
        segment_id="s0001",
        text=text,
        extraction_method="direct",
        source_location=TextLocation(start=0, end=len(text)),
    )
    validator = EvidenceValidator()

    cand = ProviderCandidate(
        fact_type=FactType.MEASUREMENT,
        label_raw="BP",
        value_raw="120/80",
        unit_raw="mmHg",
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
        evidence_refs=[ProviderEvidenceRef(segment_id="s0001", quote="BP 120/80 mmHg")],
    )

    accepted, warnings = validator.validate_candidates([cand], unit, seg, "dummy_hash")
    assert len(accepted) == 1
    # Repeated quote marked ambiguous without guessing first occurrence
    assert accepted[0].evidence_status == EvidenceStatus.AMBIGUOUS_MATCH
    assert accepted[0].evidence_records[0].location is None
    assert accepted[0].evidence_records[0].match_count == 2


# ---------------------------------------------------------------------------
# 4. CandidateDeduplicator & Conflict Tests
# ---------------------------------------------------------------------------


def test_deduplicator_exact_match_and_spans():
    text = "Morning note: BP 120/80 mmHg.\n\nEvening note: BP 120/80 mmHg."
    bundle = TextReader.read_bundle(text, uuid4())

    validator = EvidenceValidator()
    cand1 = ProviderCandidate(
        fact_type=FactType.MEASUREMENT,
        label_raw="BP",
        value_raw="120/80",
        unit_raw="mmHg",
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
        evidence_refs=[ProviderEvidenceRef(segment_id="s0001", quote="BP 120/80 mmHg")],
    )
    cand2 = ProviderCandidate(
        fact_type=FactType.MEASUREMENT,
        label_raw="BP",
        value_raw="120/80",
        unit_raw="mmHg",
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
        evidence_refs=[ProviderEvidenceRef(segment_id="s0002", quote="BP 120/80 mmHg")],
    )

    unit1 = ProviderUnit(
        unit_id="s0001",
        text=bundle.ordered_segments[0].text,
        parent_segment_id="s0001",
        parent_start=0,
        parent_end=len(bundle.ordered_segments[0].text),
    )
    unit2 = ProviderUnit(
        unit_id="s0002",
        text=bundle.ordered_segments[1].text,
        parent_segment_id="s0002",
        parent_start=0,
        parent_end=len(bundle.ordered_segments[1].text),
    )

    facts1, _ = validator.validate_candidates([cand1], unit1, bundle.ordered_segments[0], "hash")
    facts2, _ = validator.validate_candidates([cand2], unit2, bundle.ordered_segments[1], "hash")

    # Identical facts at DIFFERENT physical source spans must stay separate!
    deduped, _ = CandidateDeduplicator.deduplicate_and_detect_conflicts(facts1 + facts2)
    assert len(deduped) == 2

    # If the EXACT same candidate is fed twice (same span), it deduplicates to 1
    deduped_same, _ = CandidateDeduplicator.deduplicate_and_detect_conflicts(facts1 + facts1)
    assert len(deduped_same) == 1


def test_conflict_detection_differing_values():
    text = "Fasting glucose 95 mg/dL on 2026-03-15. Repeat fasting glucose 145 mg/dL on 2026-03-15."
    bundle = TextReader.read_bundle(text, uuid4())

    validator = EvidenceValidator()
    cand1 = ProviderCandidate(
        fact_type=FactType.MEASUREMENT,
        label_raw="Fasting glucose",
        value_raw="95",
        unit_raw="mg/dL",
        reference_range_raw="",
        abnormal_flag_raw="",
        date_raw="2026-03-15",
        time_raw="",
        qualifier_raw="",
        assertion_status=AssertionStatus.AFFIRMED,
        temporality=Temporality.NOT_APPLICABLE,
        experiencer_type=ExperiencerType.PATIENT,
        experiencer_relationship_raw="",
        reporting_source_role=ReportingSourceRole.LABORATORY,
        evidence_refs=[
            ProviderEvidenceRef(segment_id="s0001", quote="Fasting glucose 95 mg/dL on 2026-03-15")
        ],
    )
    cand2 = ProviderCandidate(
        fact_type=FactType.MEASUREMENT,
        label_raw="Fasting glucose",
        value_raw="145",
        unit_raw="mg/dL",
        reference_range_raw="",
        abnormal_flag_raw="",
        date_raw="2026-03-15",
        time_raw="",
        qualifier_raw="",
        assertion_status=AssertionStatus.AFFIRMED,
        temporality=Temporality.NOT_APPLICABLE,
        experiencer_type=ExperiencerType.PATIENT,
        experiencer_relationship_raw="",
        reporting_source_role=ReportingSourceRole.LABORATORY,
        evidence_refs=[
            ProviderEvidenceRef(
                segment_id="s0001", quote="Repeat fasting glucose 145 mg/dL on 2026-03-15"
            )
        ],
    )

    unit = ProviderUnit(
        unit_id="s0001",
        text=bundle.ordered_segments[0].text,
        parent_segment_id="s0001",
        parent_start=0,
        parent_end=len(text),
    )
    facts, _ = validator.validate_candidates(
        [cand1, cand2], unit, bundle.ordered_segments[0], "hash"
    )
    deduped, warnings = CandidateDeduplicator.deduplicate_and_detect_conflicts(facts)

    assert len(deduped) == 2
    # Defensible conflict: same label, same date, differing values -> potential_conflict warning added
    assert any(w.code == WarningCode.POTENTIAL_CONFLICT.value for w in warnings)


# ---------------------------------------------------------------------------
# 5. ExtractionWorkflow Tests with Fake Extractors
# ---------------------------------------------------------------------------


class FakeDeterministicExtractor(ClinicalFieldExtractor):
    def __init__(self, candidates_per_unit: dict[str, list[ProviderCandidate]] | None = None):
        self.candidates_per_unit = candidates_per_unit or {}
        self.call_count = 0

    def extract(
        self, unit: ProviderUnit, language_hint: str | None = None
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        self.call_count += 1
        cands = self.candidates_per_unit.get(unit.unit_id, [])
        return ProviderExtractionResponse(candidates=cands), {
            "provider": "fake_extractor",
            "model_id": "test_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "fake_sha",
            "attempt_count": 1,
            "token_usage": {"total_tokens": 100},
        }


def test_workflow_successful_execution():
    text = "Vitals: Heart rate 72 bpm."
    bundle = TextReader.read_bundle(text, uuid4())

    cand = ProviderCandidate(
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
        evidence_refs=[ProviderEvidenceRef(segment_id="s0001", quote="Heart rate 72 bpm")],
    )

    extractor = FakeDeterministicExtractor({"s0001": [cand]})
    workflow = ExtractionWorkflow(extractor=extractor)

    result = workflow.process_bundle(bundle)
    assert result.status == "success"
    assert len(result.candidates) == 1
    assert result.candidates[0].value_parse_result.numeric_value == Decimal("72")
    assert result.stage_results["clinical_extraction"].status == "success"


def test_workflow_call_budget_exhaustion():
    text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
    bundle = TextReader.read_bundle(text, uuid4())
    extractor = FakeDeterministicExtractor()

    # Limit budget to 2 calls
    workflow = ExtractionWorkflow(extractor=extractor, max_llm_calls_per_request=2)
    result = workflow.process_bundle(bundle)

    assert result.status == "partial"
    assert extractor.call_count == 2
    assert any(w.code == ErrorCode.LLM_CALL_BUDGET_EXCEEDED.value for w in result.warnings)


def test_workflow_extractor_unavailable_stops_immediately():
    text = "Vital check: BP 120/80 mmHg."
    bundle = TextReader.read_bundle(text, uuid4())

    class UnavailableExtractor(ClinicalFieldExtractor):
        def extract(self, unit: ProviderUnit, language_hint: str | None = None):
            raise ExtractorUnavailable(
                ErrorCode.REMOTE_PROCESSING_DISABLED.value, "Remote LLM processing is disabled."
            )

    workflow = ExtractionWorkflow(extractor=UnavailableExtractor())
    result = workflow.process_bundle(bundle)

    assert result.status == "partial"
    assert any(e.code == ErrorCode.REMOTE_PROCESSING_DISABLED.value for e in result.errors)
    assert result.stage_results["source_reading"].status == "success"
