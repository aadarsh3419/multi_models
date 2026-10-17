"""
Fact candidate models, enums, provider schemas, and candidate ID hashing.

Zero provider or agent dependencies.
"""

import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from healthcare_agents.contracts.locations import SourceLocation
from healthcare_agents.contracts.parsing import ParseStatus, ValueParseResult


class FactType(str, Enum):
    MEASUREMENT = "measurement"
    measurement = "measurement"
    SYMPTOM = "symptom"
    symptom = "symptom"
    MEDICATION = "medication"
    medication = "medication"
    ALLERGY = "allergy"
    allergy = "allergy"
    CONDITION = "condition"
    condition = "condition"
    HISTORY = "history"
    history = "history"
    FAMILY_HISTORY = "family_history"
    family_history = "family_history"
    PROCEDURE = "procedure"
    procedure = "procedure"
    OTHER = "other"
    other = "other"


class AssertionStatus(str, Enum):
    AFFIRMED = "affirmed"
    affirmed = "affirmed"
    NEGATED = "negated"
    negated = "negated"
    UNCERTAIN = "uncertain"
    uncertain = "uncertain"


class Temporality(str, Enum):
    CURRENT = "current"
    current = "current"
    PAST = "past"
    past = "past"
    FUTURE = "future"
    future = "future"
    RECURRENT = "recurrent"
    recurrent = "recurrent"
    HISTORICAL_UNSPECIFIED = "historical_unspecified"
    historical_unspecified = "historical_unspecified"
    UNCERTAIN = "uncertain"
    uncertain = "uncertain"
    NOT_STATED = "not_stated"
    not_stated = "not_stated"
    NOT_APPLICABLE = "not_applicable"
    not_applicable = "not_applicable"


class ExperiencerType(str, Enum):
    PATIENT = "patient"
    patient = "patient"
    FAMILY_MEMBER = "family_member"
    family_member = "family_member"
    OTHER_PERSON = "other_person"
    other_person = "other_person"
    UNKNOWN = "unknown"
    unknown = "unknown"
    NOT_APPLICABLE = "not_applicable"
    not_applicable = "not_applicable"


class ReportingSourceRole(str, Enum):
    PATIENT = "patient"
    patient = "patient"
    CAREGIVER = "caregiver"
    caregiver = "caregiver"
    CLINICIAN = "clinician"
    clinician = "clinician"
    LABORATORY = "laboratory"
    laboratory = "laboratory"
    DEVICE = "device"
    device = "device"
    DOCUMENT_AUTHOR = "document_author"
    document_author = "document_author"
    THIRD_PARTY = "third_party"
    third_party = "third_party"
    UNKNOWN = "unknown"
    unknown = "unknown"


class DateParseStatus(str, Enum):
    PARSED = "parsed"
    parsed = "parsed"
    AMBIGUOUS = "ambiguous"
    ambiguous = "ambiguous"
    INVALID = "invalid"
    invalid = "invalid"
    NOT_ATTEMPTED = "not_attempted"
    not_attempted = "not_attempted"
    NOT_STATED = "not_stated"
    not_stated = "not_stated"
    NOT_APPLICABLE = "not_applicable"
    not_applicable = "not_applicable"


class EvidenceStatus(str, Enum):
    MATCHED = "matched"
    matched = "matched"
    AMBIGUOUS_MATCH = "ambiguous_match"
    ambiguous_match = "ambiguous_match"
    UNMATCHED = "unmatched"
    unmatched = "unmatched"
    UNAVAILABLE = "unavailable"
    unavailable = "unavailable"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Provider-Facing Schema (Strict JSON Schema with extra="forbid")
# ---------------------------------------------------------------------------


class ProviderEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(description="Segment/unit identifier where quote was found.")
    quote: str = Field(description="Verbatim exact quotation from the unit text.")


class ProviderCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_type: FactType
    label_raw: str
    value_raw: str
    unit_raw: str
    reference_range_raw: str
    abnormal_flag_raw: str
    date_raw: str
    time_raw: str
    qualifier_raw: str
    assertion_status: AssertionStatus
    temporality: Temporality
    experiencer_type: ExperiencerType
    experiencer_relationship_raw: str
    reporting_source_role: ReportingSourceRole
    evidence_refs: list[ProviderEvidenceRef]

    @model_validator(mode="before")
    @classmethod
    def _fill_string_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Strip legacy medication fields if passed in older test fixtures/schemas
            for legacy_field in ("dose_raw", "frequency_raw", "route_raw", "duration_raw"):
                if legacy_field in data:
                    data.pop(legacy_field)
            defaults = {
                "value_raw": "",
                "unit_raw": "",
                "reference_range_raw": "",
                "abnormal_flag_raw": "",
                "date_raw": "",
                "time_raw": "",
                "qualifier_raw": "",
                "experiencer_relationship_raw": "",
            }
            for k, v in defaults.items():
                if k not in data or data[k] is None:
                    data[k] = v
        return data


class ProviderExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[ProviderCandidate]


# Hand-written JSON Schema strictly adhering to Groq structured output rules
PROVIDER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact_type": {
                        "type": "string",
                        "enum": [
                            "measurement",
                            "symptom",
                            "medication",
                            "allergy",
                            "condition",
                            "history",
                            "other",
                        ],
                    },
                    "label_raw": {"type": "string"},
                    "value_raw": {"type": "string"},
                    "unit_raw": {"type": "string"},
                    "reference_range_raw": {"type": "string"},
                    "abnormal_flag_raw": {"type": "string"},
                    "date_raw": {"type": "string"},
                    "time_raw": {"type": "string"},
                    "qualifier_raw": {"type": "string"},
                    "assertion_status": {
                        "type": "string",
                        "enum": ["affirmed", "negated", "uncertain"],
                    },
                    "temporality": {
                        "type": "string",
                        "enum": [
                            "current",
                            "past",
                            "future",
                            "recurrent",
                            "historical_unspecified",
                            "uncertain",
                            "not_stated",
                            "not_applicable",
                        ],
                    },
                    "experiencer_type": {
                        "type": "string",
                        "enum": [
                            "patient",
                            "family_member",
                            "other_person",
                            "unknown",
                            "not_applicable",
                        ],
                    },
                    "experiencer_relationship_raw": {"type": "string"},
                    "reporting_source_role": {
                        "type": "string",
                        "enum": [
                            "patient",
                            "caregiver",
                            "clinician",
                            "laboratory",
                            "device",
                            "document_author",
                            "third_party",
                            "unknown",
                        ],
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["segment_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "fact_type",
                    "label_raw",
                    "value_raw",
                    "unit_raw",
                    "reference_range_raw",
                    "abnormal_flag_raw",
                    "date_raw",
                    "time_raw",
                    "qualifier_raw",
                    "assertion_status",
                    "temporality",
                    "experiencer_type",
                    "experiencer_relationship_raw",
                    "reporting_source_role",
                    "evidence_refs",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Internal Rich Fact Candidate Schema
# ---------------------------------------------------------------------------


class EvidenceRecord(BaseModel):
    segment_id: str
    quote: str
    location: SourceLocation | None = None
    quote_char_span: tuple[int, int] | None = None
    match_count: int = 1
    status: EvidenceStatus = EvidenceStatus.MATCHED


class FactCandidate(BaseModel):
    """
    Internal grounded clinical fact candidate with evidence provenance,
    deterministic value parsing, and audit metadata.
    """

    candidate_id: str
    fact_type: FactType
    label_raw: str
    value_raw: str = ""
    unit_raw: str = ""
    reference_range_raw: str = ""
    abnormal_flag_raw: str = ""
    date_raw: str = ""
    time_raw: str = ""
    qualifier_raw: str = ""
    assertion_status: AssertionStatus
    temporality: Temporality
    experiencer_type: ExperiencerType
    experiencer_relationship_raw: str = ""
    reporting_source_role: ReportingSourceRole

    # Parsed and validated clinical attributes
    parsed_reference_range_low: Decimal | None = None
    parsed_reference_range_high: Decimal | None = None
    parsed_date: str | None = None  # ISO YYYY-MM-DD
    date_parse_status: DateParseStatus = DateParseStatus.NOT_ATTEMPTED
    value_parse_result: ValueParseResult | None = None

    # Evidence grounding
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.UNMATCHED
    parse_status: ParseStatus = ParseStatus.NOT_ATTEMPTED
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED

    # Provenance and context
    context: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def conflict_warning_refs(self) -> list[str]:
        return [w.split(":", 1)[1] for w in self.warnings if w.startswith("potential_conflict:")]

    @property
    def value_parsed(self) -> ValueParseResult | None:
        return self.value_parse_result

    @property
    def date_parsed(self) -> Any:
        if self.parsed_date:

            class _CompatDateParseResult:
                def __init__(self, raw: str, parsed: str):
                    self.raw_date = raw
                    self.parse_status = ParseStatus.PARSED
                    self.parsed_iso_date = parsed

            return _CompatDateParseResult(self.date_raw, self.parsed_date)
        return None


def compute_candidate_id(
    content_sha256: str,
    fact_type: FactType | str,
    normalized_label: str,
    raw_fields: dict[str, str],
    assertion: AssertionStatus | str,
    temporality: Temporality | str,
    experiencer: ExperiencerType | str,
    role: ReportingSourceRole | str,
    evidence_spans: list[tuple[str, int, int]] | list[tuple[str, str]],
) -> str:
    """
    Generates a deterministic candidate ID: 'fc_' + first 16 hex characters of SHA-256.
    Key depends on canonical JSON of content hash, normalized identity fields, and sorted evidence spans.
    """
    evidence_sorted = sorted(evidence_spans)
    identity_dict = {
        "content_sha256": content_sha256,
        "fact_type": str(fact_type),
        "normalized_label": normalized_label,
        "raw_fields": {k: v for k, v in sorted(raw_fields.items())},
        "assertion": str(assertion),
        "temporality": str(temporality),
        "experiencer": str(experiencer),
        "role": str(role),
        "evidence": evidence_sorted,
    }
    canonical_json = json.dumps(identity_dict, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return f"fc_{sha[:16]}"


def generate_candidate_id(
    source_hash: str,
    fact_type: str,
    label: str,
    value: str,
    unit: str,
    assertion: str,
    temporality: str,
    experiencer: str,
    spans: list,
) -> str:
    raw_fields = {"value": value, "unit": unit}
    fc_id = compute_candidate_id(
        content_sha256=source_hash,
        fact_type=fact_type,
        normalized_label=label.lower(),
        raw_fields=raw_fields,
        assertion=assertion,
        temporality=temporality,
        experiencer=experiencer,
        role="clinician",
        evidence_spans=spans,
    )
    return f"cnd_{fc_id[3:]}"
