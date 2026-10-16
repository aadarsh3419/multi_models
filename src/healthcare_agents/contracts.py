"""Shared request, result, and domain data contracts.

Enforces strict Pydantic v2 schemas for all pipeline stages.
Follows zero-provider dependency rule: no Groq SDK or agent imports are permitted here.
"""

import platform
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

import pydantic
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ==============================================================================
# Enumerations for Controlled Clinical Semantics
# ==============================================================================


class AssertionStatus(str, Enum):
    """Assertion level of the medical statement in the source text."""

    affirmed = "affirmed"
    negated = "negated"
    uncertain = "uncertain"


class Temporality(str, Enum):
    """Temporal frame of the extracted observation or intervention."""

    current = "current"
    past = "past"
    future = "future"
    recurrent = "recurrent"
    historical_unspecified = "historical_unspecified"
    uncertain = "uncertain"
    not_stated = "not_stated"
    not_applicable = "not_applicable"


class ExperiencerType(str, Enum):
    """Identifies the subject experiencing the condition or measurement."""

    patient = "patient"
    family_member = "family_member"
    other_person = "other_person"
    unknown = "unknown"
    not_applicable = "not_applicable"


class ReportingSourceRole(str, Enum):
    """The role of the entity communicating or recording the statement."""

    patient = "patient"
    caregiver = "caregiver"
    clinician = "clinician"
    laboratory = "laboratory"
    device = "device"
    document_author = "document_author"
    third_party = "third_party"
    unknown = "unknown"
    not_applicable = "not_applicable"


class EvidenceStatus(str, Enum):
    """Validation status of provider-cited evidence against source text."""

    matched = "matched"
    ambiguous_match = "ambiguous_match"
    unmatched = "unmatched"
    unavailable = "unavailable"


class ReviewStatus(str, Enum):
    """Human or clinical review status. Defaults to unreviewed."""

    unreviewed = "unreviewed"
    reviewed = "reviewed"
    rejected = "rejected"


class ParseStatus(str, Enum):
    """Status of deterministic parser execution on raw strings."""

    parsed = "parsed"
    ambiguous = "ambiguous"
    invalid = "invalid"
    not_attempted = "not_attempted"
    not_stated = "not_stated"
    not_applicable = "not_applicable"


class FactType(str, Enum):
    """Allowed medical fact categorization types."""

    measurement = "measurement"
    symptom = "symptom"
    condition = "condition"
    medication = "medication"
    allergy = "allergy"
    family_history = "family_history"
    procedure = "procedure"
    other = "other"


class AgentStatus(str, Enum):
    """Overall execution status of an agent workflow."""

    success = "success"
    partial = "partial"
    failed = "failed"


# ==============================================================================
# Request and Options Models
# ==============================================================================


class TextAgentOptions(BaseModel):
    """Processing options passed to the Text Agent."""

    model_config = ConfigDict(extra="forbid")

    max_text_chars: int = Field(default=36000, ge=1, le=100000)
    language_hint: str | None = Field(default=None, max_length=16)
    date_order: Literal["UNKNOWN", "DMY", "MDY"] = Field(default="UNKNOWN")
    enable_remote_llm: bool = Field(default=False)


class TextAgentRequest(BaseModel):
    """Input contract for the Text Agent with size gating and PHI-safe repr."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, repr=False)
    language_hint: str | None = Field(default=None, max_length=16)
    options: TextAgentOptions = Field(default_factory=TextAgentOptions)

    @model_validator(mode="after")
    def validate_text_size_and_content(self) -> "TextAgentRequest":
        # Check text is non-empty after stripping
        if not self.text.strip():
            raise ValueError("Input text must contain non-whitespace characters.")

        limit = self.options.max_text_chars
        # Offset and length calculations use Python Unicode code points (len of str)
        text_len = len(self.text)
        if text_len > limit:
            raise ValueError(
                f"Input text exceeds maximum allowed size ({text_len} code points > {limit})."
            )
        return self

    def __repr__(self) -> str:
        """Safe representation hiding raw clinical text to prevent log leaks."""
        return (
            f"TextAgentRequest(request_id={self.request_id}, "
            f"chars={len(self.text)}, language_hint={self.language_hint!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ==============================================================================
# Provider-Facing Schemas (Strict Structured Output Target)
# ==============================================================================


class EvidenceReference(BaseModel):
    """A quotation referenced by the LLM from a specific source segment."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    quote: str


class ExtractionCandidateRaw(BaseModel):
    """Strict candidate schema returned by the provider adapter."""

    model_config = ConfigDict(extra="forbid")

    fact_type: str
    label_raw: str
    value_raw: str | None = None
    unit_raw: str | None = None
    reference_range_raw: str | None = None
    abnormal_flag_raw: str | None = None
    date_raw: str | None = None
    time_raw: str | None = None
    qualifier_raw: str | None = None
    dose_raw: str | None = None
    frequency_raw: str | None = None
    route_raw: str | None = None
    duration_raw: str | None = None
    assertion_status: str
    temporality: str
    experiencer_type: str
    experiencer_relationship_raw: str | None = None
    reporting_source_role: str
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """The root container for candidates returned by the provider."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[ExtractionCandidateRaw] = Field(default_factory=list)


# ==============================================================================
# Parsed Value and Date Contracts
# ==============================================================================


class ValueParseResult(BaseModel):
    """Deterministic result of value parsing."""

    model_config = ConfigDict(extra="forbid")

    raw_value: str
    value_kind: Literal["scalar", "range", "ratio", "text", "unknown"]
    parse_status: ParseStatus
    parsed_scalar: float | None = None
    range_lower: float | None = None
    range_upper: float | None = None
    comparator: str | None = None
    ratio_numerator: float | None = None
    ratio_denominator: float | None = None
    rule_identifier: str | None = None
    safe_warning: str | None = None


class DateParseResult(BaseModel):
    """Deterministic result of date parsing."""

    model_config = ConfigDict(extra="forbid")

    raw_date: str
    parse_status: ParseStatus
    parsed_iso_date: str | None = None
    applied_date_format_rule: str | None = None
    safe_warning: str | None = None


class ValidatedEvidenceRecord(BaseModel):
    """Evidence quote matched and mapped back to original source offsets."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    quote: str
    start_offset: int | None = None
    end_offset: int | None = None
    status: EvidenceStatus = EvidenceStatus.unmatched
    safe_warning: str | None = None


# ==============================================================================
# Internal Validated Fact Candidate
# ==============================================================================


class FactCandidate(BaseModel):
    """Domain model of a validated, parsed, and audited clinical fact candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    fact_type: FactType
    label_raw: str
    value_raw: str | None = None
    unit_raw: str | None = None
    reference_range_raw: str | None = None
    abnormal_flag_raw: str | None = None
    date_raw: str | None = None
    time_raw: str | None = None
    qualifier_raw: str | None = None
    dose_raw: str | None = None
    frequency_raw: str | None = None
    route_raw: str | None = None
    duration_raw: str | None = None

    assertion_status: AssertionStatus
    temporality: Temporality
    experiencer_type: ExperiencerType
    experiencer_relationship_raw: str | None = None
    reporting_source_role: ReportingSourceRole

    evidence_refs: list[ValidatedEvidenceRecord] = Field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.unmatched

    value_parsed: ValueParseResult | None = None
    date_parsed: DateParseResult | None = None
    parse_status: ParseStatus = ParseStatus.not_attempted
    review_status: ReviewStatus = ReviewStatus.unreviewed
    conflict_warning_refs: list[str] = Field(default_factory=list)


# ==============================================================================
# Audit, Errors, Stage Outcomes, and Common AgentResult
# ==============================================================================


class StructuredError(BaseModel):
    """Machine-readable, safe error record without leaked PHI."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    stage: str | None = None
    retryable: bool = False


class StageOutcome(BaseModel):
    """Outcome report for an individual pipeline execution stage."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: Literal["success", "partial", "failed", "skipped"]
    duration_ms: float = 0.0
    message: str | None = None


class AuditMetadata(BaseModel):
    """Comprehensive provenance, configuration, and runtime environment audit record."""

    model_config = ConfigDict(extra="forbid")

    project_version: str = "0.1.0"
    schema_version: str = "1.0.0"
    request_id: str
    source_content_hash: str
    provider_name: str | None = None
    model_id: str | None = None
    prompt_version: str = "clinical-extraction-v1"
    prompt_template_hash: str | None = None
    provider_schema_version: str | None = "1.0.0"
    provider_response_id: str | None = None
    token_usage: dict[str, int] | None = None
    actual_provider_attempt_count: int = 0
    python_version: str = Field(default_factory=platform.python_version)
    pydantic_version: str = Field(default_factory=lambda: pydantic.__version__)
    groq_sdk_version: str | None = None
    processing_timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    configured_date_order: str = "UNKNOWN"
    stage_outcomes: dict[str, StageOutcome] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Universal top-level contract returned by all agents in the platform."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    request_id: UUID
    agent_name: str = "text"
    status: AgentStatus
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    candidates: list[FactCandidate] = Field(default_factory=list)
    stage_outcomes: dict[str, StageOutcome] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    audit_metadata: AuditMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
