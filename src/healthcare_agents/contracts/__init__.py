"""
Contracts package exposing domain schemas, enums, and location representations.
"""

from enum import Enum

from pydantic import BaseModel

from healthcare_agents.contracts.errors import ErrorCode, StructuredError, WarningCode
from healthcare_agents.contracts.facts import (
    PROVIDER_JSON_SCHEMA,
    AssertionStatus,
    DateParseStatus,
    EvidenceRecord,
    EvidenceStatus,
    ExperiencerType,
    FactCandidate,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    ReviewStatus,
    Temporality,
    compute_candidate_id,
)
from healthcare_agents.contracts.locations import (
    AudioLocation,
    ImageLocation,
    PdfLocation,
    SourceLocation,
    TextLocation,
)
from healthcare_agents.contracts.parsing import ParseStatus, ValueParseResult
from healthcare_agents.contracts.requests import (
    ImageAgentOptions,
    ImageAgentRequest,
    PDFAgentOptions,
    PDFAgentRequest,
    TextAgentOptions,
    TextAgentRequest,
    VoiceAgentOptions,
    VoiceAgentRequest,
)
from healthcare_agents.contracts.results import AgentResult, AuditMetadata, StageResult
from healthcare_agents.contracts.source_text import (
    ProviderUnit,
    SourceSegment,
    SourceTextBundle,
    SourceType,
)

__all__ = [
    "AgentResult",
    "AssertionStatus",
    "AudioLocation",
    "AuditMetadata",
    "DateParseStatus",
    "ErrorCode",
    "EvidenceRecord",
    "EvidenceStatus",
    "ExperiencerType",
    "FactCandidate",
    "FactType",
    "ImageAgentOptions",
    "ImageAgentRequest",
    "ImageLocation",
    "ParseStatus",
    "PdfLocation",
    "PDFAgentOptions",
    "PDFAgentRequest",
    "ProviderCandidate",
    "ProviderEvidenceRef",
    "ProviderExtractionResponse",
    "PROVIDER_JSON_SCHEMA",
    "ProviderUnit",
    "ReportingSourceRole",
    "ReviewStatus",
    "SourceLocation",
    "SourceSegment",
    "SourceTextBundle",
    "SourceType",
    "StageResult",
    "StructuredError",
    "Temporality",
    "TextAgentOptions",
    "TextAgentRequest",
    "TextLocation",
    "ValueParseResult",
    "VoiceAgentOptions",
    "VoiceAgentRequest",
    "WarningCode",
    "compute_candidate_id",
    # Backward compatibility aliases
    "EvidenceReference",
    "ExtractionResponse",
    "ExtractionCandidateRaw",
    "AgentStatus",
    "generate_candidate_id",
]

# Backward compatibility aliases and legacy models
from healthcare_agents.contracts.facts import generate_candidate_id

EvidenceReference = ProviderEvidenceRef
ExtractionResponse = ProviderExtractionResponse
ExtractionCandidateRaw = ProviderCandidate


class AgentStatus(str, Enum):
    SUCCESS = "success"
    success = "success"
    PARTIAL = "partial"
    partial = "partial"
    FAILED = "failed"
    failed = "failed"


class DateParseResult(BaseModel):
    raw_date: str
    parse_status: ParseStatus
    parsed_iso_date: str | None = None
    applied_date_format_rule: str | None = None
    safe_warning: str | None = None


class ValidatedEvidenceRecord(BaseModel):
    segment_id: str
    quote: str
    start_offset: int | None = None
    end_offset: int | None = None
    status: EvidenceStatus = EvidenceStatus.UNMATCHED
    safe_warning: str | None = None


class StageOutcome(BaseModel):
    stage: str
    status: str
    duration_ms: float = 0.0
    message: str | None = None
