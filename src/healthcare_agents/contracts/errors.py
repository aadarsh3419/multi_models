"""
Error codes, warning codes, and structured error schemas.

Zero provider or agent dependencies.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Account-wide, request-level, and stage error codes."""

    INVALID_INPUT = "invalid_input"
    NO_USABLE_SOURCE_CONTENT = "no_usable_source_content"
    REMOTE_PROCESSING_DISABLED = "remote_processing_disabled"
    MISSING_API_KEY = "missing_api_key"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_AUTHENTICATION_FAILED = "provider_authentication_failed"
    PROVIDER_MODEL_UNAVAILABLE = "provider_model_unavailable"
    PROVIDER_SCHEMA_UNSUPPORTED = "provider_schema_unsupported"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"
    LLM_CALL_BUDGET_EXCEEDED = "llm_call_budget_exceeded"

    # Milestone 2: PDF & OCR Error Codes
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    FILE_TOO_LARGE = "file_too_large"
    PDF_UNREADABLE = "pdf_unreadable"
    PDF_ENCRYPTED = "pdf_encrypted"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    OCR_UNAVAILABLE = "ocr_unavailable"
    OCR_TIMEOUT = "ocr_timeout"
    IMAGE_UNREADABLE = "image_unreadable"

    # Milestone 3: Image Agent Error Codes
    IMAGE_PIXEL_LIMIT_EXCEEDED = "image_pixel_limit_exceeded"

    # Milestone 4: Voice Agent Error Codes
    AUDIO_UNREADABLE = "audio_unreadable"
    AUDIO_DURATION_EXCEEDED = "audio_duration_exceeded"
    AUDIO_DURATION_UNAVAILABLE = "audio_duration_unavailable"
    AUDIO_VALIDATION_UNAVAILABLE = "audio_validation_unavailable"
    AUDIO_TRANSCRIPTION_FAILED = "audio_transcription_failed"


class WarningCode(str, Enum):
    """Candidate-level and chunk-level non-fatal warning codes."""

    EVIDENCE_REFERENCE_UNKNOWN = "evidence_reference_unknown"
    EVIDENCE_NOT_FOUND = "evidence_not_found"
    EVIDENCE_FIELD_MISMATCH = "evidence_field_mismatch"
    EVIDENCE_AMBIGUOUS = "evidence_ambiguous"
    POTENTIAL_CONFLICT = "potential_conflict"

    # Milestone 2: PDF & OCR Warning Codes
    OCR_SKIPPED_RENDER_BUDGET = "ocr_skipped_render_budget"
    PAGE_TEXT_DEFICIENT_OCR_DISABLED = "page_text_deficient_ocr_disabled"
    TABLE_STRUCTURE_UNCERTAIN = "table_structure_uncertain"
    POSSIBLE_NATIVE_OCR_DUPLICATE = "possible_native_ocr_duplicate"
    PAGE_PROCESSING_FAILED = "page_processing_failed"

    # Milestone 3: Image Agent Warning Codes
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"

    # Milestone 4: Voice Agent Warning Codes
    TRANSCRIPT_LANGUAGE_MISMATCH = "transcript_language_mismatch"
    TIMESTAMPS_UNAVAILABLE = "timestamps_unavailable"


class StructuredError(BaseModel):
    """Structured, PHI-free diagnostic record for errors and warnings."""

    code: str = Field(description="Normalized error or warning code string.")
    stage: str = Field(description="Pipeline stage in which the issue was detected.")
    message: str = Field(description="Static, non-PHI human-readable description.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured non-PHI diagnostic parameters (e.g. unit_id, counts).",
    )

    def __contains__(self, item: str) -> bool:
        return item in self.message or item in self.code

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
