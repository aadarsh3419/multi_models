"""
Request contracts for Text Agent.

Zero provider or agent dependencies.
"""

import hashlib
import re
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class TextAgentOptions(BaseModel):
    """Execution options for Text Agent requests."""

    include_source_text_in_result: bool = Field(
        default=False,
        description="Whether to echo original text in result. For local debugging only; never enable in production.",
    )
    date_order: Literal["UNKNOWN", "DMY", "MDY"] | None = Field(
        default=None,
        description="Optional date parsing order override.",
    )


class TextAgentRequest(BaseModel):
    """
    Validated ingestion request for unstructured/semi-structured clinical narratives.
    Enforces non-empty content and strictly redacts raw text in string/repr representations.
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique request identifier.",
    )
    text: str = Field(
        description="Raw, un-normalized clinical narrative.",
    )
    language_hint: str | None = Field(
        default=None,
        description="Optional ISO-639-1 language code hint (e.g. 'en', 'hi').",
    )
    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = Field(
        default=None,
        description="Optional per-request override for ambiguous calendar date parsing.",
    )
    options: TextAgentOptions = Field(
        default_factory=TextAgentOptions,
        description="Optional execution parameters.",
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text must not be empty or whitespace-only.")
        return v

    def __repr__(self) -> str:
        return (
            f"TextAgentRequest(request_id={self.request_id!r}, "
            f"text=[REDACTED {len(self.text)} chars], "
            f"language_hint={self.language_hint!r}, "
            f"date_order_hint={self.date_order_hint!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Milestone 2: PDF Agent Contracts
# ---------------------------------------------------------------------------

OCR_LANG_REGEX = re.compile(r"^[a-z]{3}(\+[a-z]{3})*$")


class PDFAgentOptions(BaseModel):
    """Execution options for PDF Agent requests."""

    ocr_mode: Literal["auto", "never", "always"] = Field(
        default="auto",
        description="OCR mode policy ('auto', 'never', 'always').",
    )
    ocr_languages: str = Field(
        default="eng",
        description="Tesseract language code (e.g. 'eng' or 'eng+hin'). Strictly validated against regex.",
    )
    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = Field(
        default=None,
        description="Optional per-request override for ambiguous calendar date parsing.",
    )
    include_source_text_in_result: bool = Field(
        default=False,
        description="Whether to echo extracted text in result. For local debugging only.",
    )

    @field_validator("ocr_languages")
    @classmethod
    def validate_ocr_languages(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not OCR_LANG_REGEX.match(v_clean):
            raise ValueError(
                f"Invalid ocr_languages format '{v}'. Must be 3-letter ISO-639-2 codes joined by '+' (e.g. 'eng', 'eng+hin')."
            )
        return v_clean


class PDFAgentRequest(BaseModel):
    """
    Validated ingestion request for native, scanned, or hybrid PDF clinical documents.
    Enforces non-empty content and strictly redacts raw bytes and source filename in repr/str/serialization.
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique request identifier.",
    )
    source_name: str | None = Field(
        default=None,
        description="Internal source filename or identifier. Redacted from logs and external repr.",
    )
    content: bytes = Field(
        exclude=True,
        description="Raw PDF file bytes. Strictly excluded from serialization, repr, and logs.",
    )
    options: PDFAgentOptions = Field(
        default_factory=PDFAgentOptions,
        description="Optional PDF processing and OCR parameters.",
    )

    @field_validator("content")
    @classmethod
    def validate_non_empty_content(cls, v: bytes) -> bytes:
        if not v or len(v) == 0:
            raise ValueError("PDF content bytes must not be empty.")
        return v

    def __repr__(self) -> str:
        content_sha = hashlib.sha256(self.content).hexdigest()[:16] if self.content else "empty"
        return (
            f"PDFAgentRequest(request_id={self.request_id!r}, "
            f"source_name=[REDACTED], "
            f"content=[REDACTED {len(self.content)} bytes, sha256={content_sha}...], "
            f"options={self.options!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Milestone 3: Image Agent Contracts
# ---------------------------------------------------------------------------


class ImageAgentOptions(BaseModel):
    """Execution options for Image Agent requests."""

    ocr_languages: str = Field(
        default="eng",
        description="Tesseract language code (e.g. 'eng' or 'eng+hin'). Strictly validated against regex.",
    )
    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = Field(
        default=None,
        description="Optional per-request override for ambiguous calendar date parsing.",
    )
    include_source_text_in_result: bool = Field(
        default=False,
        description="Whether to echo extracted text in result. For local debugging only.",
    )
    upscale_enabled: bool = Field(
        default=True,
        description="Whether to upscale small images (min side < image_upscale_min_side) prior to OCR.",
    )

    @field_validator("ocr_languages")
    @classmethod
    def validate_ocr_languages(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not OCR_LANG_REGEX.match(v_clean):
            raise ValueError(
                f"Invalid ocr_languages format '{v}'. Must be 3-letter ISO-639-2 codes joined by '+' (e.g. 'eng', 'eng+hin')."
            )
        return v_clean


class ImageAgentRequest(BaseModel):
    """
    Validated ingestion request for image documents (photos of reports, prescriptions, lab slips).
    Enforces non-empty content and strictly redacts raw bytes and source filename in repr/str/serialization.
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique request identifier.",
    )
    source_name: str | None = Field(
        default=None,
        description="Internal source filename or identifier. Redacted from logs and external repr.",
    )
    content: bytes = Field(
        exclude=True,
        description="Raw image file bytes. Strictly excluded from serialization, repr, and logs.",
    )
    options: ImageAgentOptions = Field(
        default_factory=ImageAgentOptions,
        description="Optional image processing and OCR parameters.",
    )

    @field_validator("content")
    @classmethod
    def validate_non_empty_content(cls, v: bytes) -> bytes:
        if not v or len(v) == 0:
            raise ValueError("Image content bytes must not be empty.")
        return v

    def __repr__(self) -> str:
        content_sha = hashlib.sha256(self.content).hexdigest()[:16] if self.content else "empty"
        return (
            f"ImageAgentRequest(request_id={self.request_id!r}, "
            f"source_name=[REDACTED], "
            f"content=[REDACTED {len(self.content)} bytes, sha256={content_sha}...], "
            f"options={self.options!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# Milestone 4: Voice Agent Contracts
# ---------------------------------------------------------------------------


class VoiceAgentOptions(BaseModel):
    """Execution options for Voice Agent requests."""

    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = Field(
        default=None,
        description="Optional per-request override for ambiguous calendar date parsing.",
    )
    include_source_text_in_result: bool = Field(
        default=False,
        description="Whether to echo transcribed text in result. For local debugging only.",
    )


class VoiceAgentRequest(BaseModel):
    """
    Validated ingestion request for clinical audio recordings (consultation dictations, telehealth audio).
    Enforces non-empty content and strictly redacts raw bytes and source filename in repr/str/serialization.
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique request identifier.",
    )
    source_name: str | None = Field(
        default=None,
        description="Internal source filename or identifier. Redacted from logs and external repr.",
    )
    content: bytes = Field(
        exclude=True,
        description="Raw audio file bytes. Strictly excluded from serialization, repr, and logs.",
    )
    language_hint: str | None = Field(
        default=None,
        description="Optional ISO-639-1 language code hint (e.g. 'en', 'hi').",
    )
    options: VoiceAgentOptions = Field(
        default_factory=VoiceAgentOptions,
        description="Optional audio processing and transcription parameters.",
    )

    @field_validator("content")
    @classmethod
    def validate_non_empty_content(cls, v: bytes) -> bytes:
        if not v or len(v) == 0:
            raise ValueError("Audio content bytes must not be empty.")
        return v

    def __repr__(self) -> str:
        content_sha = hashlib.sha256(self.content).hexdigest()[:16] if self.content else "empty"
        return (
            f"VoiceAgentRequest(request_id={self.request_id!r}, "
            f"source_name=[REDACTED], "
            f"content=[REDACTED {len(self.content)} bytes, sha256={content_sha}...], "
            f"language_hint={self.language_hint!r}, "
            f"options={self.options!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()
