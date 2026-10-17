"""
Source Text Bundle, Segments, and Provider Units contracts.

Zero provider or agent dependencies.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from healthcare_agents.contracts.locations import SourceLocation

SourceType = Literal["text", "pdf", "image", "audio"]


class SourceSegment(BaseModel):
    """
    An atomic segment of extracted source text with associated coordinates and provenance.
    For text sources, text == original_text[start:end] strictly holds.
    """

    segment_id: str = Field(
        description="Unique segment identifier within the bundle (e.g. 's0001')."
    )
    text: str = Field(description="Exact character slice of the segment.")
    extraction_method: str = Field(
        description="Method used to extract segment (e.g. 'direct_split', 'pymupdf', 'tesseract')."
    )
    source_location: SourceLocation = Field(description="Typed physical/temporal location.")
    parent_segment_id: str | None = Field(
        default=None, description="Parent segment ID if this is a sub-fragment."
    )
    parent_start: int | None = Field(
        default=None, ge=0, description="Start offset within parent segment."
    )
    parent_end: int | None = Field(
        default=None, ge=0, description="End offset within parent segment."
    )
    page_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    cell_id: str | None = None
    layout_group_id: str | None = None
    ocr_word_confidences: list[float] | None = None
    is_table_header: bool = False
    table_row_cells: list[str] | None = None
    table_cell_spans: list[tuple[int, int]] | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def exact_text(self) -> str:
        return self.text

    @property
    def start_offset(self) -> int:
        return self.source_location.start if hasattr(self.source_location, "start") else 0

    @property
    def end_offset(self) -> int:
        return self.source_location.end if hasattr(self.source_location, "end") else 0


class ProviderUnit(BaseModel):
    """
    The atomic text unit dispatched to a provider (LLM) during extraction.
    Produced by the TextChunker, preserving lineage to parent segment(s).
    """

    unit_id: str = Field(
        description="Unit identifier dispatched in API prompt (e.g. 's0001' or 's0007#f001')."
    )
    text: str = Field(description="Exact unit text dispatched to the LLM.")
    parent_segment_id: str = Field(description="Primary parent segment ID.")
    parent_start: int = Field(
        ge=0, description="Offset within parent segment where this unit begins."
    )
    parent_end: int = Field(ge=0, description="Offset within parent segment where this unit ends.")
    member_segments: list[Any] = Field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.unit_id

    @property
    def total_chars(self) -> int:
        return len(self.text)

    @property
    def segments(self) -> list[Any]:
        if self.member_segments:
            return self.member_segments

        class _SegWrapper:
            def __init__(self, uid: str, text: str, p_id: str, p_s: int, p_e: int):
                self.segment_id = uid
                self.exact_text = text
                self.start_offset = p_s
                self.end_offset = p_e
                self.parent_segment_id = p_id
                self.parent_start_offset = p_s
                self.parent_end_offset = p_e

        return [
            _SegWrapper(
                self.unit_id, self.text, self.parent_segment_id, self.parent_start, self.parent_end
            )
        ]


class SourceTextBundle(BaseModel):
    """
    Immutable container of source segments representing the complete ingested document.
    """

    request_id: UUID = Field(description="Request ID associated with this document bundle.")
    source_type: SourceType = Field(default="text", description="Ingestion modality.")
    safe_metadata: dict[str, Any] = Field(
        description="Sanitized metadata: generated source_id (never path/filename), media_type, char_count, byte_count, content_sha256.",
    )
    extraction_method: str = Field(description="Ingestion technique used to build the bundle.")
    ordered_segments: list[SourceSegment] = Field(
        description="Chronological or visual order of segments."
    )
    warnings: list[str] = Field(default_factory=list)

    @property
    def segments(self) -> list[SourceSegment]:
        return self.ordered_segments

    @model_validator(mode="after")
    def validate_bundle_invariants(self) -> "SourceTextBundle":
        if not self.ordered_segments:
            raise ValueError("SourceTextBundle must contain at least one SourceSegment.")

        segment_ids = set()
        for seg in self.ordered_segments:
            if seg.segment_id in segment_ids:
                raise ValueError(f"Duplicate segment_id detected in bundle: '{seg.segment_id}'")
            segment_ids.add(seg.segment_id)

        return self
