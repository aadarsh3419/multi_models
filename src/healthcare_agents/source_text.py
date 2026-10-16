"""Source text bundling and lossless segment mapping.

Ensures that every segmented piece of clinical text retains exact,
un-normalized Python string offsets back to the original source text.
"""

import re
from hashlib import sha256
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SourceSegment(BaseModel):
    """An exact, source-preserving slice of clinical text with original coordinates."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(
        ...,
        description="Stable, request-local identifier (e.g. txt_000001).",
    )
    exact_text: str = Field(
        ...,
        repr=False,
        description="Exact raw segment text. Excluded from default repr to prevent PHI exposure.",
    )
    start_offset: int = Field(
        ...,
        ge=0,
        description="Zero-based Python Unicode code-point offset where this segment starts.",
    )
    end_offset: int = Field(
        ...,
        ge=0,
        description="Zero-based Python Unicode code-point offset where this segment ends.",
    )
    source_type: str = Field(
        default="text",
        description="Source modality ('text' in Milestone 1; extensible to 'pdf', 'image', 'audio').",
    )
    extraction_method: str = Field(
        default="paragraph_line_split",
        description="Method used to split this segment from the source.",
    )

    # Extension points for fragments or multimodal source locations
    parent_segment_id: str | None = Field(
        default=None,
        description="If this segment is a sub-chunk fragment, points to parent segment ID.",
    )
    parent_start_offset: int | None = Field(
        default=None,
        description="Offset relative to parent segment if fragmented.",
    )
    parent_end_offset: int | None = Field(
        default=None,
        description="End offset relative to parent segment if fragmented.",
    )
    multimodal_location: dict[str, Any] | None = Field(
        default=None,
        description="Future extension point for PDF page/bbox, image ROI, or audio timestamps.",
    )

    def __repr__(self) -> str:
        return (
            f"SourceSegment(id={self.segment_id!r}, "
            f"span=[{self.start_offset}:{self.end_offset}], "
            f"chars={len(self.exact_text)})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class SourceTextBundle(BaseModel):
    """Collection of ordered source segments retaining source-level provenance and hashing."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    source_type: str = "text"
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    segments: list[SourceSegment] = Field(default_factory=list)
    source_content_hash: str
    processing_warnings: list[str] = Field(default_factory=list)

    def get_segment(self, segment_id: str) -> SourceSegment | None:
        """Lookup a segment by its stable identifier."""
        for seg in self.segments:
            if seg.segment_id == segment_id:
                return seg
        return None

    def __repr__(self) -> str:
        return (
            f"SourceTextBundle(request_id={self.request_id}, "
            f"segments_count={len(self.segments)}, "
            f"hash={self.source_content_hash[:8]}...)"
        )

    def __str__(self) -> str:
        return self.__repr__()


def create_source_text_bundle(
    raw_text: str,
    request_id: UUID,
    metadata: dict[str, Any] | None = None,
) -> SourceTextBundle:
    """Build a SourceTextBundle from raw text using boundary-preserving paragraph/line segmentation.

    Guarantees:
    1. raw_text is never modified, normalized, or discarded.
    2. For every segment, raw_text[seg.start_offset:seg.end_offset] == seg.exact_text.
    3. Python string indices count Unicode code points (len of str).
    4. Stable, sequential IDs: txt_000001, txt_000002, ...
    """
    if metadata is None:
        metadata = {}

    content_hash = sha256(raw_text.encode("utf-8")).hexdigest()
    segments: list[SourceSegment] = []
    warnings: list[str] = []

    # Segment on paragraph boundaries (double newlines) and lines.
    # We find blocks of text separated by blank lines or section delimiters,
    # ensuring non-empty slices keep exact start and end offsets.
    # Regex splits by contiguous whitespace lines while keeping track of match spans.
    pattern = re.compile(r"(\r\n\r\n+|\n\n+)")
    seg_counter = 1

    # Split text into paragraphs first
    paragraphs: list[tuple[int, int]] = []
    last_end = 0
    for match in pattern.finditer(raw_text):
        p_start = last_end
        p_end = match.start()
        if p_end > p_start:
            paragraphs.append((p_start, p_end))
        last_end = match.end()
    if last_end < len(raw_text):
        paragraphs.append((last_end, len(raw_text)))

    # If no paragraphs or empty, fall back to single span if text exists
    if not paragraphs and raw_text:
        paragraphs.append((0, len(raw_text)))

    for p_start, p_end in paragraphs:
        p_text = raw_text[p_start:p_end]
        # If paragraph is very short or single line, keep as segment
        # Trim leading/trailing blank lines within paragraph to make neat segments
        # but calculate offsets precisely against raw_text
        match_content = re.search(r"\S.*\S", p_text, flags=re.DOTALL)
        if not match_content:
            # Check single non-whitespace char
            match_content = re.search(r"\S", p_text)

        if match_content:
            rel_start = match_content.start()
            rel_end = match_content.end()
            exact_seg_text = p_text[rel_start:rel_end]
            abs_start = p_start + rel_start
            abs_end = p_start + rel_end

            # Verify invariant
            assert raw_text[abs_start:abs_end] == exact_seg_text, "Segment offset mismatch!"

            seg_id = f"txt_{seg_counter:06d}"
            segments.append(
                SourceSegment(
                    segment_id=seg_id,
                    exact_text=exact_seg_text,
                    start_offset=abs_start,
                    end_offset=abs_end,
                    source_type="text",
                    extraction_method="paragraph_boundary_split",
                )
            )
            seg_counter += 1

    if not segments and raw_text:
        # Whitespace-only edge case: retain single segment
        seg_id = f"txt_{seg_counter:06d}"
        segments.append(
            SourceSegment(
                segment_id=seg_id,
                exact_text=raw_text,
                start_offset=0,
                end_offset=len(raw_text),
                source_type="text",
                extraction_method="fallback_full_span",
            )
        )

    return SourceTextBundle(
        request_id=request_id,
        source_type="text",
        source_metadata=metadata,
        segments=segments,
        source_content_hash=content_hash,
        processing_warnings=warnings,
    )
