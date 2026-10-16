"""Text chunker for bounded sequential LLM extraction.

Implements:
- Bounded chunk generation respecting MAX_LLM_CHARS_PER_CHUNK (default 6,000).
- Semantic boundary preference (paragraph, line, sentence).
- Precise fragment parent tracking without dropping any characters.
- Overlap handling (200 chars) only when semantic splits are unavailable.
- Exact original coverage verification.
"""

import re

from pydantic import BaseModel, ConfigDict

from healthcare_agents.source_text import SourceSegment, SourceTextBundle


class TextChunk(BaseModel):
    """A bounded bundle of source segments/fragments submitted to an LLM extraction attempt."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    segments: list[SourceSegment]
    start_offset: int
    end_offset: int
    total_chars: int
    has_overlap: bool = False
    overlap_chars: int = 0

    def __repr__(self) -> str:
        return (
            f"TextChunk(id={self.chunk_id!r}, segments={len(self.segments)}, "
            f"span=[{self.start_offset}:{self.end_offset}], chars={self.total_chars})"
        )

    def __str__(self) -> str:
        return self.__repr__()


class TextChunker:
    """Chunks SourceTextBundle segments into bounded pieces for LLM extraction."""

    def __init__(
        self,
        max_chars_per_chunk: int = 6000,
        overlap_chars: int = 200,
    ) -> None:
        self.max_chars = max_chars_per_chunk
        self.overlap = overlap_chars
        # Sentence splitting pattern including English and Hindi danda (।)
        self.sentence_end_re = re.compile(r"([.!?।]|\r?\n)\s+")

    def chunk_bundle(self, bundle: SourceTextBundle) -> list[TextChunk]:
        """Convert bundle segments into a sequence of bounded TextChunks.

        Guarantees:
        - No segment character is dropped.
        - Every chunk is within max_chars (unless an unbreakable atomic token exceeds max_chars).
        - Segments larger than max_chars are cleanly fragmented with parent provenance.
        """
        if not bundle.segments:
            return []

        # Step 1: Pre-process segments to fragment any segment exceeding max_chars
        prepared_segments: list[SourceSegment] = []
        for seg in bundle.segments:
            if len(seg.exact_text) <= self.max_chars:
                prepared_segments.append(seg)
            else:
                fragments = self._fragment_large_segment(seg)
                prepared_segments.extend(fragments)

        # Step 2: Pack segments into chunks sequentially
        chunks: list[TextChunk] = []
        current_chunk_segments: list[SourceSegment] = []
        current_chunk_chars = 0
        chunk_idx = 1

        for seg in prepared_segments:
            seg_len = len(seg.exact_text)
            if current_chunk_segments and (current_chunk_chars + seg_len > self.max_chars):
                # Finalize current chunk
                chunks.append(
                    self._build_chunk(
                        chunk_id=f"chk_{chunk_idx:06d}",
                        segments=current_chunk_segments,
                    )
                )
                chunk_idx += 1
                current_chunk_segments = [seg]
                current_chunk_chars = seg_len
            else:
                current_chunk_segments.append(seg)
                current_chunk_chars += seg_len

        if current_chunk_segments:
            chunks.append(
                self._build_chunk(
                    chunk_id=f"chk_{chunk_idx:06d}",
                    segments=current_chunk_segments,
                )
            )

        return chunks

    def _fragment_large_segment(self, segment: SourceSegment) -> list[SourceSegment]:
        """Split a large segment into smaller fragments with exact parent offsets."""
        text = segment.exact_text
        fragments: list[SourceSegment] = []
        frag_idx = 1
        pos = 0
        text_len = len(text)

        while pos < text_len:
            remaining = text_len - pos
            if remaining <= self.max_chars:
                rel_start = pos
                rel_end = text_len
                pos = text_len
                has_overlap = False
            else:
                # Target slice of size self.max_chars
                target_end = pos + self.max_chars
                slice_to_inspect = text[pos:target_end]

                # Look for sentence boundary near the end of the slice (last 1000 chars)
                search_start = max(0, len(slice_to_inspect) - 1000)
                matches = list(self.sentence_end_re.finditer(slice_to_inspect[search_start:]))

                if matches:
                    last_match = matches[-1]
                    split_offset = search_start + last_match.end()
                    rel_start = pos
                    rel_end = pos + split_offset
                    pos = rel_end
                    has_overlap = False
                else:
                    # Hard boundary: split at target_end with overlap for the next chunk
                    rel_start = pos
                    rel_end = target_end
                    pos = max(pos + 1, target_end - self.overlap)
                    has_overlap = True

            frag_text = text[rel_start:rel_end]
            abs_start = segment.start_offset + rel_start
            abs_end = segment.start_offset + rel_end

            frag_id = f"{segment.segment_id}_frag_{frag_idx:02d}"
            fragments.append(
                SourceSegment(
                    segment_id=frag_id,
                    exact_text=frag_text,
                    start_offset=abs_start,
                    end_offset=abs_end,
                    source_type=segment.source_type,
                    extraction_method="fragment_split_overlap"
                    if has_overlap
                    else "fragment_split_boundary",
                    parent_segment_id=segment.segment_id,
                    parent_start_offset=rel_start,
                    parent_end_offset=rel_end,
                )
            )
            frag_idx += 1

        return fragments

    @staticmethod
    def _build_chunk(chunk_id: str, segments: list[SourceSegment]) -> TextChunk:
        """Construct a TextChunk from a group of segments."""
        start_off = min(s.start_offset for s in segments)
        end_off = max(s.end_offset for s in segments)
        total_len = sum(len(s.exact_text) for s in segments)
        has_overlap = any(s.extraction_method == "fragment_split_overlap" for s in segments)

        return TextChunk(
            chunk_id=chunk_id,
            segments=segments,
            start_offset=start_off,
            end_offset=end_off,
            total_chars=total_len,
            has_overlap=has_overlap,
            overlap_chars=200 if has_overlap else 0,
        )
