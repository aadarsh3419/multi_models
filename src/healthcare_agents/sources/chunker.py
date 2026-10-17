"""
Text Chunker and fragmenter with zero-drop character coverage guarantee.

Packs consecutive segments into provider units without exceeding MAX_LLM_INPUT_CHARS.
Splits oversized segments on sentence boundaries (including Hindi danda '।') with overlap.
Zero provider or agent dependencies.
"""

import re
import unicodedata

from healthcare_agents.contracts.source_text import ProviderUnit, SourceSegment, SourceTextBundle


class TextChunker:
    """
    Transforms SourceTextBundle into a sequence of ProviderUnits bounded by max_input_chars.
    Enforces lossless character coverage and stable parent lineage.
    """

    SENTENCE_SPLIT_REGEX = re.compile(r"([।\.\?!]+(?:\s+|$))")

    def __init__(
        self,
        max_input_chars: int = 6000,
        overlap_chars: int = 200,
        max_chars_per_chunk: int | None = None,
    ):
        if max_chars_per_chunk is not None:
            max_input_chars = max_chars_per_chunk

        if max_input_chars <= 0:
            raise ValueError("max_input_chars must be strictly positive.")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")
        if overlap_chars >= max_input_chars:
            overlap_chars = max(0, max_input_chars // 2)

        self.max_input_chars = max_input_chars
        self.overlap_chars = overlap_chars

    def chunk_bundle(self, bundle: SourceTextBundle) -> list[ProviderUnit]:
        """
        Processes segments in the bundle, keeping table rows unsplit and repeating
        table headers across continuation chunks, while fragmenting oversized
        non-table segments losslessly.
        """
        units: list[ProviderUnit] = []
        i = 0
        segs = bundle.ordered_segments
        n = len(segs)

        while i < n:
            seg = segs[i]

            # If segment is part of a table
            if seg.table_id is not None:
                table_id = seg.table_id
                table_segs: list[SourceSegment] = []
                while i < n and segs[i].table_id == table_id:
                    table_segs.append(segs[i])
                    i += 1

                # Locate header row if present
                header_seg: SourceSegment | None = None
                for ts in table_segs:
                    if ts.is_table_header:
                        header_seg = ts
                        break
                if header_seg is None and table_segs:
                    header_seg = table_segs[0]

                # Chunk table rows into provider units without splitting rows
                current_chunk_segs: list[SourceSegment] = []
                current_chunk_chars = 0
                chunk_counter = 1

                for ts in table_segs:
                    # If ts is the header itself and chunk is empty, add it
                    if ts == header_seg and not current_chunk_segs:
                        current_chunk_segs.append(ts)
                        current_chunk_chars += len(ts.text)
                        continue

                    needed_len = len(ts.text) + (1 if current_chunk_segs else 0)
                    if current_chunk_segs and (
                        current_chunk_chars + needed_len > self.max_input_chars
                    ):
                        # Emit current chunk
                        unit_text = "\n".join(s.text for s in current_chunk_segs)
                        first_seg = current_chunk_segs[0]
                        units.append(
                            ProviderUnit(
                                unit_id=f"{first_seg.segment_id}#chk{chunk_counter:02d}",
                                text=unit_text,
                                parent_segment_id=first_seg.segment_id,
                                parent_start=0,
                                parent_end=len(unit_text),
                                member_segments=list(current_chunk_segs),
                            )
                        )
                        chunk_counter += 1
                        # Continuation chunk: repeat table header as context
                        current_chunk_segs = []
                        current_chunk_chars = 0
                        if header_seg is not None and ts != header_seg:
                            current_chunk_segs.append(header_seg)
                            current_chunk_chars = len(header_seg.text)

                    current_chunk_segs.append(ts)
                    current_chunk_chars += len(ts.text) + (1 if current_chunk_segs else 0)

                if current_chunk_segs:
                    unit_text = "\n".join(s.text for s in current_chunk_segs)
                    first_seg = current_chunk_segs[0]
                    units.append(
                        ProviderUnit(
                            unit_id=f"{first_seg.segment_id}#chk{chunk_counter:02d}",
                            text=unit_text,
                            parent_segment_id=first_seg.segment_id,
                            parent_start=0,
                            parent_end=len(unit_text),
                            member_segments=list(current_chunk_segs),
                        )
                    )
            else:
                # Non-table segment: standard M1 logic
                seg_len = len(seg.text)
                if seg_len <= self.max_input_chars:
                    units.append(
                        ProviderUnit(
                            unit_id=seg.segment_id,
                            text=seg.text,
                            parent_segment_id=seg.segment_id,
                            parent_start=0,
                            parent_end=seg_len,
                            member_segments=[seg],
                        )
                    )
                else:
                    fragments = self._fragment_large_segment(seg)
                    units.extend(fragments)
                i += 1

        return units

    def _fragment_large_segment(self, segment: SourceSegment) -> list[ProviderUnit]:
        """
        Fragments an oversized segment using sentence and boundary heuristics with overlap.
        Guarantees that every character in segment.text is covered by at least one fragment.
        """
        text = segment.text
        total_len = len(text)
        fragments: list[ProviderUnit] = []
        frag_idx = 1
        current_start = 0

        while current_start < total_len:
            ideal_end = min(current_start + self.max_input_chars, total_len)

            if ideal_end == total_len:
                cut_point = total_len
            else:
                cut_point = self._find_best_cut_point(text, current_start, ideal_end)

            # Prevent splitting within a combining mark sequence
            while cut_point < total_len and unicodedata.category(text[cut_point]).startswith("M"):
                cut_point += 1

            # Prevent infinite loop if cut_point didn't advance
            if cut_point <= current_start:
                cut_point = ideal_end

            unit_id = f"{segment.segment_id}#f{frag_idx:03d}"
            frag_text = text[current_start:cut_point]

            fragments.append(
                ProviderUnit(
                    unit_id=unit_id,
                    text=frag_text,
                    parent_segment_id=segment.segment_id,
                    parent_start=current_start,
                    parent_end=cut_point,
                )
            )

            if cut_point >= total_len:
                break

            # Calculate next start with overlap
            next_start = cut_point - self.overlap_chars
            if next_start <= current_start:
                next_start = cut_point

            # Ensure next_start doesn't start in the middle of a combining mark sequence
            while next_start < total_len and unicodedata.category(text[next_start]).startswith("M"):
                next_start += 1

            current_start = next_start
            frag_idx += 1

        return fragments

    def _find_best_cut_point(self, text: str, start: int, max_end: int) -> int:
        """
        Searches backwards from max_end down to a minimum boundary for sentence or whitespace breaks.
        Includes Hindi danda '।'.
        """
        search_window_start = max(start, max_end - int(self.max_input_chars * 0.3))
        sub = text[search_window_start:max_end]

        # 1. Search for sentence terminators (. ! ? ।)
        matches = list(self.SENTENCE_SPLIT_REGEX.finditer(sub))
        if matches:
            last_match = matches[-1]
            return search_window_start + last_match.end()

        # 2. Search for newline
        nl_pos = sub.rfind("\n")
        if nl_pos != -1:
            return search_window_start + nl_pos + 1

        # 3. Search for whitespace
        sp_pos = sub.rfind(" ")
        if sp_pos != -1:
            return search_window_start + sp_pos + 1

        # Fallback to hard split
        return max_end
