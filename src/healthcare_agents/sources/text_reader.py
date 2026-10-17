"""
Source text ingestion and paragraph segmentation.

Never modifies or normalizes original text code points.
Zero provider or agent dependencies.
"""

import hashlib
import re
from uuid import UUID, uuid4

from healthcare_agents.contracts.locations import TextLocation
from healthcare_agents.contracts.source_text import SourceSegment, SourceTextBundle


class TextReader:
    """
    Ingests raw clinical narratives and produces coordinate-grounded SourceTextBundle.
    Splits along paragraph / blank-line boundaries, ensuring every segment's text
    matches untouched slicing: raw_text[start:end] == segment.text.
    """

    # Matches one or more non-empty lines bounded by blank lines or start/end of string
    PARAGRAPH_REGEX = re.compile(r"[^\r\n]+(?:\r?\n[^\r\n]+)*")

    @classmethod
    def read_bundle(cls, text: str, request_id: UUID) -> SourceTextBundle:
        """
        Parses raw text into ordered SourceSegments and bundles them with cryptographic metadata.
        """
        if not text or not text.strip():
            raise ValueError(
                "Cannot construct SourceTextBundle from empty or whitespace-only text."
            )

        content_bytes = text.encode("utf-8")
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        generated_source_id = f"src_{uuid4().hex[:12]}"

        segments: list[SourceSegment] = []
        segment_counter = 1

        for match in cls.PARAGRAPH_REGEX.finditer(text):
            seg_text = match.group(0)
            if not seg_text.strip():
                continue

            start_cp = match.start()
            end_cp = match.end()

            # Mathematical invariant guarantee:
            assert text[start_cp:end_cp] == seg_text, (
                "Segment offset mismatch detected during text ingestion"
            )

            seg_id = f"s{segment_counter:04d}"
            segment = SourceSegment(
                segment_id=seg_id,
                text=seg_text,
                extraction_method="direct_split",
                source_location=TextLocation(kind="text", start=start_cp, end=end_cp),
            )
            segments.append(segment)
            segment_counter += 1

        # Fallback if text contained only newlines with non-matching pattern
        if not segments:
            # Whole string as single segment
            segments.append(
                SourceSegment(
                    segment_id="s0001",
                    text=text,
                    extraction_method="direct_split",
                    source_location=TextLocation(kind="text", start=0, end=len(text)),
                )
            )

        safe_metadata = {
            "source_id": generated_source_id,
            "media_type": "text/plain",
            "char_count": len(text),
            "byte_count": len(content_bytes),
            "content_sha256": content_sha256,
        }

        return SourceTextBundle(
            request_id=request_id,
            source_type="text",
            safe_metadata=safe_metadata,
            extraction_method="direct_split",
            ordered_segments=segments,
        )
