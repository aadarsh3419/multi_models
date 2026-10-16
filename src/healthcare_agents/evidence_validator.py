"""Source evidence validator for clinical extraction.

Enforces strict provenance rules:
- Verifies referenced segment IDs exist.
- Performs exact quotation matching against untouched segment text.
- Provides a tokenized whitespace-fallback with exact coordinate mapping back to original text.
- Rejects duplicate quote occurrences as ambiguous_match (never picks arbitrarily).
- Never trusts model-provided coordinates.
"""

import re
import unicodedata

from healthcare_agents.contracts import (
    EvidenceReference,
    EvidenceStatus,
    ExtractionCandidateRaw,
    ValidatedEvidenceRecord,
)
from healthcare_agents.source_text import SourceTextBundle


class EvidenceValidator:
    """Validates extracted quotations against SourceTextBundle segments."""

    @classmethod
    def validate_candidate_evidence(
        cls,
        candidate: ExtractionCandidateRaw,
        bundle: SourceTextBundle,
    ) -> tuple[list[ValidatedEvidenceRecord], EvidenceStatus]:
        """Validate all evidence references for a raw candidate against the bundle.

        Returns:
            Tuple of (list of ValidatedEvidenceRecords, aggregate EvidenceStatus)
        """
        if not candidate.evidence_refs:
            return (
                [],
                EvidenceStatus.unmatched,
            )

        validated_records: list[ValidatedEvidenceRecord] = []
        statuses: list[EvidenceStatus] = []

        for ref in candidate.evidence_refs:
            record = cls.validate_single_reference(ref, bundle)
            validated_records.append(record)
            statuses.append(record.status)

        # Aggregate evidence status
        if all(s == EvidenceStatus.matched for s in statuses):
            agg_status = EvidenceStatus.matched
        elif any(s == EvidenceStatus.ambiguous_match for s in statuses):
            agg_status = EvidenceStatus.ambiguous_match
        elif any(s == EvidenceStatus.unmatched for s in statuses):
            agg_status = EvidenceStatus.unmatched
        else:
            agg_status = EvidenceStatus.unavailable

        return validated_records, agg_status

    @classmethod
    def validate_single_reference(
        cls,
        ref: EvidenceReference,
        bundle: SourceTextBundle,
    ) -> ValidatedEvidenceRecord:
        """Validate a single EvidenceReference against a specific segment in the bundle."""
        seg_id = ref.segment_id.strip()
        quote = ref.quote

        segment = bundle.get_segment(seg_id)
        if segment is None:
            return ValidatedEvidenceRecord(
                segment_id=seg_id,
                quote=quote,
                start_offset=None,
                end_offset=None,
                status=EvidenceStatus.unmatched,
                safe_warning="evidence_reference_unknown: Segment ID does not exist in bundle.",
            )

        if not quote:
            return ValidatedEvidenceRecord(
                segment_id=seg_id,
                quote=quote,
                start_offset=None,
                end_offset=None,
                status=EvidenceStatus.unmatched,
                safe_warning="evidence_not_found: Empty quotation provided.",
            )

        seg_text = segment.exact_text

        # 1. Exact match attempt
        exact_matches = cls._find_all_substring_indices(seg_text, quote)
        if len(exact_matches) == 1:
            rel_start = exact_matches[0]
            rel_end = rel_start + len(quote)
            abs_start = segment.start_offset + rel_start
            abs_end = segment.start_offset + rel_end
            return ValidatedEvidenceRecord(
                segment_id=seg_id,
                quote=quote,
                start_offset=abs_start,
                end_offset=abs_end,
                status=EvidenceStatus.matched,
            )
        elif len(exact_matches) > 1:
            # Duplicate occurrences inside the same segment cannot be safely disambiguated
            return ValidatedEvidenceRecord(
                segment_id=seg_id,
                quote=quote,
                start_offset=None,
                end_offset=None,
                status=EvidenceStatus.ambiguous_match,
                safe_warning=f"duplicate_quote_occurrences: Quote occurs {len(exact_matches)} times in segment {seg_id}.",
            )

        # 2. Whitespace-normalized fallback with precise original offset mapping
        ws_mapped = cls._find_whitespace_normalized_match(seg_text, quote)
        if ws_mapped is not None:
            rel_start, rel_end, match_count = ws_mapped
            if match_count == 1:
                abs_start = segment.start_offset + rel_start
                abs_end = segment.start_offset + rel_end
                return ValidatedEvidenceRecord(
                    segment_id=seg_id,
                    quote=quote,
                    start_offset=abs_start,
                    end_offset=abs_end,
                    status=EvidenceStatus.matched,
                    safe_warning="matched_via_whitespace_normalization",
                )
            else:
                return ValidatedEvidenceRecord(
                    segment_id=seg_id,
                    quote=quote,
                    start_offset=None,
                    end_offset=None,
                    status=EvidenceStatus.ambiguous_match,
                    safe_warning=f"ambiguous_whitespace_match: Quote occurs {match_count} times under whitespace normalization.",
                )

        # 3. Unicode NFC fallback check (verifies if quote differs only by decomposed accents/characters)
        nfc_mapped = cls._check_unicode_nfc_fallback(seg_text, quote)
        if nfc_mapped is not None:
            rel_start, rel_end, match_count = nfc_mapped
            if match_count == 1:
                abs_start = segment.start_offset + rel_start
                abs_end = segment.start_offset + rel_end
                return ValidatedEvidenceRecord(
                    segment_id=seg_id,
                    quote=quote,
                    start_offset=abs_start,
                    end_offset=abs_end,
                    status=EvidenceStatus.matched,
                    safe_warning="matched_via_unicode_nfc_alignment",
                )
            elif match_count > 1:
                return ValidatedEvidenceRecord(
                    segment_id=seg_id,
                    quote=quote,
                    start_offset=None,
                    end_offset=None,
                    status=EvidenceStatus.ambiguous_match,
                    safe_warning="ambiguous_unicode_match: Multiple NFC matches in segment.",
                )

        # Quote not found in segment
        return ValidatedEvidenceRecord(
            segment_id=seg_id,
            quote=quote,
            start_offset=None,
            end_offset=None,
            status=EvidenceStatus.unmatched,
            safe_warning="evidence_not_found: Quotation not found in segment.",
        )

    @staticmethod
    def _find_all_substring_indices(haystack: str, needle: str) -> list[int]:
        """Find all start offsets of needle in haystack."""
        indices: list[int] = []
        idx = 0
        while True:
            pos = haystack.find(needle, idx)
            if pos == -1:
                break
            indices.append(pos)
            idx = pos + 1
        return indices

    @staticmethod
    def _find_whitespace_normalized_match(
        seg_text: str,
        quote: str,
    ) -> tuple[int, int, int] | None:
        """Map whitespace-collapsed quote tokens back to precise original slice in seg_text.

        Returns (rel_start, rel_end, match_count) or None if no match.
        """
        # Tokenize quote by whitespace
        quote_tokens = [tok for tok in re.split(r"\s+", quote.strip()) if tok]
        if not quote_tokens:
            return None

        # Tokenize segment by whitespace while tracking original start/end char indices
        seg_token_spans: list[tuple[str, int, int]] = []
        for match in re.finditer(r"\S+", seg_text):
            seg_token_spans.append((match.group(), match.start(), match.end()))

        k = len(quote_tokens)
        matches: list[tuple[int, int]] = []

        for i in range(len(seg_token_spans) - k + 1):
            window = [seg_token_spans[i + j][0] for j in range(k)]
            if window == quote_tokens:
                start_char = seg_token_spans[i][1]
                end_char = seg_token_spans[i + k - 1][2]
                matches.append((start_char, end_char))

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0][0], matches[0][1], 1
        else:
            return 0, 0, len(matches)

    @staticmethod
    def _check_unicode_nfc_fallback(
        seg_text: str,
        quote: str,
    ) -> tuple[int, int, int] | None:
        """Check for match when quote and segment text are NFC normalized.

        Only produces valid offsets if the character lengths align or exact slice maps back.
        """
        quote_nfc = unicodedata.normalize("NFC", quote)
        seg_nfc = unicodedata.normalize("NFC", seg_text)

        # If seg_text is already NFC normalized and lengths equal:
        if seg_text == seg_nfc:
            matches = EvidenceValidator._find_all_substring_indices(seg_text, quote_nfc)
            if len(matches) == 1:
                return matches[0], matches[0] + len(quote_nfc), 1
            elif len(matches) > 1:
                return 0, 0, len(matches)

        return None
