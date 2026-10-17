"""
EvidenceValidator: Independent quote grounding, coordinate verification, and field containment.

Never trusts model-reported offsets. Computes coordinates from application metadata.
Zero provider SDK imports.
"""

import unicodedata
from typing import Any

from healthcare_agents.contracts.errors import StructuredError, WarningCode
from healthcare_agents.contracts.facts import (
    EvidenceRecord,
    EvidenceStatus,
    FactCandidate,
    ProviderCandidate,
    ReviewStatus,
    compute_candidate_id,
)
from healthcare_agents.contracts.locations import SourceLocation, TextLocation
from healthcare_agents.contracts.source_text import ProviderUnit, SourceSegment
from healthcare_agents.sources.normalization import build_normalized_index_map, canonicalize_label


class EvidenceValidator:
    """
    Verifies that model-cited evidence quotations exist verbatim in the ingested text units,
    maps them back to original code-point coordinates, and verifies raw-field containment.
    """

    @classmethod
    def validate_single_reference(cls, ref: Any, bundle: Any) -> Any:
        from healthcare_agents.contracts import ValidatedEvidenceRecord

        target_seg = None
        segs = (
            bundle.ordered_segments
            if hasattr(bundle, "ordered_segments")
            else getattr(bundle, "segments", [])
        )
        for s in segs:
            if s.segment_id == ref.segment_id:
                target_seg = s
                break
        if not target_seg:
            return ValidatedEvidenceRecord(
                segment_id=ref.segment_id,
                quote=ref.quote,
                status=EvidenceStatus.UNMATCHED,
                safe_warning="evidence_reference_unknown",
            )

        quote = ref.quote
        text = target_seg.text
        count_raw = text.count(quote)
        seg_start = (
            target_seg.source_location.start
            if hasattr(target_seg.source_location, "start")
            else getattr(target_seg, "start_offset", 0)
        )

        if count_raw > 1:
            return ValidatedEvidenceRecord(
                segment_id=ref.segment_id,
                quote=quote,
                status=EvidenceStatus.AMBIGUOUS_MATCH,
                safe_warning="duplicate_quote_occurrences",
            )
        elif count_raw == 1:
            start_in_seg = text.find(quote)
            end_in_seg = start_in_seg + len(quote)
            return ValidatedEvidenceRecord(
                segment_id=ref.segment_id,
                quote=quote,
                start_offset=seg_start + start_in_seg,
                end_offset=seg_start + end_in_seg,
                status=EvidenceStatus.MATCHED,
            )
        else:
            norm_text, o_starts, o_ends = build_normalized_index_map(text)
            norm_quote, _, _ = build_normalized_index_map(quote)
            count_norm = norm_text.count(norm_quote)
            if count_norm == 1:
                idx = norm_text.find(norm_quote)
                orig_start = o_starts[idx]
                orig_end = o_ends[idx + len(norm_quote) - 1]
                return ValidatedEvidenceRecord(
                    segment_id=ref.segment_id,
                    quote=quote,
                    start_offset=seg_start + orig_start,
                    end_offset=seg_start + orig_end,
                    status=EvidenceStatus.MATCHED,
                )
            elif count_norm > 1:
                return ValidatedEvidenceRecord(
                    segment_id=ref.segment_id,
                    quote=quote,
                    status=EvidenceStatus.AMBIGUOUS_MATCH,
                    safe_warning="duplicate_quote_occurrences",
                )
            else:
                return ValidatedEvidenceRecord(
                    segment_id=ref.segment_id,
                    quote=quote,
                    status=EvidenceStatus.UNMATCHED,
                    safe_warning="evidence_not_found",
                )

    @classmethod
    def validate_candidate_evidence(
        cls, candidate: Any, bundle: Any
    ) -> tuple[list[Any], EvidenceStatus]:
        val_records = []
        overall_status = EvidenceStatus.MATCHED
        for ref in candidate.evidence_refs:
            rec = cls.validate_single_reference(ref, bundle)
            val_records.append(rec)
            if rec.status != EvidenceStatus.MATCHED:
                overall_status = rec.status
        return val_records, overall_status

    def validate_candidates(
        self,
        provider_candidates: list[ProviderCandidate],
        unit: ProviderUnit,
        parent_segment: SourceSegment,
        content_sha256: str,
        provenance: dict[str, str] | None = None,
    ) -> tuple[list[FactCandidate], list[StructuredError]]:
        """
        Validates provider candidates for a single unit.
        Returns:
            list[FactCandidate]: Accepted candidates with grounded evidence locations.
            list[StructuredError]: Structured warnings for rejected or ambiguous candidates.
        """
        provenance = provenance or {}
        accepted_candidates: list[FactCandidate] = []
        warnings: list[StructuredError] = []

        # Precompute normalized unit text and offset maps
        norm_unit_text, orig_starts, orig_ends = build_normalized_index_map(unit.text)

        for candidate in provider_candidates:
            # Rule 1: No evidence refs -> Reject
            if not candidate.evidence_refs:
                warnings.append(
                    StructuredError(
                        code=WarningCode.EVIDENCE_NOT_FOUND.value,
                        stage="evidence_validation",
                        message="Candidate rejected because no evidence references were cited.",
                        details={"unit_id": unit.unit_id},
                    )
                )
                continue

            candidate_valid = True
            evidence_records: list[EvidenceRecord] = []
            overall_evidence_status = EvidenceStatus.MATCHED
            evidence_spans: list[tuple[str, int, int]] = []
            combined_quotes: list[str] = []

            for ref in candidate.evidence_refs:
                # Rule 1: Segment ID must match current unit or parent segment
                if ref.segment_id not in (unit.unit_id, parent_segment.segment_id):
                    warnings.append(
                        StructuredError(
                            code=WarningCode.EVIDENCE_REFERENCE_UNKNOWN.value,
                            stage="evidence_validation",
                            message="Candidate rejected because evidence segment_id does not match active unit.",
                            details={"unit_id": unit.unit_id, "cited_segment_id": ref.segment_id},
                        )
                    )
                    candidate_valid = False
                    break

                quote = ref.quote.strip() if ref.quote else ""
                if not quote:
                    warnings.append(
                        StructuredError(
                            code=WarningCode.EVIDENCE_NOT_FOUND.value,
                            stage="evidence_validation",
                            message="Candidate rejected because cited evidence quote is empty.",
                            details={"unit_id": unit.unit_id},
                        )
                    )
                    candidate_valid = False
                    break

                combined_quotes.append(quote)

                # Rule 2: Quote matching via NFC + whitespace collapse, case-sensitive
                norm_quote, _, _ = build_normalized_index_map(quote)
                matches = self._find_all_substring_matches(norm_unit_text, norm_quote)

                if len(matches) == 0:
                    warnings.append(
                        StructuredError(
                            code=WarningCode.EVIDENCE_NOT_FOUND.value,
                            stage="evidence_validation",
                            message="Candidate rejected because cited quote was not found in active unit text.",
                            details={"unit_id": unit.unit_id, "quote_length": len(quote)},
                        )
                    )
                    candidate_valid = False
                    break

                elif len(matches) > 1:
                    # Rule 3: Multiple occurrences -> ambiguous_match
                    overall_evidence_status = EvidenceStatus.AMBIGUOUS_MATCH
                    evidence_records.append(
                        EvidenceRecord(
                            segment_id=unit.unit_id,
                            quote=quote,
                            location=None,  # No guessing first occurrence
                            match_count=len(matches),
                            status=EvidenceStatus.AMBIGUOUS_MATCH,
                        )
                    )
                    warnings.append(
                        StructuredError(
                            code=WarningCode.EVIDENCE_AMBIGUOUS.value,
                            stage="evidence_validation",
                            message="Evidence quote appears multiple times in unit; marked ambiguous without coordinate guessing.",
                            details={"unit_id": unit.unit_id, "match_count": len(matches)},
                        )
                    )

                else:
                    # Rule 4: Unique match -> derive coordinates from application metadata
                    norm_start = matches[0]
                    norm_end = norm_start + len(norm_quote)

                    unit_char_start = orig_starts[norm_start]
                    unit_char_end = orig_ends[norm_end - 1]

                    # Map to parent segment coordinates
                    parent_seg_char_start = unit.parent_start + unit_char_start
                    parent_seg_char_end = unit.parent_start + unit_char_end

                    # Map to document-level coordinates if parent segment has TextLocation
                    doc_location: SourceLocation
                    if isinstance(parent_segment.source_location, TextLocation):
                        base_doc_offset = parent_segment.source_location.start
                        doc_start = base_doc_offset + parent_seg_char_start
                        doc_end = base_doc_offset + parent_seg_char_end
                        doc_location = TextLocation(kind="text", start=doc_start, end=doc_end)
                        evidence_spans.append((parent_segment.segment_id, doc_start, doc_end))
                    else:
                        doc_location = parent_segment.source_location
                        evidence_spans.append(
                            (parent_segment.segment_id, parent_seg_char_start, parent_seg_char_end)
                        )

                    evidence_records.append(
                        EvidenceRecord(
                            segment_id=unit.unit_id,
                            quote=quote,
                            location=doc_location,
                            quote_char_span=(parent_seg_char_start, parent_seg_char_end),
                            match_count=1,
                            status=EvidenceStatus.MATCHED,
                        )
                    )

            if not candidate_valid:
                continue

            # Rule 5: Raw-Field Containment Check
            containment_ok, mismatch_field = self._verify_raw_field_containment(
                candidate, combined_quotes
            )
            if not containment_ok:
                warnings.append(
                    StructuredError(
                        code=WarningCode.EVIDENCE_FIELD_MISMATCH.value,
                        stage="evidence_validation",
                        message=f"Candidate rejected because field '{mismatch_field}' does not appear in cited quotes.",
                        details={"unit_id": unit.unit_id, "field": mismatch_field},
                    )
                )
                continue

            # Generate deterministic candidate ID
            raw_fields = {
                "value_raw": candidate.value_raw,
                "unit_raw": candidate.unit_raw,
                "reference_range_raw": candidate.reference_range_raw,
                "abnormal_flag_raw": candidate.abnormal_flag_raw,
                "date_raw": candidate.date_raw,
                "time_raw": candidate.time_raw,
                "qualifier_raw": candidate.qualifier_raw,
                "experiencer_relationship_raw": candidate.experiencer_relationship_raw,
            }

            candidate_id = compute_candidate_id(
                content_sha256=content_sha256,
                fact_type=candidate.fact_type,
                normalized_label=canonicalize_label(candidate.label_raw),
                raw_fields=raw_fields,
                assertion=candidate.assertion_status,
                temporality=candidate.temporality,
                experiencer=candidate.experiencer_type,
                role=candidate.reporting_source_role,
                evidence_spans=evidence_spans
                if evidence_spans
                else [(unit.unit_id, q) for q in combined_quotes],
            )

            fact_candidate = FactCandidate(
                candidate_id=candidate_id,
                fact_type=candidate.fact_type,
                label_raw=candidate.label_raw,
                value_raw=candidate.value_raw,
                unit_raw=candidate.unit_raw,
                reference_range_raw=candidate.reference_range_raw,
                abnormal_flag_raw=candidate.abnormal_flag_raw,
                date_raw=candidate.date_raw,
                time_raw=candidate.time_raw,
                qualifier_raw=candidate.qualifier_raw,
                assertion_status=candidate.assertion_status,
                temporality=candidate.temporality,
                experiencer_type=candidate.experiencer_type,
                experiencer_relationship_raw=candidate.experiencer_relationship_raw,
                reporting_source_role=candidate.reporting_source_role,
                evidence_records=evidence_records,
                evidence_status=overall_evidence_status,
                review_status=ReviewStatus.UNREVIEWED,
                provenance=provenance,
            )

            accepted_candidates.append(fact_candidate)

        return accepted_candidates, warnings

    def _find_all_substring_matches(self, text: str, sub: str) -> list[int]:
        """Finds all start indices of sub in text."""
        indices = []
        start = 0
        while True:
            idx = text.find(sub, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1
        return indices

    def _verify_raw_field_containment(
        self, candidate: ProviderCandidate, quotes: list[str]
    ) -> tuple[bool, str]:
        """
        Verifies that every non-empty raw field physically appears in the combined evidence quotes
        (normalized: NFC + whitespace collapse + casefold).
        """
        combined = " ".join(quotes)
        norm_combined = unicodedata.normalize("NFC", combined).casefold()
        norm_combined = " ".join(norm_combined.split())

        fields_to_check = [
            ("label_raw", candidate.label_raw),
            ("value_raw", candidate.value_raw),
            ("unit_raw", candidate.unit_raw),
            ("reference_range_raw", candidate.reference_range_raw),
            ("abnormal_flag_raw", candidate.abnormal_flag_raw),
            ("date_raw", candidate.date_raw),
            ("time_raw", candidate.time_raw),
            ("qualifier_raw", candidate.qualifier_raw),
            ("experiencer_relationship_raw", candidate.experiencer_relationship_raw),
        ]

        for field_name, field_val in fields_to_check:
            val = field_val.strip() if field_val else ""
            if not val:
                continue

            norm_val = unicodedata.normalize("NFC", val).casefold()
            norm_val = " ".join(norm_val.split())

            if norm_val not in norm_combined:
                return False, field_name

        return True, ""
