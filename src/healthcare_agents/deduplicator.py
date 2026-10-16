"""Deterministic candidate ID generation, exact deduplication, and conflict detection.

Enforces:
- Deterministic candidate IDs derived from source hash, evidence spans, and canonical fields.
- Deduplication only when all semantic fields and exact original evidence spans match.
- Defensible conflict detection only when explicit grouping context (label, date, experiencer) matches.
- Never merges or deletes distinct observations based on medical similarity.
"""

import re
import unicodedata
from hashlib import sha256

from healthcare_agents.contracts import FactCandidate


def normalize_comparison_key(text: str | None) -> str:
    """Normalize text strictly for comparison keys without modifying source text.

    Steps:
    1. Unicode NFC normalization.
    2. Case folding.
    3. Collapse whitespace runs to a single space.
    4. Strip leading/trailing whitespace.
    """
    if text is None:
        return ""
    nfc = unicodedata.normalize("NFC", text)
    folded = nfc.casefold()
    collapsed = re.sub(r"\s+", " ", folded).strip()
    return collapsed


def generate_candidate_id(
    source_content_hash: str,
    fact_type: str,
    label_raw: str,
    value_raw: str | None,
    unit_raw: str | None,
    assertion_status: str,
    temporality: str,
    experiencer_type: str,
    evidence_spans: list[tuple[str, int | None, int | None]],
) -> str:
    """Generate a deterministic, stable candidate ID.

    Combines:
    - Source content SHA-256 hash.
    - Canonical comparison keys of semantic fields.
    - Sorted evidence span coordinates (segment_id, start_offset, end_offset).

    Guarantees that identical facts at different positions receive distinct IDs,
    while processing identical spans produces identical IDs regardless of chunk order.
    """
    norm_label = normalize_comparison_key(label_raw)
    norm_val = normalize_comparison_key(value_raw)
    norm_unit = (unit_raw or "").strip()
    norm_assert = assertion_status.lower().strip()
    norm_temp = temporality.lower().strip()
    norm_exp = experiencer_type.lower().strip()

    # Sort evidence spans to ensure deterministic hashing
    sorted_spans = sorted(
        (span[0], -1 if span[1] is None else span[1], -1 if span[2] is None else span[2])
        for span in evidence_spans
    )
    spans_repr = ";".join(f"{s[0]}:{s[1]}:{s[2]}" for s in sorted_spans)

    seed_str = (
        f"{source_content_hash}|{fact_type}|{norm_label}|{norm_val}|"
        f"{norm_unit}|{norm_assert}|{norm_temp}|{norm_exp}|{spans_repr}"
    )
    digest = sha256(seed_str.encode("utf-8")).hexdigest()
    return f"cnd_{digest[:16]}"


class CandidateDeduplicator:
    """Handles exact candidate deduplication and defensible conflict detection."""

    @classmethod
    def deduplicate_and_detect_conflicts(
        cls,
        candidates: list[FactCandidate],
    ) -> tuple[list[FactCandidate], list[str]]:
        """Deduplicate exact redundant observations and detect defensible conflicts.

        Returns:
            Tuple of (deduplicated FactCandidate list, list of conflict warning strings)
        """
        if not candidates:
            return [], []

        deduped: list[FactCandidate] = []
        seen_keys: set[str] = set()

        # Step 1: Exact Deduplication
        for c in candidates:
            # Build span identifier from all evidence references
            spans = sorted(
                (
                    ref.segment_id,
                    ref.start_offset if ref.start_offset is not None else -1,
                    ref.end_offset if ref.end_offset is not None else -1,
                )
                for ref in c.evidence_refs
            )
            spans_key = ";".join(f"{s[0]}:{s[1]}:{s[2]}" for s in spans)

            dedup_key = (
                f"{c.fact_type.value}|"
                f"{normalize_comparison_key(c.label_raw)}|"
                f"{normalize_comparison_key(c.value_raw)}|"
                f"{(c.unit_raw or '').strip()}|"
                f"{(c.date_raw or '').strip()}|"
                f"{(c.time_raw or '').strip()}|"
                f"{c.assertion_status.value}|"
                f"{c.temporality.value}|"
                f"{c.experiencer_type.value}|"
                f"{spans_key}"
            )

            if dedup_key in seen_keys:
                # Redundant duplicate (e.g. from overlapping chunks) -> safely drop
                continue

            seen_keys.add(dedup_key)
            deduped.append(c)

        # Step 2: Conflict Detection
        # Only flag conflicts if an exact defensible context groups observations:
        # Same normalized label, same explicit date/time, same experiencer,
        # but differing values or conflicting units.
        warnings: list[str] = []
        group_map: dict[str, list[FactCandidate]] = {}

        for c in deduped:
            norm_label = normalize_comparison_key(c.label_raw)
            norm_date = (c.date_raw or "").strip()
            norm_time = (c.time_raw or "").strip()
            norm_exp = c.experiencer_type.value

            # Only group if an explicit date/time or specific anchor is present
            if norm_date:
                group_key = f"{norm_label}|{norm_date}|{norm_time}|{norm_exp}"
                group_map.setdefault(group_key, []).append(c)

        for _group_key, group_cands in group_map.items():
            if len(group_cands) > 1:
                # Check for differing values or conflicting units
                distinct_values = {normalize_comparison_key(c.value_raw) for c in group_cands}
                distinct_units = {(c.unit_raw or "").strip() for c in group_cands}

                if len(distinct_values) > 1 or len(distinct_units) > 1:
                    cand_ids = [c.candidate_id for c in group_cands]
                    warning_msg = (
                        f"Conflict detected for '{group_cands[0].label_raw}' on date "
                        f"'{group_cands[0].date_raw}' among candidates: {cand_ids}. "
                        f"Differing values or units recorded."
                    )
                    warnings.append(warning_msg)

                    # Cross-reference conflict warning IDs in the candidates
                    for c in group_cands:
                        other_ids = [cid for cid in cand_ids if cid != c.candidate_id]
                        for other_id in other_ids:
                            if other_id not in c.conflict_warning_refs:
                                c.conflict_warning_refs.append(other_id)

        return deduped, warnings
