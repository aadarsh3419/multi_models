"""
Deterministic Candidate Deduplication and Conflict Detection.

Relies exclusively on exact key comparisons and spatial/chronological context.
Zero fuzzy matching, embeddings, or medical synonym inference.
Zero provider SDK imports.
"""

from healthcare_agents.contracts.errors import StructuredError, WarningCode
from healthcare_agents.contracts.facts import FactCandidate
from healthcare_agents.contracts.locations import TextLocation
from healthcare_agents.sources.normalization import canonicalize_label


class CandidateDeduplicator:
    """
    Performs exact deduplication of duplicate candidates across chunks
    and flags potential clinical conflicts within defensible exact contexts.
    """

    @classmethod
    def deduplicate_and_detect_conflicts(
        cls,
        candidates: list[FactCandidate],
    ) -> tuple[list[FactCandidate], list[StructuredError]]:
        """
        Deduplicates exact matches and flags potential contradictions.
        Returns:
            list[FactCandidate]: Deduplicated candidates.
            list[StructuredError]: Potential conflict warnings.
        """
        if not candidates:
            return [], []

        deduplicated: list[FactCandidate] = []
        warnings: list[StructuredError] = []

        # ----------------------------------------------------------------------
        # 1. Exact Deduplication
        # ----------------------------------------------------------------------
        # Two facts are duplicates iff identity key AND evidence spans match exactly.
        seen_fingerprints: set[str] = set()

        for cand in candidates:
            fp = cls._compute_dedup_fingerprint(cand)
            if fp in seen_fingerprints:
                # Exact duplicate from overlapping chunks -> drop safely
                continue
            seen_fingerprints.add(fp)
            deduplicated.append(cand)

        # ----------------------------------------------------------------------
        # 2. Defensible Conflict Detection
        # ----------------------------------------------------------------------
        # Conflict condition: Same normalized label + same experiencer + same explicit date/time (or same table row id)
        # BUT differing values, differing units, or opposing assertions (affirmed vs negated).
        for i in range(len(deduplicated)):
            for j in range(i + 1, len(deduplicated)):
                c1 = deduplicated[i]
                c2 = deduplicated[j]

                if cls._is_potential_conflict(c1, c2):
                    warning = StructuredError(
                        code=WarningCode.POTENTIAL_CONFLICT.value,
                        stage="post_processing",
                        message=(
                            f"Potential clinical conflict detected between candidates '{c1.candidate_id}' "
                            f"and '{c2.candidate_id}' for label '{c1.label_raw}' with differing values or assertions."
                        ),
                        details={
                            "candidate_id_a": c1.candidate_id,
                            "candidate_id_b": c2.candidate_id,
                            "label": c1.label_raw,
                            "value_a": c1.value_raw,
                            "value_b": c2.value_raw,
                            "assertion_a": str(c1.assertion_status),
                            "assertion_b": str(c2.assertion_status),
                        },
                    )
                    warnings.append(warning)
                    c1.warnings.append(f"potential_conflict:{c2.candidate_id}")
                    c2.warnings.append(f"potential_conflict:{c1.candidate_id}")

        return deduplicated, warnings

    @classmethod
    def _compute_dedup_fingerprint(cls, cand: FactCandidate) -> str:
        """
        Builds a canonical fingerprint encompassing semantic identity AND exact physical span.
        """
        spans: list[str] = []
        for ref in cand.evidence_records:
            if isinstance(ref.location, TextLocation):
                spans.append(f"{ref.segment_id}:{ref.location.start}-{ref.location.end}")
            else:
                spans.append(f"{ref.segment_id}:{ref.quote}")
        spans_str = "|".join(sorted(spans))

        norm_label = canonicalize_label(cand.label_raw)
        norm_val = " ".join(cand.value_raw.strip().split())
        norm_unit = " ".join(cand.unit_raw.strip().split())

        return (
            f"{cand.fact_type}|{norm_label}|{norm_val}|{norm_unit}|"
            f"{cand.reference_range_raw}|{cand.date_raw}|{cand.time_raw}|"
            f"{cand.assertion_status}|{cand.temporality}|{cand.experiencer_type}|"
            f"{cand.experiencer_relationship_raw}|{spans_str}"
        )

    @classmethod
    def _is_potential_conflict(cls, c1: FactCandidate, c2: FactCandidate) -> bool:
        """
        Determines if two candidates exhibit a defensible clinical contradiction.
        """
        # Labels must match under strict canonicalization ("Hb" != "Hemoglobin")
        if canonicalize_label(c1.label_raw) != canonicalize_label(c2.label_raw):
            return False

        # Must share the same experiencer
        if c1.experiencer_type != c2.experiencer_type:
            return False

        # Context Check 1: Matching Table Row ID (if present in context)
        row1 = c1.context.get("row_id")
        row2 = c2.context.get("row_id")
        has_same_row = bool(row1 and row2 and row1 == row2)

        # Context Check 2: Same explicit non-empty date and time
        has_same_date = bool(
            c1.date_raw and c2.date_raw and c1.date_raw.strip() == c2.date_raw.strip()
        )
        has_same_time = bool(c1.time_raw == c2.time_raw)

        if not (has_same_row or (has_same_date and has_same_time)):
            # Insufficient context to defensibly assert conflict
            return False

        # Check for opposing assertion (affirmed vs negated)
        if (c1.assertion_status.value == "affirmed" and c2.assertion_status.value == "negated") or (
            c1.assertion_status.value == "negated" and c2.assertion_status.value == "affirmed"
        ):
            return True

        # Check for differing values or units
        norm_val1 = " ".join(c1.value_raw.strip().split())
        norm_val2 = " ".join(c2.value_raw.strip().split())
        if norm_val1 and norm_val2 and norm_val1 != norm_val2:
            return True

        norm_unit1 = " ".join(c1.unit_raw.strip().split())
        norm_unit2 = " ".join(c2.unit_raw.strip().split())
        if norm_unit1 and norm_unit2 and norm_unit1 != norm_unit2:
            return True

        return False
