"""Evaluation scorer and benchmarking script for healthcare text extraction.

Implements deterministic evaluation matching rules (Section 24):
- Exact category, label, value, unit, assertion, temporality, experiencer matching.
- Zero semantic fuzzy matching.
- Micro-level and per-category Precision, Recall, and F1 calculations.
- Clean 'N/A' reporting when denominators are zero.
- Offline fake benchmark double and opt-in live Groq evaluation.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Ensure 'src' is importable when script is run directly
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from healthcare_agents.config import get_settings
from healthcare_agents.contracts import (
    AgentResult,
    EvidenceReference,
    EvidenceStatus,
    ExtractionCandidateRaw,
    ExtractionResponse,
    FactCandidate,
    TextAgentOptions,
    TextAgentRequest,
)
from healthcare_agents.extractor_interface import (
    ClinicalFieldExtractor,
    ExtractorCallMetadata,
)
from healthcare_agents.groq_extractor import GroqFieldExtractor
from healthcare_agents.source_text import SourceSegment
from healthcare_agents.text_agent import TextAgent
from healthcare_agents.workflow import ExtractionWorkflow


def normalize_comparison_label(label: str | None) -> str:
    """Normalize label using Unicode NFC, case folding, and whitespace collapse."""
    if not label:
        return ""
    nfc = unicodedata.normalize("NFC", label)
    folded = nfc.casefold()
    import re

    return re.sub(r"\s+", " ", folded).strip()


def normalize_comparison_value(val: str | None) -> str:
    """Normalize raw value strictly by outer-whitespace trimming and whitespace collapse."""
    if val is None:
        return ""
    import re

    return re.sub(r"\s+", " ", str(val)).strip()


class SyntheticBenchmarkFakeExtractor(ClinicalFieldExtractor):
    """Offline test double injecting expected candidates grounded in actual segment text."""

    def __init__(self, expected_facts_by_text: dict[str, list[dict[str, Any]]]) -> None:
        self.expected_facts_by_text = expected_facts_by_text
        self.call_count = 0

    def extract(
        self,
        segments: list[SourceSegment],
        language_hint: str | None = None,
    ) -> tuple[ExtractionResponse, ExtractorCallMetadata]:
        self.call_count += 1
        candidates: list[ExtractionCandidateRaw] = []
        for text_key, facts in self.expected_facts_by_text.items():
            if any(s.exact_text in text_key or text_key in s.exact_text for s in segments):
                for f in facts:
                    # Ground evidence in the first segment containing the quote
                    refs: list[EvidenceReference] = []
                    for q in f.get("expected_quotes", []):
                        found_seg_id = segments[0].segment_id
                        for s in segments:
                            if q in s.exact_text:
                                found_seg_id = s.segment_id
                                break
                        refs.append(EvidenceReference(segment_id=found_seg_id, quote=q))

                    cand = ExtractionCandidateRaw(
                        fact_type=f["fact_type"],
                        label_raw=f["label_raw"],
                        value_raw=f.get("value_raw"),
                        unit_raw=f.get("unit_raw"),
                        reference_range_raw=f.get("reference_range_raw"),
                        abnormal_flag_raw=f.get("abnormal_flag_raw"),
                        date_raw=f.get("date_raw"),
                        assertion_status=f["assertion_status"],
                        temporality=f["temporality"],
                        experiencer_type=f["experiencer_type"],
                        reporting_source_role=f["reporting_source_role"],
                        evidence_refs=refs,
                    )
                    candidates.append(cand)
                break

        resp = ExtractionResponse(candidates=candidates)
        meta = ExtractorCallMetadata(
            provider_name="offline_synthetic_double",
            model_id="benchmark-fake-v1",
            response_id=f"bench_resp_{self.call_count}",
            input_tokens=100,
            output_tokens=50,
            attempts_made=1,
            duration_ms=1.0,
        )
        return resp, meta


class ExtractionScorer:
    """Computes exact-match precision and recall for clinical fact candidates."""

    @classmethod
    def match_fact(cls, predicted: FactCandidate, expected: dict[str, Any]) -> bool:
        """Check if predicted FactCandidate exactly matches expected specification."""
        # 1. Fact type
        if predicted.fact_type.value != expected["fact_type"]:
            return False

        # 2. Label comparison (NFC, casefold, whitespace collapse)
        if normalize_comparison_label(predicted.label_raw) != normalize_comparison_label(
            expected["label_raw"]
        ):
            return False

        # 3. Value comparison
        pred_val = normalize_comparison_value(predicted.value_raw)
        exp_val = normalize_comparison_value(expected.get("value_raw"))
        if pred_val != exp_val:
            return False

        # 4. Unit comparison (exact case and characters, trimmed)
        pred_unit = (predicted.unit_raw or "").strip()
        exp_unit = (expected.get("unit_raw") or "").strip()
        if pred_unit != exp_unit:
            return False

        # 5. Assertion status
        if predicted.assertion_status.value != expected["assertion_status"]:
            return False

        # 6. Temporality
        if predicted.temporality.value != expected["temporality"]:
            return False

        # 7. Experiencer type
        if predicted.experiencer_type.value != expected["experiencer_type"]:
            return False

        # 8. Reporting source role
        if "reporting_source_role" in expected:
            if predicted.reporting_source_role.value != expected["reporting_source_role"]:
                return False

        return True


def run_benchmark(
    fixture_path: Path,
    use_live_groq: bool = False,
) -> dict[str, Any]:
    """Execute evaluation benchmark and compute performance metrics."""
    with open(fixture_path, encoding="utf-8") as f:
        data = json.load(f)

    benchmark_meta = data["metadata"]
    cases = data["cases"]

    # Assemble expected facts map for fake extractor
    facts_by_text = {c["text"]: c["expected_facts"] for c in cases}

    settings = get_settings()
    if use_live_groq:
        if not settings.allow_remote_llm_processing or not settings.groq_api_key:
            raise RuntimeError(
                "Live Groq evaluation requested but ALLOW_REMOTE_LLM_PROCESSING is False or GROQ_API_KEY is unset."
            )
        extractor = GroqFieldExtractor(settings=settings)
    else:
        extractor = SyntheticBenchmarkFakeExtractor(expected_facts_by_text=facts_by_text)

    workflow = ExtractionWorkflow(
        extractor=extractor,
        date_order="UNKNOWN",
        max_llm_calls_per_request=8,
    )
    agent = TextAgent(workflow=workflow)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    evidence_tp = 0
    evidence_fn = 0

    category_stats: dict[str, dict[str, int]] = {}

    for case in cases:
        cat = case["category"]
        if cat not in category_stats:
            category_stats[cat] = {"tp": 0, "fp": 0, "fn": 0}

        req = TextAgentRequest(
            text=case["text"],
            language_hint=case.get("language"),
            options=TextAgentOptions(date_order="UNKNOWN"),
        )
        result: AgentResult = agent.process(req)

        expected_list = list(case["expected_facts"])
        predicted_list = list(result.candidates)

        matched_exp_indices = set()
        for pred in predicted_list:
            matched_idx = None
            for idx, exp in enumerate(expected_list):
                if idx not in matched_exp_indices and ExtractionScorer.match_fact(pred, exp):
                    matched_idx = idx
                    break

            if matched_idx is not None:
                total_tp += 1
                category_stats[cat]["tp"] += 1
                matched_exp_indices.add(matched_idx)

                # Check evidence quote grounding
                if pred.evidence_status == EvidenceStatus.matched:
                    evidence_tp += 1
                else:
                    evidence_fn += 1
            else:
                total_fp += 1
                category_stats[cat]["fp"] += 1

        unmatched_expected = len(expected_list) - len(matched_exp_indices)
        total_fn += unmatched_expected
        category_stats[cat]["fn"] += unmatched_expected

    def calc_metrics(tp: int, fp: int, fn: int) -> tuple[Any, Any, Any]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else "N/A"
        recall = tp / (tp + fn) if (tp + fn) > 0 else "N/A"
        if isinstance(precision, float) and isinstance(recall, float):
            f1 = (
                (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )
        else:
            f1 = "N/A"
        return precision, recall, f1

    micro_p, micro_r, micro_f1 = calc_metrics(total_tp, total_fp, total_fn)

    results = {
        "benchmark_metadata": benchmark_meta,
        "mode": "live_groq" if use_live_groq else "offline_test_double",
        "total_cases_evaluated": len(cases),
        "overall_fact_metrics": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": f"{micro_p:.4f}" if isinstance(micro_p, float) else micro_p,
            "recall": f"{micro_r:.4f}" if isinstance(micro_r, float) else micro_r,
            "f1_score": f"{micro_f1:.4f}" if isinstance(micro_f1, float) else micro_f1,
        },
        "evidence_metrics": {
            "evidence_matched_tp": evidence_tp,
            "evidence_unmatched_fn": evidence_fn,
        },
        "per_category_metrics": {},
    }

    for cat, stats in category_stats.items():
        cp, cr, cf1 = calc_metrics(stats["tp"], stats["fp"], stats["fn"])
        results["per_category_metrics"][cat] = {
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "precision": f"{cp:.4f}" if isinstance(cp, float) else cp,
            "recall": f"{cr:.4f}" if isinstance(cr, float) else cr,
            "f1_score": f"{cf1:.4f}" if isinstance(cf1, float) else cf1,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Healthcare Text Extraction Foundation.")
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/synthetic/labeled_text_cases.json",
        help="Path to synthetic labeled test cases JSON fixture.",
    )
    parser.add_argument(
        "--live-groq",
        action="store_true",
        help="Run live evaluation against Groq Cloud (requires ALLOW_REMOTE_LLM_PROCESSING=true and GROQ_API_KEY).",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        sys.stderr.write(f"Error: Fixture file '{fixture_path}' does not exist.\n")
        sys.exit(1)

    print(f"Running evaluation benchmark on {fixture_path} (Live Groq: {args.live_groq})...")
    report = run_benchmark(fixture_path=fixture_path, use_live_groq=args.live_groq)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    print(
        f"Benchmark: {report['benchmark_metadata']['benchmark_name']} v{report['benchmark_metadata']['version']}"
    )
    print(f"Date:      {report['benchmark_metadata']['benchmark_date']}")
    print(f"Mode:      {report['mode']}")
    print(f"Cases:     {report['total_cases_evaluated']}")
    print("-" * 60)
    print("Overall Fact Extraction Performance:")
    print(f"  True Positives (TP):  {report['overall_fact_metrics']['true_positives']}")
    print(f"  False Positives (FP): {report['overall_fact_metrics']['false_positives']}")
    print(f"  False Negatives (FN): {report['overall_fact_metrics']['false_negatives']}")
    print(f"  Precision:            {report['overall_fact_metrics']['precision']}")
    print(f"  Recall:               {report['overall_fact_metrics']['recall']}")
    print(f"  F1 Score:             {report['overall_fact_metrics']['f1_score']}")
    print("-" * 60)
    print("Evidence Grounding Metrics:")
    print(f"  Grounded Evidence TP: {report['evidence_metrics']['evidence_matched_tp']}")
    print(f"  Grounded Evidence FN: {report['evidence_metrics']['evidence_unmatched_fn']}")
    print("-" * 60)
    print("Category Breakdown:")
    for cat, stats in report["per_category_metrics"].items():
        print(
            f"  [{cat}] P: {stats['precision']} | R: {stats['recall']} | F1: {stats['f1_score']} (TP:{stats['tp']}, FP:{stats['fp']}, FN:{stats['fn']})"
        )
    print("=" * 60)
    print("NOTICE: " + report["benchmark_metadata"]["disclaimer"])
    print("=" * 60)


if __name__ == "__main__":
    main()
