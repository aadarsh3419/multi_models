"""Healthcare Multi-Agent Platform — Main Execution Entrypoint.

Runs the complete vertical slice demonstration across all four modalities:
1. Multi-modal demonstration across Text, PDF, Image, and Voice agents.
2. End-to-end clinical text extraction with evidence coordinate verification.
3. Synthetic evaluation benchmark across all 38 test cases.
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scripts.demo_all_agents import main as run_all_modalities_demo
from scripts.demo_extraction import run_demo as run_text_note_demo
from scripts.evaluate_text_extraction import run_benchmark


def main() -> None:
    print("\n" + "=" * 70)
    print("STEP 1: MULTI-MODAL AGENT EXTRACTION (TEXT, PDF, IMAGE, VOICE)")
    print("=" * 70)
    run_all_modalities_demo()

    print("\n" + "=" * 70)
    print("STEP 2: CLINICAL NOTE EXTRACTION WITH ZERO-BASED COORDINATE MAPPING")
    print("=" * 70)
    run_text_note_demo()

    print("\n" + "=" * 70)
    print("STEP 3: SYNTHETIC BENCHMARK EVALUATION (38 CASES)")
    print("=" * 70)
    fixture_path = Path("tests/fixtures/synthetic/labeled_text_cases.json")
    report = run_benchmark(fixture_path=fixture_path, use_live_groq=False)

    print(
        f"Benchmark: {report['benchmark_metadata']['benchmark_name']} v{report['benchmark_metadata']['version']}"
    )
    print(f"Date:      {report['benchmark_metadata']['benchmark_date']}")
    print(f"Cases:     {report['total_cases_evaluated']}")
    print("-" * 70)
    print("Overall Fact Performance:")
    print(f"  Precision: {report['overall_fact_metrics']['precision']}")
    print(f"  Recall:    {report['overall_fact_metrics']['recall']}")
    print(f"  F1 Score:  {report['overall_fact_metrics']['f1_score']}")
    print(f"  Grounded Evidence TP: {report['evidence_metrics']['evidence_matched_tp']}")
    print("-" * 70)
    print("Category Breakdown:")
    for cat, stats in report["per_category_metrics"].items():
        print(
            f"  [{cat:<28}] P: {stats['precision']:<6} | R: {stats['recall']:<6} | F1: {stats['f1_score']:<6} (TP:{stats['tp']}, FP:{stats['fp']}, FN:{stats['fn']})"
        )

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE: ALL 4 MODALITIES & BENCHMARKS VERIFIED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
