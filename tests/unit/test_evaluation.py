"""
Unit tests for the evaluation scoring engine and dataset sanity checks.

Tests bipartite matching, degraded precision/recall scenarios, and ensures
every quote in dev and held-out datasets exists verbatim in its case text.
"""

import json
from decimal import Decimal
from pathlib import Path

from healthcare_agents.contracts.facts import (
    AssertionStatus,
    EvidenceRecord,
    EvidenceStatus,
    ExperiencerType,
    FactCandidate,
    FactType,
    ReportingSourceRole,
    ReviewStatus,
    Temporality,
)
from healthcare_agents.contracts.parsing import ParseStatus, ValueParseResult
from healthcare_agents.evaluation import ExtractionScorer


def make_fact(label: str, value: str, unit: str, assertion: str = "affirmed") -> FactCandidate:
    return FactCandidate(
        candidate_id=f"fc_{label}_{value}",
        fact_type=FactType.MEASUREMENT,
        label_raw=label,
        value_raw=value,
        unit_raw=unit,
        reference_range_raw="",
        abnormal_flag_raw="",
        date_raw="",
        time_raw="",
        qualifier_raw="",
        assertion_status=AssertionStatus(assertion),
        temporality=Temporality.NOT_APPLICABLE,
        experiencer_type=ExperiencerType.PATIENT,
        experiencer_relationship_raw="",
        reporting_source_role=ReportingSourceRole.CLINICIAN,
        evidence_records=[
            EvidenceRecord(
                segment_id="s0001",
                quote=f"{label} {value} {unit}",
                status=EvidenceStatus.MATCHED,
            )
        ],
        evidence_status=EvidenceStatus.MATCHED,
        parse_status=ParseStatus.PARSED,
        review_status=ReviewStatus.UNREVIEWED,
        value_parse_result=ValueParseResult(
            raw_value=value,
            value_kind="number",
            parse_status=ParseStatus.PARSED,
            numeric_value=Decimal(value) if value.isdigit() else None,
        ),
    )


def test_scorer_perfect_match():
    gold = [
        {
            "case_id": "case_1",
            "categories": ["measurements"],
            "gold_facts": [
                {
                    "fact_type": "measurement",
                    "label_raw": "Heart rate",
                    "value_raw": "72",
                    "unit_raw": "bpm",
                    "assertion_status": "affirmed",
                    "temporality": "not_applicable",
                    "experiencer_type": "patient",
                    "experiencer_relationship_raw": "",
                    "reporting_source_role": "clinician",
                    "reference_range_raw": "",
                    "abnormal_flag_raw": "",
                    "date_raw": "",
                }
            ],
        }
    ]

    preds = {"case_1": [make_fact("Heart rate", "72", "bpm")]}

    report = ExtractionScorer.evaluate_predictions(gold, preds)

    assert report["metrics"]["micro_precision"] == 1.0
    assert report["metrics"]["micro_recall"] == 1.0
    assert report["metrics"]["micro_f1"] == 1.0
    assert report["metrics"]["strict_score"] == 1.0
    assert report["counts"]["true_positives"] == 1
    assert report["counts"]["false_positives"] == 0
    assert report["counts"]["false_negatives"] == 0


def test_scorer_degraded_precision_and_recall():
    gold = [
        {
            "case_id": "case_1",
            "categories": ["measurements"],
            "gold_facts": [
                {
                    "fact_type": "measurement",
                    "label_raw": "Heart rate",
                    "value_raw": "72",
                    "unit_raw": "bpm",
                    "assertion_status": "affirmed",
                    "temporality": "not_applicable",
                    "experiencer_type": "patient",
                    "experiencer_relationship_raw": "",
                    "reporting_source_role": "clinician",
                    "reference_range_raw": "",
                    "abnormal_flag_raw": "",
                    "date_raw": "",
                },
                {
                    "fact_type": "measurement",
                    "label_raw": "Respiratory rate",
                    "value_raw": "16",
                    "unit_raw": "breaths/min",
                    "assertion_status": "affirmed",
                    "temporality": "not_applicable",
                    "experiencer_type": "patient",
                    "experiencer_relationship_raw": "",
                    "reporting_source_role": "clinician",
                    "reference_range_raw": "",
                    "abnormal_flag_raw": "",
                    "date_raw": "",
                },
            ],
        }
    ]

    # Predicts 1 TP (Heart rate) and 1 FP (SpO2) -> missed Respiratory rate (1 FN)
    preds = {
        "case_1": [
            make_fact("Heart rate", "72", "bpm"),
            make_fact("SpO2", "98", "%"),
        ]
    }

    report = ExtractionScorer.evaluate_predictions(gold, preds)

    # TP: 1, FP: 1, FN: 1 -> P: 0.5, R: 0.5, F1: 0.5
    assert report["counts"]["true_positives"] == 1
    assert report["counts"]["false_positives"] == 1
    assert report["counts"]["false_negatives"] == 1
    assert report["metrics"]["micro_precision"] == 0.5
    assert report["metrics"]["micro_recall"] == 0.5
    assert report["metrics"]["micro_f1"] == 0.5


def test_dataset_sanity_checks():
    """Verify that all quotes in dev and held-out cases exist verbatim in their respective case text."""
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation"
    dev_path = eval_dir / "dev_cases.json"
    heldout_path = eval_dir / "heldout_cases.json"

    assert dev_path.exists()
    assert heldout_path.exists()

    with open(dev_path, encoding="utf-8") as f:
        dev_cases = json.load(f)

    with open(heldout_path, encoding="utf-8") as f:
        heldout_cases = json.load(f)

    assert len(dev_cases) >= 20, f"Expected at least 20 dev cases, got {len(dev_cases)}"
    assert len(heldout_cases) >= 12, f"Expected at least 12 heldout cases, got {len(heldout_cases)}"

    for split_name, cases in [("dev", dev_cases), ("heldout", heldout_cases)]:
        for c in cases:
            text = c["text"]
            for f in c.get("gold_facts", []):
                for ref in f.get("evidence_refs", []):
                    quote = ref["quote"]
                    assert quote in text, (
                        f"[{split_name}] Quote '{quote}' not found in case {c['case_id']}"
                    )
