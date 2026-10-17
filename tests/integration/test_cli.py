"""
Integration tests for CLI: in-process injection and subprocess smoke execution.

Tests exit codes:
  0: Success (all stages executed)
  2: Partial (e.g. remote LLM disabled)
  1: Failed (invalid input, nonexistent file, oversized file)
"""

import json
import subprocess
import sys

from healthcare_agents.agents.text_agent import TextAgent
from healthcare_agents.cli import main
from healthcare_agents.contracts.facts import (
    AssertionStatus,
    ExperiencerType,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    Temporality,
)
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.extraction.workflow import ExtractionWorkflow


class FakeExtractor(ClinicalFieldExtractor):
    def extract(self, unit: ProviderUnit, language_hint: str | None = None):
        cand = ProviderCandidate(
            fact_type=FactType.MEASUREMENT,
            label_raw="Heart rate",
            value_raw="72",
            unit_raw="bpm",
            reference_range_raw="",
            abnormal_flag_raw="",
            date_raw="",
            time_raw="",
            qualifier_raw="",
            assertion_status=AssertionStatus.AFFIRMED,
            temporality=Temporality.NOT_APPLICABLE,
            experiencer_type=ExperiencerType.PATIENT,
            experiencer_relationship_raw="",
            reporting_source_role=ReportingSourceRole.CLINICIAN,
            evidence_refs=[ProviderEvidenceRef(segment_id="s0001", quote="Heart rate 72 bpm")],
        )
        return ProviderExtractionResponse(candidates=[cand]), {
            "provider": "fake",
            "model_id": "fake_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "fake_sha",
            "attempt_count": 1,
            "token_usage": {},
        }


def test_cli_in_process_with_fake_extractor(capsys):
    """Verify in-process main() invocation with an injected fake agent factory exits 0."""

    def custom_factory(settings):
        workflow = ExtractionWorkflow(extractor=FakeExtractor())
        return TextAgent(workflow=workflow)

    exit_code = main(
        ["text", "--text", "Vitals: Heart rate 72 bpm."],
        agent_factory=custom_factory,
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    res_data = json.loads(captured.out)
    assert res_data["status"] == "success"
    assert len(res_data["candidates"]) == 1
    assert res_data["candidates"][0]["label_raw"] == "Heart rate"


def test_cli_subprocess_smoke_remote_disabled():
    """Verify CLI subprocess run with remote processing disabled returns exit code 2 (partial)."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "text",
            "--text",
            "Vital check: Heart rate 72 bpm.",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2, (
        f"Expected returncode 2 (partial), got {res.returncode}. Stderr: {res.stderr}"
    )

    result_data = json.loads(res.stdout)
    assert result_data["agent_name"] == "text_agent"
    assert result_data["status"] == "partial"
    assert any(e["code"] == "remote_processing_disabled" for e in result_data["errors"])
    # Verify no source text leaked to stderr
    assert "Heart rate 72 bpm" not in res.stderr


def test_cli_file_input_and_missing_file(tmp_path):
    """Verify reading from file and handling missing file."""
    # 1. Valid file
    valid_file = tmp_path / "valid.txt"
    valid_file.write_text("Patient feels well.", encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "text", "--file", str(valid_file)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2  # Partial because remote processing is disabled

    # 2. Non-existent file -> exit code 1
    res_missing = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "text", "--file", "missing_file_12345.txt"],
        capture_output=True,
        text=True,
    )
    assert res_missing.returncode == 1
    assert "does not exist" in res_missing.stderr


def test_cli_empty_input_rejected():
    """Verify empty text input returns exit code 1."""
    res = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "text", "--text", "   "],
        capture_output=True,
        text=True,
    )
    err_lower = res.stderr.lower()
    assert "empty" in err_lower or "whitespace" in err_lower
