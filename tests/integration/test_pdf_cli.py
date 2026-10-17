"""
Integration tests for PDF CLI: in-process injection, file handling, and subprocess execution.
"""

import json
import subprocess
import sys
from pathlib import Path

import pymupdf

from healthcare_agents.agents.pdf_agent import PDFAgent
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
from healthcare_agents.ocr.service import FakeOcrService


class FakePdfExtractor(ClinicalFieldExtractor):
    def extract(self, unit: ProviderUnit, language_hint: str | None = None):
        cand = ProviderCandidate(
            fact_type=FactType.MEASUREMENT,
            label_raw="Blood Pressure",
            value_raw="120/80",
            unit_raw="mmHg",
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
            evidence_refs=[
                ProviderEvidenceRef(segment_id=unit.unit_id, quote="Blood Pressure: 120/80 mmHg")
            ],
        )
        return ProviderExtractionResponse(candidates=[cand]), {
            "provider": "fake",
            "model_id": "fake_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "fake_sha",
            "attempt_count": 1,
            "token_usage": {},
        }


def make_test_pdf_file(file_path: Path, lines: list[str]) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 24
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_pdf_cli_in_process_with_fake_extractor(tmp_path, capsys):
    pdf_file = make_test_pdf_file(
        tmp_path / "clinic_record.pdf",
        [
            "Clinical Vitals Summary",
            "Patient sitting comfortably.",
            "Blood Pressure: 120/80 mmHg recorded.",
        ],
    )

    def custom_pdf_factory(settings):
        workflow = ExtractionWorkflow(extractor=FakePdfExtractor())
        return PDFAgent(workflow=workflow, ocr_service=FakeOcrService(), settings=settings)

    exit_code = main(
        ["pdf", "--file", str(pdf_file)],
        pdf_agent_factory=custom_pdf_factory,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    stdout_json = captured.out[captured.out.find("{") :]
    res_data = json.loads(stdout_json)
    assert res_data["agent_name"] == "pdf_agent"
    assert res_data["status"] == "success"
    assert len(res_data["candidates"]) == 1
    assert res_data["candidates"][0]["label_raw"] == "Blood Pressure"


def test_pdf_cli_subprocess_smoke_remote_disabled(tmp_path):
    pdf_file = make_test_pdf_file(
        tmp_path / "smoke.pdf",
        [
            "Patient Examination Record",
            "Blood Pressure: 120/80 mmHg.",
            "Heart rate regular.",
        ],
    )

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "pdf",
            "--file",
            str(pdf_file),
        ],
        capture_output=True,
        text=True,
    )

    # Partial exit code (2) because remote LLM processing is disabled by default
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}. Stderr: {res.stderr}"

    stdout_json = res.stdout[res.stdout.find("{") :]
    result_data = json.loads(stdout_json)
    assert result_data["agent_name"] == "pdf_agent"
    assert result_data["status"] == "partial"
    assert any(e["code"] == "remote_processing_disabled" for e in result_data["errors"])
    # Diagnostics printed to stderr without leaking PHI/source content
    assert "PDFAgent Execution" in res.stderr
    assert "Blood Pressure: 120/80 mmHg" not in res.stderr


def test_pdf_cli_missing_file_returns_code_1():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "pdf",
            "--file",
            "nonexistent_file_xyz_987.pdf",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "does not exist" in res.stderr


def test_pdf_cli_invalid_ocr_languages_option_returns_code_1(tmp_path):
    pdf_file = make_test_pdf_file(tmp_path / "test.pdf", ["Clinical note line."])

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "pdf",
            "--file",
            str(pdf_file),
            "--ocr-languages",
            "invalid_lang_code!!",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "ocr_languages" in res.stderr.lower() or "error" in res.stderr.lower()
