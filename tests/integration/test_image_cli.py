"""
Integration tests for Image CLI: in-process injection, file validation, and subprocess execution.
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from healthcare_agents.agents.image_agent import ImageAgent
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
from healthcare_agents.ocr.service import FakeOcrService, OCRLine, OCRResult, OCRWord


class FakeImageExtractor(ClinicalFieldExtractor):
    """Stub extractor providing a deterministic clinical fact for in-process CLI testing."""

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


def make_test_image_file(file_path: Path, width: int = 800, height: int = 600) -> Path:
    img = Image.new("RGB", (width, height), color="white")
    img.save(str(file_path), format="PNG")
    return file_path


def test_image_cli_in_process_with_fake_extractor(tmp_path, capsys):
    image_file = make_test_image_file(tmp_path / "rx_scan.png")

    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text="Blood Pressure: 120/80 mmHg recorded today.",
            lines=[
                OCRLine(
                    text="Blood Pressure: 120/80 mmHg recorded today.",
                    bbox=[20, 30, 400, 60],
                    words=[
                        OCRWord(text="Blood", bbox=[20, 30, 70, 60], confidence=95.0),
                        OCRWord(text="Pressure:", bbox=[75, 30, 150, 60], confidence=95.0),
                        OCRWord(text="120/80", bbox=[155, 30, 220, 60], confidence=95.0),
                        OCRWord(text="mmHg", bbox=[225, 30, 275, 60], confidence=95.0),
                        OCRWord(text="recorded", bbox=[280, 30, 340, 60], confidence=95.0),
                        OCRWord(text="today.", bbox=[345, 30, 400, 60], confidence=95.0),
                    ],
                )
            ],
            words=[],
        )
    )

    def custom_image_factory(settings):
        workflow = ExtractionWorkflow(extractor=FakeImageExtractor())
        return ImageAgent(workflow=workflow, ocr_service=fake_ocr, settings=settings)

    exit_code = main(
        ["image", "--file", str(image_file)],
        image_agent_factory=custom_image_factory,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    stdout_json = captured.out[captured.out.find("{") :]
    res_data = json.loads(stdout_json)

    assert res_data["agent_name"] == "image_agent"
    assert res_data["source_type"] == "image"
    assert res_data["status"] == "success"
    assert len(res_data["candidates"]) == 1
    assert res_data["candidates"][0]["label_raw"] == "Blood Pressure"
    assert res_data["candidates"][0]["value_raw"] == "120/80"


def test_image_cli_help_option():
    """Verify image --help prints CLI usage flags."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "image",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--file" in res.stdout
    assert "--ocr-languages" in res.stdout
    assert "--no-upscale" in res.stdout


def test_image_cli_missing_file_returns_code_1():
    """Verify nonexistent image path exits with code 1."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "image",
            "--file",
            "nonexistent_image_scan_xyz_999.png",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "does not exist" in res.stderr


def test_image_cli_invalid_ocr_languages_option_returns_code_1(tmp_path):
    """Verify regex-violating language code exits with code 1."""
    image_file = make_test_image_file(tmp_path / "valid.png")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "image",
            "--file",
            str(image_file),
            "--ocr-languages",
            "INVALID_LANG!!",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "ocr_languages" in res.stderr.lower() or "error" in res.stderr.lower()


def test_image_cli_empty_file_returns_code_1(tmp_path):
    """Verify empty 0-byte image file exits with code 1."""
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "image",
            "--file",
            str(empty_file),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "empty" in res.stderr.lower()
