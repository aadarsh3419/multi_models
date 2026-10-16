"""Integration tests for the local CLI and composition root."""

import json
import subprocess
import sys


def test_cli_help():
    """Verify CLI --help exits 0 and prints usage."""
    res = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "extract-text" in res.stdout


def test_cli_extract_text_stdin_success(tmp_path):
    """Verify CLI reading from stdin produces valid AgentResult JSON to stdout."""
    sample_text = "Patient vital signs: Heart rate 72 bpm."
    res = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "extract-text", "--stdin"],
        input=sample_text,
        capture_output=True,
        text=True,
    )

    # When remote LLM is disabled (default offline test env), extraction is skipped -> exit code 3 (partial)
    assert res.returncode == 3
    result_data = json.loads(res.stdout)
    assert result_data["agent_name"] == "text"
    assert result_data["status"] == "partial"
    assert "Heart rate 72 bpm." not in res.stderr  # PHI not in stderr


def test_cli_extract_text_file_input(tmp_path):
    """Verify CLI reading from input file."""
    test_file = tmp_path / "patient_record.txt"
    test_file.write_text("Patient reports feeling dizzy.", encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "extract-text",
            "--input-file",
            str(test_file),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 3
    result_data = json.loads(res.stdout)
    assert result_data["status"] == "partial"


def test_cli_nonexistent_file():
    """Verify non-existent file returns exit code 2."""
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "extract-text",
            "--input-file",
            "nonexistent_file_path_12345.txt",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "does not exist" in res.stderr


def test_cli_empty_file(tmp_path):
    """Verify empty input returns exit code 2."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\t  ", encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "extract-text",
            "--input-file",
            str(empty_file),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "empty" in res.stderr.lower()


def test_cli_oversized_file(tmp_path):
    """Verify oversized file is rejected with exit code 2 before full processing."""
    oversized_file = tmp_path / "huge.txt"
    # Write text exceeding 36,000 characters
    oversized_file.write_text("A" * 36001, encoding="utf-8")

    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "healthcare_agents.cli",
            "extract-text",
            "--input-file",
            str(oversized_file),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "exceeds maximum allowed size" in res.stderr
