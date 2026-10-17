"""
Integration tests for Voice CLI: in-process injection, file validation, and subprocess execution.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from healthcare_agents.agents.voice_agent import VoiceAgent
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
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    TranscriptionResult,
    TranscriptionSegment,
)
from tests.unit.test_audio_reader import (
    FakeTranscriber,
    _make_dummy_wav_bytes,
    _make_mock_popen,
)


class FakeVoiceExtractor(ClinicalFieldExtractor):
    """Stub extractor providing a deterministic clinical fact for in-process Voice CLI testing."""

    def extract(self, unit: ProviderUnit, language_hint: str | None = None):
        cand = ProviderCandidate(
            fact_type=FactType.CONDITION,
            label_raw="Cough",
            value_raw="",
            unit_raw="",
            reference_range_raw="",
            abnormal_flag_raw="",
            date_raw="",
            time_raw="",
            qualifier_raw="",
            assertion_status=AssertionStatus.AFFIRMED,
            temporality=Temporality.CURRENT,
            experiencer_type=ExperiencerType.PATIENT,
            experiencer_relationship_raw="",
            reporting_source_role=ReportingSourceRole.PATIENT,
            evidence_refs=[ProviderEvidenceRef(segment_id=unit.unit_id, quote="persistent cough")],
        )
        return ProviderExtractionResponse(candidates=[cand]), {
            "provider": "fake",
            "model_id": "fake_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "fake_sha",
            "attempt_count": 1,
            "token_usage": {},
        }


def make_test_audio_file(file_path: Path, duration_sec: float = 2.0) -> Path:
    data = _make_dummy_wav_bytes(duration_sec=duration_sec)
    file_path.write_bytes(data)
    return file_path


def test_voice_cli_in_process_with_fake_extractor(tmp_path, capsys):
    """Verify in-process CLI execution with injected voice agent factory produces valid JSON and exit 0."""
    audio_file = make_test_audio_file(tmp_path / "consultation.wav")

    stt_res = TranscriptionResult(
        text="Patient presents with persistent cough for three days.",
        segments=[
            TranscriptionSegment(
                text="Patient presents with persistent cough for three days.",
                start_seconds=0.5,
                end_seconds=2.0,
            )
        ],
        language="english",
    )
    fake_transcriber = FakeTranscriber(canned_result=stt_res)

    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "2.0"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "2.0"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)

    def custom_voice_factory(settings):
        settings.allow_remote_audio_transcription = True
        reader = AudioReader(
            transcriber=fake_transcriber,
            settings=settings,
            popen_factory=mock_popen_factory,
        )
        workflow = ExtractionWorkflow(extractor=FakeVoiceExtractor())
        return VoiceAgent(workflow=workflow, reader=reader, settings=settings)

    exit_code = main(
        ["voice", "--file", str(audio_file)],
        voice_agent_factory=custom_voice_factory,
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    result_data = json.loads(captured.out)
    assert result_data["status"] == "success"
    assert len(result_data["candidates"]) == 1
    assert result_data["candidates"][0]["label_raw"] == "Cough"
    assert "VoiceAgent Execution: status=success" in captured.err


def test_voice_cli_file_not_found(capsys):
    """Verify CLI returns exit code 1 when the specified audio file does not exist."""
    exit_code = main(["voice", "--file", "nonexistent_audio_recording.wav"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Specified audio file does not exist" in captured.err


def test_voice_cli_empty_file(tmp_path, capsys):
    """Verify CLI returns exit code 1 when the audio file is 0 bytes."""
    empty_file = tmp_path / "empty_audio.wav"
    empty_file.touch()

    exit_code = main(["voice", "--file", str(empty_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Audio file is empty" in captured.err


def test_voice_cli_help(capsys):
    """Verify --help returns exit code 0 and displays voice subcommand flags."""
    with pytest.raises(SystemExit) as excinfo:
        main(["voice", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--file" in captured.out
    assert "--lang" in captured.out
    assert "--date-order" in captured.out


def test_voice_cli_subprocess_smoke():
    """Verify CLI runs cleanly as a standalone subprocess."""
    proc = subprocess.run(
        [sys.executable, "-m", "healthcare_agents.cli", "voice", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--file" in proc.stdout
    assert "--lang" in proc.stdout
    assert "--date-order" in proc.stdout
