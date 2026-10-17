"""
Integration test running real host ffprobe binary on synthetic WAV generated via stdlib wave.
"""

import shutil
import wave
from io import BytesIO
from uuid import uuid4

import pytest

from healthcare_agents.config import Settings
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    TranscriptionResult,
    TranscriptionSegment,
)
from tests.unit.test_audio_reader import FakeTranscriber


def make_wave_bytes(duration_sec: float = 1.5, sample_rate: int = 16000) -> bytes:
    """Generates standard uncompressed 16-bit PCM mono audio using stdlib wave."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


@pytest.mark.audio_integration
def test_audio_reader_real_ffprobe_execution():
    """Verify AudioReader can run host ffprobe against real WAV bytes if ffprobe is installed."""
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe executable not found on host PATH")

    content = make_wave_bytes(duration_sec=1.5)
    stt_res = TranscriptionResult(
        text="Normal breathing sound.",
        segments=[
            TranscriptionSegment(
                text="Normal breathing sound.",
                start_seconds=0.1,
                end_seconds=1.4,
            )
        ],
        language="english",
    )
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(canned_result=stt_res),
        settings=settings,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
    )

    assert status == "success"
    assert bundle is not None
    assert len(bundle.segments) == 1
    assert bundle.segments[0].source_location.start_seconds == 0.1
    assert details.get("duration_seconds") == pytest.approx(1.5, abs=0.1)

