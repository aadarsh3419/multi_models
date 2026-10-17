"""
Unit tests for AudioReader, pre-ffprobe sniffing, ffprobe subprocess bounds, and temporal provenance.
"""

import json
import subprocess
from unittest.mock import MagicMock
from uuid import uuid4

from healthcare_agents.config import Settings
from healthcare_agents.contracts.errors import ErrorCode, WarningCode
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    TranscriptionResult,
    TranscriptionSegment,
    sniff_audio_format,
)


class FakeTranscriber:
    """In-memory fake Transcriber for deterministic testing."""

    def __init__(
        self,
        canned_result: TranscriptionResult | None = None,
        error: Exception | None = None,
    ):
        self.canned_result = canned_result or TranscriptionResult(
            text="Patient reports mild cough and fever.",
            segments=[
                TranscriptionSegment(
                    text="Patient reports mild cough and fever.",
                    start_seconds=0.5,
                    end_seconds=3.2,
                )
            ],
            language="english",
        )
        self.error = error
        self.last_audio_bytes: bytes | None = None
        self.last_filename: str | None = None
        self.last_language_hint: str | None = None

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        self.last_audio_bytes = audio_bytes
        self.last_filename = filename
        self.last_language_hint = language_hint
        if self.error:
            raise self.error
        return self.canned_result


def _make_dummy_wav_bytes(duration_sec: float = 1.0) -> bytes:
    """Constructs minimal valid RIFF WAVE header bytes."""
    # RIFF header (12 bytes) + 'fmt ' chunk (24 bytes) + 'data' chunk (8 bytes)
    # Total header is 44 bytes
    riff_header = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
    fmt_chunk = (
        b"fmt "
        + (16).to_bytes(4, "little")  # Subchunk1Size (16 for PCM)
        + (1).to_bytes(2, "little")  # AudioFormat (1 = PCM)
        + (1).to_bytes(2, "little")  # NumChannels (1 = Mono)
        + (8000).to_bytes(4, "little")  # SampleRate (8000 Hz)
        + (16000).to_bytes(4, "little")  # ByteRate (8000 * 1 * 2)
        + (2).to_bytes(2, "little")  # BlockAlign
        + (16).to_bytes(2, "little")  # BitsPerSample
    )
    num_samples = int(8000 * duration_sec)
    data_bytes = b"\x00\x00" * num_samples
    data_chunk = b"data" + len(data_bytes).to_bytes(4, "little") + data_bytes
    return riff_header + fmt_chunk + data_chunk


def _make_mock_popen(
    stdout_data: str = "",
    stderr_data: str = "",
    returncode: int = 0,
    timeout_on_communicate: bool = False,
):
    """Creates a mock subprocess.Popen instance."""
    mock_proc = MagicMock()
    mock_proc.returncode = returncode

    if timeout_on_communicate:
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=10.0)
    else:
        mock_proc.communicate.return_value = (stdout_data, stderr_data)

    mock_factory = MagicMock(return_value=mock_proc)
    return mock_factory, mock_proc


# ---------------------------------------------------------------------------
# Pre-ffprobe Magic Sniffing Tests
# ---------------------------------------------------------------------------


def test_sniff_audio_format_allowlist():
    """Verify magic byte detection for all allowed audio formats."""
    # WAV
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "
    assert sniff_audio_format(wav_bytes) == "wav"

    # ID3 MP3
    mp3_id3 = b"ID3\x04\x00\x00\x00"
    assert sniff_audio_format(mp3_id3) == "mp3"

    # Sync word MP3
    mp3_sync = b"\xff\xfb\x90\x64"
    assert sniff_audio_format(mp3_sync) == "mp3"

    # Ogg
    ogg_bytes = b"OggS\x00\x02\x00\x00"
    assert sniff_audio_format(ogg_bytes) == "ogg"

    # FLAC
    flac_bytes = b"fLaC\x00\x00\x00\x22"
    assert sniff_audio_format(flac_bytes) == "flac"

    # M4A / MP4
    m4a_bytes = b"\x00\x00\x00\x20ftypM4A "
    assert sniff_audio_format(m4a_bytes) == "m4a"

    # WebM
    webm_bytes = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81"
    assert sniff_audio_format(webm_bytes) == "webm"


def test_sniff_audio_format_rejects_unknown_and_short():
    """Verify unknown or truncated headers return None."""
    assert sniff_audio_format(b"") is None
    assert sniff_audio_format(b"RIF") is None
    assert sniff_audio_format(b"%PDF-1.7") is None
    assert sniff_audio_format(b"\x89PNG\r\n\x1a\n") is None


def test_audio_reader_rejects_unsupported_without_calling_ffprobe():
    """Verify unknown format is rejected before ever launching ffprobe."""
    mock_popen_factory, _ = _make_mock_popen()
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=b"NOT_AUDIO_DATA_AT_ALL",
        request_id=uuid4(),
    )

    assert status == "failed"
    assert bundle is None
    assert any(e.code == ErrorCode.UNSUPPORTED_FILE_TYPE.value for e in errors)
    # ffprobe MUST NOT have been called
    mock_popen_factory.assert_not_called()


# ---------------------------------------------------------------------------
# Patched ffprobe Popen Tests
# ---------------------------------------------------------------------------


def test_audio_reader_success_with_ffprobe_and_transcriber():
    """Verify successful audio validation and transcription pipeline."""
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "5.250"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "5.250"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber()
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    content = _make_dummy_wav_bytes(duration_sec=5.25)
    req_id = uuid4()
    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=content,
        request_id=req_id,
        language_hint="en",
    )

    assert status == "success"
    assert bundle is not None
    assert len(bundle.segments) == 1
    assert bundle.segments[0].source_location.kind == "audio"
    assert bundle.segments[0].source_location.timestamps_available is True
    assert bundle.segments[0].source_location.start_seconds == 0.5
    assert bundle.segments[0].source_location.end_seconds == 3.2
    assert details["duration_seconds"] == 5.25
    assert details["codec_name"] == "pcm_s16le"
    assert len(errors) == 0


def test_audio_reader_no_audio_stream():
    """Verify container without an audio stream produces unsupported_file_type."""
    ffprobe_json = json.dumps(
        {
            "format": {"format_name": "mov", "duration": "2.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=ffprobe_json)
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    m4a_dummy = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00"
    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=m4a_dummy,
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.UNSUPPORTED_FILE_TYPE.value for e in errors)


def test_audio_reader_missing_duration():
    """Verify audio file with missing duration produces audio_duration_unavailable."""
    ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=ffprobe_json)
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_DURATION_UNAVAILABLE.value for e in errors)


def test_audio_reader_exceeded_duration():
    """Verify audio exceeding max duration produces audio_duration_exceeded."""
    ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "601.5"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "601.5"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=ffprobe_json)
    settings = Settings(allow_remote_audio_transcription=True, max_audio_duration_seconds=600.0)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_DURATION_EXCEEDED.value for e in errors)


def test_audio_reader_ffprobe_nonzero_exit():
    """Verify ffprobe non-zero exit code produces audio_unreadable."""
    mock_popen_factory, _ = _make_mock_popen(
        stdout_data="", stderr_data="corrupt header", returncode=1
    )
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_UNREADABLE.value for e in errors)


def test_audio_reader_ffprobe_timeout():
    """Verify ffprobe execution timeout produces audio_unreadable and terminates process."""
    mock_popen_factory, mock_proc = _make_mock_popen(timeout_on_communicate=True)
    settings = Settings(allow_remote_audio_transcription=True, ffprobe_timeout_seconds=5.0)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_UNREADABLE.value for e in errors)
    mock_proc.kill.assert_called()


def test_audio_reader_1mb_overflow_killed():
    """Verify ffprobe output exceeding 1MB buffer cap produces audio_unreadable and kills process."""
    massive_stdout = "x" * (1024 * 1024 + 100)
    mock_popen_factory, mock_proc = _make_mock_popen(stdout_data=massive_stdout)
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_UNREADABLE.value for e in errors)
    mock_proc.kill.assert_called()


def test_audio_reader_missing_ffprobe():
    """Verify missing ffprobe executable produces audio_validation_unavailable."""

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("ffprobe not found on PATH")

    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=FakeTranscriber(),
        settings=settings,
        popen_factory=_raise_not_found,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert any(e.code == ErrorCode.AUDIO_VALIDATION_UNAVAILABLE.value for e in errors)


def test_hindi_language_hint_is_forwarded_to_transcriber():
    """Verify language hint ('hi') is passed directly to the transcriber."""
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "2.0"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "2.0"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber(
        canned_result=TranscriptionResult(
            text="मरीज़ को हल्का बुखार है।",
            segments=[
                TranscriptionSegment(
                    text="मरीज़ को हल्का बुखार है।", start_seconds=0.0, end_seconds=1.8
                )
            ],
            language="hindi",
        )
    )
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(duration_sec=2.0),
        request_id=uuid4(),
        language_hint="hi",
    )

    assert status == "success"
    assert fake_transcriber.last_language_hint == "hi"
    assert fake_transcriber.last_filename.startswith("audio_")
    assert fake_transcriber.last_filename.endswith(".wav")


def test_transcript_language_mismatch_warning():
    """Verify warning is added when transcribed language differs from language hint."""
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "2.0"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "2.0"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber(
        canned_result=TranscriptionResult(
            text="Patient has mild fever.",
            segments=[
                TranscriptionSegment(
                    text="Patient has mild fever.", start_seconds=0.0, end_seconds=1.8
                )
            ],
            language="hindi",  # Mismatch with hint 'en'
        )
    )
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
        language_hint="en",
    )

    assert status == "success"
    assert any(w.code == WarningCode.TRANSCRIPT_LANGUAGE_MISMATCH.value for w in warnings)


def test_timestamps_unavailable_fallback():
    """Verify missing segment timestamps gracefully fallback to timestamps_available=False with warning."""
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "2.0"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "2.0"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber(
        canned_result=TranscriptionResult(
            text="No timestamp segment here.",
            segments=[
                TranscriptionSegment(
                    text="No timestamp segment here.",
                    start_seconds=None,
                    end_seconds=None,
                )
            ],
            language="english",
        )
    )
    settings = Settings(allow_remote_audio_transcription=True)
    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "success"
    assert bundle.segments[0].source_location.timestamps_available is False
    assert bundle.segments[0].source_location.start_seconds is None
    assert bundle.segments[0].source_location.end_seconds is None
    assert any(w.code == WarningCode.TIMESTAMPS_UNAVAILABLE.value for w in warnings)


def test_remote_transcription_disabled_safety_gate():
    """Verify allow_remote_audio_transcription=False stops before calling transcriber."""
    valid_ffprobe_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "2.0"},
            "streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "duration": "2.0"}],
        }
    )
    mock_popen_factory, _ = _make_mock_popen(stdout_data=valid_ffprobe_json)
    fake_transcriber = FakeTranscriber()
    settings = Settings(allow_remote_audio_transcription=False)
    reader = AudioReader(
        transcriber=fake_transcriber,
        settings=settings,
        popen_factory=mock_popen_factory,
    )

    bundle, status, errors, warnings, stage_res, details = reader.read_bundle(
        content=_make_dummy_wav_bytes(),
        request_id=uuid4(),
    )

    assert status == "failed"
    assert bundle is None
    assert any(e.code == ErrorCode.REMOTE_PROCESSING_DISABLED.value for e in errors)
    # Transcriber MUST NOT have been called
    assert fake_transcriber.last_audio_bytes is None
