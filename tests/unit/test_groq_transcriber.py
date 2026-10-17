"""
Unit tests for GroqTranscriber adapter and GroqClientWrapper transcription retry logic.
"""

from unittest.mock import MagicMock

import pytest

from healthcare_agents.contracts.errors import ErrorCode
from healthcare_agents.extraction.field_extractor import ExtractorUnavailable
from healthcare_agents.providers.groq_transcriber import GroqTranscriber
from healthcare_agents.sources.audio_reader import TranscriptionResult


def test_groq_transcriber_success():
    """Verify GroqTranscriber parses verbose_json response correctly."""
    mock_client = MagicMock()
    mock_resp = {
        "text": "Patient has mild cough and fever.",
        "language": "english",
        "segments": [
            {
                "id": 0,
                "text": "Patient has mild cough and fever.",
                "start": 0.5,
                "end": 3.2,
            }
        ],
    }
    mock_client.create_transcription.return_value = mock_resp

    transcriber = GroqTranscriber(client=mock_client, model="whisper-large-v3-turbo")
    res = transcriber.transcribe(
        audio_bytes=b"FAKE_AUDIO",
        filename="audio_12345678.wav",
        language_hint="en",
    )

    assert isinstance(res, TranscriptionResult)
    assert res.text == "Patient has mild cough and fever."
    assert res.language == "english"
    assert len(res.segments) == 1
    assert res.segments[0].start_seconds == 0.5
    assert res.segments[0].end_seconds == 3.2

    mock_client.create_transcription.assert_called_once_with(
        file=("audio_12345678.wav", b"FAKE_AUDIO"),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        timestamp_granularities=["segment"],
        language="en",
        temperature=0.0,
    )


def test_groq_transcriber_error_propagation():
    """Verify provider errors are passed through to caller."""
    mock_client = MagicMock()
    mock_client.create_transcription.side_effect = ExtractorUnavailable(
        code=ErrorCode.FILE_TOO_LARGE.value,
        message="Audio exceeds size limit",
    )

    transcriber = GroqTranscriber(client=mock_client)
    with pytest.raises(ExtractorUnavailable) as exc_info:
        transcriber.transcribe(
            audio_bytes=b"LARGE_AUDIO",
            filename="audio_large.wav",
        )

    assert exc_info.value.code == ErrorCode.FILE_TOO_LARGE.value
