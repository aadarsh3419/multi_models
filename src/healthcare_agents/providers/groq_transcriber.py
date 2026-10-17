"""
GroqTranscriber: Official Groq SDK Whisper adapter implementing Transcriber protocol.

Coordinates speech-to-text transcription with segment-level timestamp granularities,
delegating retry and backoff ownership to GroqClientWrapper.
"""

from healthcare_agents.providers.groq_client import GroqClientWrapper
from healthcare_agents.sources.audio_reader import (
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
)


class GroqTranscriber(Transcriber):
    """
    Adapter implementing Transcriber protocol via Groq speech-to-text API.
    """

    def __init__(
        self,
        client: GroqClientWrapper,
        model: str = "whisper-large-v3-turbo",
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Dispatches audio bytes to Groq Whisper STT endpoint with verbose_json formatting.
        """
        resp = self.client.create_transcription(
            file=(filename, audio_bytes),
            model=self.model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=language_hint,
            temperature=self.temperature,
        )

        # Extract text and language safely from response object or dict
        if isinstance(resp, dict):
            full_text = str(resp.get("text", ""))
            language = resp.get("language")
            raw_segments = resp.get("segments", [])
        else:
            full_text = str(getattr(resp, "text", ""))
            language = getattr(resp, "language", None)
            raw_segments = getattr(resp, "segments", []) or []

        segments: list[TranscriptionSegment] = []
        for raw_seg in raw_segments:
            if isinstance(raw_seg, dict):
                seg_text = str(raw_seg.get("text", "")).strip()
                start_val = raw_seg.get("start")
                end_val = raw_seg.get("end")
            else:
                seg_text = str(getattr(raw_seg, "text", "")).strip()
                start_val = getattr(raw_seg, "start", None)
                end_val = getattr(raw_seg, "end", None)

            if not seg_text:
                continue

            start_sec = float(start_val) if start_val is not None else None
            end_sec = float(end_val) if end_val is not None else None

            segments.append(
                TranscriptionSegment(
                    text=seg_text,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                )
            )

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=str(language) if language else None,
        )
