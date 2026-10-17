"""
Audio Reader and Speech-to-Text Ingestion Engine.

Layering Rule: This module does NOT import groq, pymupdf, or pytesseract.
Enforces audio safety boundaries (magic sniffing before ffprobe, bounded Popen
ffprobe stream cap at 1MB, duration limits, tempfile cleanup, no transcoding).
Coordinates Transcriber protocol execution and assigns temporal AudioLocations.
"""

import hashlib
import json
import logging
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts.errors import ErrorCode, StructuredError, WarningCode
from healthcare_agents.contracts.locations import AudioLocation
from healthcare_agents.contracts.requests import VoiceAgentOptions
from healthcare_agents.contracts.results import StageResult
from healthcare_agents.contracts.source_text import SourceSegment, SourceTextBundle

logger = logging.getLogger("healthcare_agents.sources.audio_reader")

LANGUAGE_CODE_MAP: dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "fr": "fr",
    "fra": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "german": "de",
}


class TranscriptionSegment(BaseModel):
    """Atomic transcript segment with optional temporal boundaries."""

    text: str = Field(description="Transcribed segment text.")
    start_seconds: float | None = Field(
        default=None, ge=0.0, description="Segment start offset in seconds."
    )
    end_seconds: float | None = Field(
        default=None, ge=0.0, description="Segment end offset in seconds."
    )


class TranscriptionResult(BaseModel):
    """Complete speech-to-text response with detected language and granular segments."""

    text: str = Field(description="Full concatenated transcript.")
    segments: list[TranscriptionSegment] = Field(
        default_factory=list, description="Ordered transcript segments."
    )
    language: str | None = Field(
        default=None, description="Detected language identifier (e.g. 'english', 'en')."
    )


class TranscriberError(Exception):
    """Raised when an audio transcriber encounters an unrecoverable failure."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class Transcriber(Protocol):
    """Consumer-owned protocol for speech-to-text transcription."""

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult: ...


def sniff_audio_format(content: bytes) -> str | None:
    """
    Pre-ffprobe magic-byte allowlist sniffing.
    Returns extension string ('wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm') or None.
    Rejects unknown files immediately without invoking external processes.
    """
    if len(content) < 4:
        return None

    # RIFF/WAVE: 'RIFF' .... 'WAVE'
    if content[:4] == b"RIFF" and len(content) >= 12 and content[8:12] == b"WAVE":
        return "wav"

    # ID3 tag (MP3)
    if content[:3] == b"ID3":
        return "mp3"

    # MPEG-1/2 Audio Layer 3 frame sync: 11 set bits (0xFF, upper 3 bits of byte 1 set)
    if len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0:
        return "mp3"

    # Ogg container
    if content[:4] == b"OggS":
        return "ogg"

    # FLAC format
    if content[:4] == b"fLaC":
        return "flac"

    # ISO BMFF (MP4, M4A, etc.): '....ftyp'
    if len(content) >= 8 and content[4:8] == b"ftyp":
        return "m4a"

    # Matroska / WebM EBML header
    if content[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"

    return None


class AudioReader:
    """
    Ingests raw audio bytes, enforces safety checks (magic sniffing, bounded ffprobe
    inspection with 1MB cap, duration checks, tempfile cleanup), coordinates Transcriber
    protocol execution, and constructs temporally-provenanced SourceSegments.
    """

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        settings: Settings | None = None,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ):
        self.transcriber = transcriber
        self.settings = settings or get_settings()
        self._popen_factory = popen_factory

    def _run_ffprobe(
        self,
        content: bytes,
        ext: str,
    ) -> tuple[dict[str, Any] | None, StructuredError | None]:
        """
        Executes bounded ffprobe metadata inspection via safe temporary directory.
        Strictly limits output to 1MB and enforces timeout.
        """
        ffprobe_cmd = self.settings.ffprobe_path or "ffprobe"

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / f"inspect_{uuid4().hex[:8]}.{ext}"
            try:
                temp_path.write_bytes(content)
            except Exception as write_err:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_UNREADABLE.value,
                    stage="audio_validation",
                    message=f"Failed to stage audio for inspection: {write_err}",
                )

            args = [
                ffprobe_cmd,
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                "format=format_name,duration:stream=codec_type,codec_name,duration",
                "-of",
                "json",
                str(temp_path),
            ]

            try:
                proc = self._popen_factory(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except (FileNotFoundError, PermissionError, OSError) as os_err:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_VALIDATION_UNAVAILABLE.value,
                    stage="audio_validation",
                    message=f"ffprobe is unavailable for audio validation: {os_err}",
                    details={"cmd": ffprobe_cmd},
                )

            try:
                stdout_data, stderr_data = proc.communicate(
                    timeout=self.settings.ffprobe_timeout_seconds
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate()
                except Exception:
                    pass
                return None, StructuredError(
                    code=ErrorCode.AUDIO_UNREADABLE.value,
                    stage="audio_validation",
                    message="ffprobe inspection timed out.",
                    details={"timeout_seconds": self.settings.ffprobe_timeout_seconds},
                )

            max_inspection_bytes = 1024 * 1024
            if stdout_data and len(stdout_data.encode("utf-8")) > max_inspection_bytes:
                proc.kill()
                return None, StructuredError(
                    code=ErrorCode.AUDIO_UNREADABLE.value,
                    stage="audio_validation",
                    message="ffprobe output exceeded maximum inspection buffer (1MB).",
                )

            if proc.returncode != 0:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_UNREADABLE.value,
                    stage="audio_validation",
                    message=f"ffprobe validation failed with exit code {proc.returncode}.",
                    details={"stderr": stderr_data[:200] if stderr_data else ""},
                )

            try:
                data = json.loads(stdout_data or "{}")
            except Exception as json_err:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_UNREADABLE.value,
                    stage="audio_validation",
                    message=f"Failed to parse ffprobe JSON output: {json_err}",
                )

            # Check for at least one audio stream
            streams = data.get("streams", [])
            audio_stream = next(
                (s for s in streams if s.get("codec_type") == "audio"),
                None,
            )
            if not audio_stream:
                return None, StructuredError(
                    code=ErrorCode.UNSUPPORTED_FILE_TYPE.value,
                    stage="audio_validation",
                    message="No valid audio stream detected in file.",
                )

            # Duration check (format duration or stream duration)
            duration_raw = data.get("format", {}).get("duration") or audio_stream.get("duration")
            if duration_raw is None or str(duration_raw).strip() == "":
                return None, StructuredError(
                    code=ErrorCode.AUDIO_DURATION_UNAVAILABLE.value,
                    stage="audio_validation",
                    message="Audio duration could not be determined.",
                )

            try:
                duration = float(duration_raw)
            except (ValueError, TypeError):
                return None, StructuredError(
                    code=ErrorCode.AUDIO_DURATION_UNAVAILABLE.value,
                    stage="audio_validation",
                    message=f"Invalid duration value: {duration_raw}",
                )

            if duration <= 0.0:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_DURATION_UNAVAILABLE.value,
                    stage="audio_validation",
                    message=f"Audio duration is non-positive: {duration}s",
                )

            if duration > self.settings.max_audio_duration_seconds:
                return None, StructuredError(
                    code=ErrorCode.AUDIO_DURATION_EXCEEDED.value,
                    stage="audio_validation",
                    message=(
                        f"Audio duration ({duration:.1f}s) exceeds maximum allowed limit "
                        f"({self.settings.max_audio_duration_seconds}s)."
                    ),
                    details={
                        "duration_seconds": duration,
                        "limit": self.settings.max_audio_duration_seconds,
                    },
                )

            audio_details = {
                "format_name": data.get("format", {}).get("format_name", ext),
                "duration_seconds": round(duration, 3),
                "codec_name": audio_stream.get("codec_name", "unknown"),
            }
            return audio_details, None

    def read_bundle(
        self,
        content: bytes,
        request_id: UUID,
        language_hint: str | None = None,
        options: VoiceAgentOptions | None = None,
    ) -> tuple[
        SourceTextBundle | None,
        Literal["success", "partial", "failed"],
        list[StructuredError],
        list[StructuredError],
        StageResult,
        dict[str, Any],
    ]:
        """
        Ingests audio bytes, runs pre-ffprobe magic sniffing, bounds-checked ffprobe inspection,
        executes transcriber, and returns a temporally-provenanced SourceTextBundle.
        """
        content_len = len(content) if content else 0
        content_sha256 = hashlib.sha256(content).hexdigest() if content else ""
        errors: list[StructuredError] = []
        warnings_list: list[StructuredError] = []

        # 1. File size pre-check
        if content_len > self.settings.max_audio_bytes:
            err = StructuredError(
                code=ErrorCode.FILE_TOO_LARGE.value,
                stage="audio_validation",
                message=f"Audio file size ({content_len} bytes) exceeds maximum configured limit ({self.settings.max_audio_bytes} bytes).",
                details={"byte_count": content_len, "limit": self.settings.max_audio_bytes},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="audio_validation", status="failed")
            return None, "failed", errors, warnings_list, stage_res, {}

        # 2. Magic byte allowlist sniffing BEFORE invoking ffprobe
        ext = sniff_audio_format(content)
        if ext is None:
            err = StructuredError(
                code=ErrorCode.UNSUPPORTED_FILE_TYPE.value,
                stage="audio_validation",
                message="Unsupported or unrecognizable audio file format.",
                details={"magic_bytes": content[:8].hex() if len(content) >= 8 else ""},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="audio_validation", status="failed")
            return None, "failed", errors, warnings_list, stage_res, {}

        # 3. ffprobe inspection
        audio_details, probe_err = self._run_ffprobe(content, ext)
        if probe_err is not None:
            errors.append(probe_err)
            stage_res = StageResult(stage_name="audio_validation", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details or {}

        audio_details = audio_details or {}

        # 4. Remote transcription safety gate
        if not self.settings.allow_remote_audio_transcription:
            err = StructuredError(
                code=ErrorCode.REMOTE_PROCESSING_DISABLED.value,
                stage="transcription",
                message="Remote audio transcription is disabled by configuration (ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false).",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="transcription", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details

        # 5. Transcriber presence check
        if self.transcriber is None:
            err = StructuredError(
                code=ErrorCode.AUDIO_TRANSCRIPTION_FAILED.value,
                stage="transcription",
                message="No audio transcriber configured.",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="transcription", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details

        # 6. Execute transcription with generated upload name (never original filename)
        upload_name = f"audio_{uuid4().hex[:8]}.{ext}"
        try:
            stt_result = self.transcriber.transcribe(
                audio_bytes=content,
                filename=upload_name,
                language_hint=language_hint,
            )
        except TranscriberError as terr:
            err = StructuredError(
                code=terr.code,
                stage="transcription",
                message=terr.message,
                details=terr.details,
            )
            errors.append(err)
            stage_res = StageResult(stage_name="transcription", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details
        except Exception as gen_err:
            # Check for error code attribute if present
            code = getattr(gen_err, "code", ErrorCode.AUDIO_TRANSCRIPTION_FAILED.value)
            err = StructuredError(
                code=code,
                stage="transcription",
                message=f"Audio transcription failed: {type(gen_err).__name__}: {gen_err}",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="transcription", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details

        # 7. Check for language hint vs detected language mismatch
        if language_hint and stt_result.language:
            norm_hint = LANGUAGE_CODE_MAP.get(language_hint.strip().lower())
            norm_detected = LANGUAGE_CODE_MAP.get(stt_result.language.strip().lower())
            if norm_hint and norm_detected and norm_hint != norm_detected:
                warnings_list.append(
                    StructuredError(
                        code=WarningCode.TRANSCRIPT_LANGUAGE_MISMATCH.value,
                        stage="transcription",
                        message=(
                            f"Transcribed language '{stt_result.language}' differs "
                            f"from language hint '{language_hint}'."
                        ),
                        details={"hint": language_hint, "detected": stt_result.language},
                    )
                )

        # 8. Check for empty transcript
        full_text = (stt_result.text or "").strip()
        if not full_text:
            err = StructuredError(
                code=ErrorCode.NO_USABLE_SOURCE_CONTENT.value,
                stage="transcription",
                message="No usable clinical text produced from audio transcription.",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="transcription", status="failed")
            return None, "failed", errors, warnings_list, stage_res, audio_details

        # 9. Build SourceSegments with temporal provenance
        segments: list[SourceSegment] = []
        timestamps_missing = False

        if stt_result.segments:
            for idx, seg in enumerate(stt_result.segments, start=1):
                seg_text = seg.text.strip()
                if not seg_text:
                    continue

                if (
                    seg.start_seconds is not None
                    and seg.end_seconds is not None
                    and 0.0 <= seg.start_seconds <= seg.end_seconds
                ):
                    loc = AudioLocation(
                        kind="audio",
                        timestamps_available=True,
                        start_seconds=round(seg.start_seconds, 3),
                        end_seconds=round(seg.end_seconds, 3),
                    )
                else:
                    timestamps_missing = True
                    loc = AudioLocation(
                        kind="audio",
                        timestamps_available=False,
                        start_seconds=None,
                        end_seconds=None,
                    )

                segment = SourceSegment(
                    segment_id=f"aud_seg_{idx:04d}",
                    text=seg_text,
                    extraction_method="stt:whisper",
                    source_location=loc,
                )
                segments.append(segment)

        # Fallback if segments list was empty but full_text existed
        if not segments:
            timestamps_missing = True
            loc = AudioLocation(
                kind="audio",
                timestamps_available=False,
                start_seconds=None,
                end_seconds=None,
            )
            segment = SourceSegment(
                segment_id="aud_seg_0001",
                text=full_text,
                extraction_method="stt:whisper",
                source_location=loc,
            )
            segments.append(segment)

        if timestamps_missing:
            warnings_list.append(
                StructuredError(
                    code=WarningCode.TIMESTAMPS_UNAVAILABLE.value,
                    stage="transcription",
                    message="Fine-grained temporal timestamps were unavailable for one or more audio segments.",
                )
            )

        # 10. Construct SourceTextBundle
        bundle = SourceTextBundle(
            request_id=request_id,
            source_type="audio",
            extraction_method="stt:whisper",
            ordered_segments=segments,
            safe_metadata={
                "source_id": f"src_aud_{content_sha256[:12]}",
                "media_type": f"audio/{ext}",
                "format": ext,
                "byte_count": content_len,
                "content_sha256": content_sha256,
                "duration_seconds": audio_details.get("duration_seconds"),
            },
        )

        stage_res = StageResult(stage_name="source_reading", status="success")
        return bundle, "success", [], warnings_list, stage_res, audio_details
