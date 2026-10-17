"""
VoiceAgent: Modality agent for clinical audio recordings (consultation dictations, telehealth audio).

Enforces audio size and duration boundaries, coordinates AudioReader (magic sniffing,
bounded ffprobe metadata inspection, Transcriber protocol execution), and delegates
clinical extraction to the shared ExtractionWorkflow.
Zero direct provider SDK, PyMuPDF, or pytesseract imports in this module.
"""

import hashlib
import logging
import platform
import time
from datetime import UTC, datetime
from typing import Any

from healthcare_agents.agents.base import BaseAgent
from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts.requests import VoiceAgentRequest
from healthcare_agents.contracts.results import AgentResult, AuditMetadata
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.sources.audio_reader import AudioReader, Transcriber

logger = logging.getLogger("healthcare_agents.agents.voice_agent")


class VoiceAgent(BaseAgent):
    """
    Ingests clinical audio streams, extracts temporally-provenanced text segments
    via AudioReader, and delegates fact extraction to the shared ExtractionWorkflow.
    """

    def __init__(
        self,
        workflow: ExtractionWorkflow,
        reader: AudioReader | None = None,
        transcriber: Transcriber | None = None,
        settings: Settings | None = None,
    ):
        self.workflow = workflow
        self.settings = settings or get_settings()
        if reader is not None:
            self.reader = reader
        else:
            self.reader = AudioReader(
                transcriber=transcriber,
                settings=self.settings,
            )

    def _get_installed_versions(self) -> dict[str, str]:
        """Gathers runtime environment versions without violating layering boundaries."""
        return {
            "python": platform.python_version(),
        }

    def process(self, request: VoiceAgentRequest) -> AgentResult:
        """
        Ingests audio bytes, coordinates transcription and temporal mapping,
        and delegates candidate extraction to the shared ExtractionWorkflow.
        """
        start_time = time.perf_counter()
        content_len = len(request.content) if request.content else 0
        content_sha256 = hashlib.sha256(request.content).hexdigest() if request.content else ""

        # Strict PHI-free logging
        logger.info(
            "VoiceAgent: Processing request_id=%s, size=%d bytes, sha256=%s",
            request.request_id,
            content_len,
            content_sha256[:12],
        )

        audio_settings: dict[str, Any] = {
            "groq_stt_model": self.settings.groq_stt_model,
            "max_audio_bytes": self.settings.max_audio_bytes,
            "max_audio_duration_seconds": self.settings.max_audio_duration_seconds,
            "allow_remote_audio_transcription": self.settings.allow_remote_audio_transcription,
        }

        # 1. Read audio bytes into SourceTextBundle via AudioReader
        (
            bundle,
            overall_status,
            reader_errors,
            reader_warnings,
            stage_res,
            audio_details,
        ) = self.reader.read_bundle(
            content=request.content,
            request_id=request.request_id,
            language_hint=request.language_hint,
            options=request.options,
        )

        installed_versions = self._get_installed_versions()

        # 2. Handle reader failure (corrupt, missing duration, limit exceeded, no text)
        if bundle is None:
            logger.warning(
                "VoiceAgent: Ingestion failed for request_id=%s, errors=%d",
                request.request_id,
                len(reader_errors),
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            audit = AuditMetadata(
                provider="groq"
                if self.settings.allow_remote_audio_transcription
                else "unavailable",
                model_id=self.settings.groq_stt_model
                if self.settings.allow_remote_audio_transcription
                else "unavailable",
                schema_enforcement="none",
                prompt_version="1.0.0",
                prompt_sha256="unavailable",
                schema_version="1.0.0",
                python_version=platform.python_version(),
                installed_versions=installed_versions,
                key_settings=audio_settings,
                audio_details=audio_details,
                stt_model_id=self.settings.groq_stt_model,
            )

            fmt_name = audio_details.get("format_name", "unknown") if audio_details else "unknown"
            return AgentResult(
                schema_version="1.0.0",
                request_id=request.request_id,
                agent_name="voice_agent",
                source_type="audio",
                status="failed",
                sanitized_source_metadata={
                    "source_id": f"src_aud_{content_sha256[:12]}",
                    "media_type": f"audio/{fmt_name.lower()}",
                    "byte_count": content_len,
                    "content_sha256": content_sha256,
                    "duration_seconds": audio_details.get("duration_seconds")
                    if audio_details
                    else None,
                },
                content_sha256=content_sha256,
                candidates=[],
                stage_results={stage_res.stage_name: stage_res},
                warnings=reader_warnings,
                errors=reader_errors,
                audit=audit,
                created_at_utc=datetime.now(UTC),
                duration_ms=duration_ms,
            )

        # 3. Delegate clinical extraction across segmented bundle to shared ExtractionWorkflow
        result = self.workflow.process_bundle(
            bundle=bundle,
            language_hint=request.language_hint,
            date_order_hint=request.options.date_order_hint,
            options=request.options,
            agent_name="voice_agent",
            original_text=None,
        )

        # 4. Integrate reader results and audit provenance into final AgentResult
        result.stage_results["source_reading"] = stage_res
        if reader_warnings:
            result.warnings = reader_warnings + result.warnings
        if reader_errors:
            result.errors = reader_errors + result.errors

        if overall_status == "partial" and result.status == "success":
            result.status = "partial"

        # Enrich audit trail
        result.audit.installed_versions.update(installed_versions)
        result.audit.key_settings.update(audio_settings)
        result.audit.audio_details = audio_details
        result.audit.stt_model_id = self.settings.groq_stt_model

        total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        result.duration_ms = total_duration_ms

        logger.info(
            "VoiceAgent: Completed request_id=%s, status=%s, candidates=%d, duration=%.2fms",
            result.request_id,
            result.status,
            len(result.candidates),
            result.duration_ms,
        )

        return result
