"""
ImageAgent: Modality agent for clinical document images (prescriptions, reports, lab slips).

Enforces image size and pixel boundaries, coordinates ImageReader (Pillow safety,
EXIF transposition, upscaling, OCR token grouping), and delegates clinical extraction
to the shared ExtractionWorkflow.
Zero direct provider SDK, PyMuPDF, or pytesseract imports in this module.
"""

import hashlib
import logging
import platform
import time
from datetime import UTC, datetime
from typing import Any

import PIL

from healthcare_agents.agents.base import BaseAgent
from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts.requests import ImageAgentRequest
from healthcare_agents.contracts.results import AgentResult, AuditMetadata
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.ocr.service import OcrService, TesseractOcrService
from healthcare_agents.sources.image_reader import ImageReader

logger = logging.getLogger("healthcare_agents.agents.image_agent")


class ImageAgent(BaseAgent):
    """
    Ingests clinical document images (scanned reports, lab slips, photo prescriptions),
    extracts spatially-provenanced text segments via ImageReader and OcrService,
    and delegates fact extraction to the shared ExtractionWorkflow.
    """

    def __init__(
        self,
        workflow: ExtractionWorkflow,
        reader: ImageReader | None = None,
        ocr_service: OcrService | None = None,
        settings: Settings | None = None,
    ):
        self.workflow = workflow
        self.settings = settings or get_settings()
        if reader is not None:
            self.reader = reader
        else:
            active_ocr = ocr_service or TesseractOcrService(
                tesseract_cmd=self.settings.tesseract_cmd
            )
            self.reader = ImageReader(ocr_service=active_ocr, settings=self.settings)

    def _get_installed_versions(self) -> dict[str, str]:
        """Gathers runtime environment versions without violating layering boundaries."""
        tesseract_ver = "unavailable"
        try:
            tesseract_ver = self.reader.ocr_service.get_version()
        except Exception:
            pass

        return {
            "python": platform.python_version(),
            "pillow": getattr(PIL, "__version__", "unknown"),
            "pytesseract": getattr(self.reader.ocr_service, "pytesseract_version", "unknown"),
            "tesseract_engine": tesseract_ver,
        }

    def process(self, request: ImageAgentRequest) -> AgentResult:
        """
        Ingests image bytes, coordinates OCR and coordinate mapping,
        and delegates candidate extraction to the shared ExtractionWorkflow.
        """
        start_time = time.perf_counter()
        content_len = len(request.content) if request.content else 0
        content_sha256 = hashlib.sha256(request.content).hexdigest() if request.content else ""

        # PHI-free logging
        logger.info(
            "ImageAgent: Processing request_id=%s, size=%d bytes, sha256=%s",
            request.request_id,
            content_len,
            content_sha256[:12],
        )

        ocr_settings: dict[str, Any] = {
            "ocr_languages": request.options.ocr_languages,
            "ocr_low_confidence": self.settings.ocr_low_confidence,
            "image_upscale_min_side": self.settings.image_upscale_min_side,
            "upscale_enabled": request.options.upscale_enabled,
            "ocr_timeout_seconds": self.settings.ocr_timeout_seconds,
        }

        # 1. Read image bytes into SourceTextBundle via ImageReader
        (
            bundle,
            overall_status,
            reader_errors,
            reader_warnings,
            stage_res,
            image_details,
        ) = self.reader.read_bundle(
            content=request.content,
            request_id=request.request_id,
            options=request.options,
        )

        installed_versions = self._get_installed_versions()

        # 2. Handle reader failure (corrupt, unreadable, bomb, limit exceeded, no text)
        if bundle is None:
            logger.warning(
                "ImageAgent: Ingestion failed for request_id=%s, errors=%d",
                request.request_id,
                len(reader_errors),
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            audit = AuditMetadata(
                provider="unavailable",
                model_id="unavailable",
                schema_enforcement="none",
                prompt_version="1.0.0",
                prompt_sha256="unavailable",
                schema_version="1.0.0",
                python_version=platform.python_version(),
                installed_versions=installed_versions,
                key_settings={
                    "max_image_bytes": self.settings.max_image_bytes,
                    "max_image_pixels": self.settings.max_image_pixels,
                    "image_upscale_min_side": self.settings.image_upscale_min_side,
                    "ocr_low_confidence": self.settings.ocr_low_confidence,
                },
                ocr_settings=ocr_settings,
                image_details=image_details,
            )

            return AgentResult(
                schema_version="1.0.0",
                request_id=request.request_id,
                agent_name="image_agent",
                source_type="image",
                status="failed",
                sanitized_source_metadata={
                    "source_id": f"src_img_{content_sha256[:12]}",
                    "media_type": f"image/{image_details.get('format', 'unknown').lower()}",
                    "byte_count": content_len,
                    "content_sha256": content_sha256,
                },
                content_sha256=content_sha256,
                candidates=[],
                stage_results={"source_reading": stage_res},
                warnings=reader_warnings,
                errors=reader_errors,
                audit=audit,
                created_at_utc=datetime.now(UTC),
                duration_ms=duration_ms,
            )

        # 3. Delegate clinical extraction across segmented bundle to shared ExtractionWorkflow
        result = self.workflow.process_bundle(
            bundle=bundle,
            language_hint=None,
            date_order_hint=request.options.date_order_hint,
            options=request.options,
            agent_name="image_agent",
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
        result.audit.key_settings.update(
            {
                "max_image_bytes": self.settings.max_image_bytes,
                "max_image_pixels": self.settings.max_image_pixels,
                "image_upscale_min_side": self.settings.image_upscale_min_side,
                "ocr_low_confidence": self.settings.ocr_low_confidence,
            }
        )
        result.audit.ocr_settings = ocr_settings
        result.audit.image_details = image_details

        total_duration_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        result.duration_ms = total_duration_ms

        logger.info(
            "ImageAgent: Completed request_id=%s, status=%s, candidates=%d, duration=%.2fms",
            result.request_id,
            result.status,
            len(result.candidates),
            result.duration_ms,
        )

        return result
