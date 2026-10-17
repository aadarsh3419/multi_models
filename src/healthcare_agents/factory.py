"""
Factory module: Composition root for building TextAgent instances.

Only factory.py (and test suites) may import providers.*.
Enforces the remote processing safety gate without instantiating network clients when disabled.
"""

from typing import Any

from healthcare_agents.agents.image_agent import ImageAgent
from healthcare_agents.agents.pdf_agent import PDFAgent
from healthcare_agents.agents.text_agent import TextAgent
from healthcare_agents.agents.voice_agent import VoiceAgent
from healthcare_agents.config import Settings
from healthcare_agents.contracts.errors import ErrorCode
from healthcare_agents.contracts.facts import ProviderExtractionResponse
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.deduplication import CandidateDeduplicator
from healthcare_agents.extraction.evidence_validator import EvidenceValidator
from healthcare_agents.extraction.field_extractor import (
    ClinicalFieldExtractor,
    ExtractorUnavailable,
)
from healthcare_agents.extraction.value_parser import ValueParser
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.ocr.service import OcrService, TesseractOcrService
from healthcare_agents.providers.groq_client import GroqClientWrapper
from healthcare_agents.providers.groq_field_extractor import GroqFieldExtractor
from healthcare_agents.providers.groq_transcriber import GroqTranscriber
from healthcare_agents.sources.audio_reader import AudioReader, Transcriber
from healthcare_agents.sources.chunker import TextChunker
from healthcare_agents.sources.image_reader import ImageReader
from healthcare_agents.sources.pdf_reader import PdfReader


class _UnavailableExtractor(ClinicalFieldExtractor):
    """Fallback extractor wired when remote processing is disabled or API key is absent."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message

    def extract(
        self,
        unit: ProviderUnit,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        raise ExtractorUnavailable(code=self.code, message=self.message)


class _UnavailableTranscriber(Transcriber):
    """Fallback transcriber wired when remote audio transcription is disabled or API key absent."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_hint: str | None = None,
    ) -> Any:
        raise ExtractorUnavailable(code=self.code, message=self.message)


def _build_extractor(
    cfg: Settings,
    extractor: ClinicalFieldExtractor | None = None,
) -> ClinicalFieldExtractor:
    if extractor is not None:
        return extractor
    if not cfg.allow_remote_llm_processing:
        return _UnavailableExtractor(
            code=ErrorCode.REMOTE_PROCESSING_DISABLED.value,
            message="Remote LLM processing is disabled by configuration (ALLOW_REMOTE_LLM_PROCESSING=false).",
        )
    if not cfg.groq_api_key or not cfg.groq_api_key.get_secret_value().strip():
        return _UnavailableExtractor(
            code=ErrorCode.MISSING_API_KEY.value,
            message="Groq API key is not configured (GROQ_API_KEY is empty).",
        )

    # Construct Groq client and adapter
    client_wrapper = GroqClientWrapper(
        api_key=cfg.groq_api_key.get_secret_value(),
        timeout_seconds=cfg.provider_timeout_seconds,
        max_retries=cfg.provider_max_retries,
        max_backoff_seconds=cfg.max_backoff_seconds,
    )
    return GroqFieldExtractor(
        client=client_wrapper,
        model=cfg.groq_text_model,
        schema_mode=cfg.groq_schema_mode,
        max_completion_tokens=cfg.groq_max_completion_tokens,
    )


def _build_workflow(
    cfg: Settings,
    extractor: ClinicalFieldExtractor | None = None,
) -> ExtractionWorkflow:
    active_extractor = _build_extractor(cfg, extractor)
    value_parser = ValueParser(bp_labels=cfg.bp_labels)
    validator = EvidenceValidator()
    chunker = TextChunker(
        max_input_chars=cfg.max_llm_input_chars,
        overlap_chars=cfg.chunk_overlap_chars,
    )
    deduplicator = CandidateDeduplicator()

    return ExtractionWorkflow(
        extractor=active_extractor,
        validator=validator,
        value_parser=value_parser,
        chunker=chunker,
        deduplicator=deduplicator,
        date_order=cfg.date_order,
        max_llm_calls_per_request=cfg.max_llm_calls_per_request,
        max_consecutive_chunk_failures=cfg.max_consecutive_chunk_failures,
    )


def build_text_agent(
    settings: Settings | None = None,
    extractor: ClinicalFieldExtractor | None = None,
) -> TextAgent:
    """
    Constructs a fully wired TextAgent with dependency inversion.
    """
    cfg = settings or Settings()
    workflow = _build_workflow(cfg, extractor)

    return TextAgent(
        workflow=workflow,
        max_text_chars=cfg.max_text_chars,
    )


def build_pdf_agent(
    settings: Settings | None = None,
    extractor: ClinicalFieldExtractor | None = None,
    ocr_service: OcrService | None = None,
    reader: PdfReader | None = None,
) -> PDFAgent:
    """
    Constructs a fully wired PDFAgent with dependency inversion.
    """
    cfg = settings or Settings()
    workflow = _build_workflow(cfg, extractor)
    active_ocr = ocr_service or TesseractOcrService(tesseract_cmd=cfg.tesseract_cmd)
    active_reader = reader or PdfReader(ocr_service=active_ocr, settings=cfg)

    return PDFAgent(
        workflow=workflow,
        reader=active_reader,
        ocr_service=active_ocr,
        settings=cfg,
    )


def build_image_agent(
    settings: Settings | None = None,
    extractor: ClinicalFieldExtractor | None = None,
    ocr_service: OcrService | None = None,
    reader: ImageReader | None = None,
) -> ImageAgent:
    """
    Constructs a fully wired ImageAgent with dependency inversion.
    Shares the exact same OcrService implementation pattern.
    """
    cfg = settings or Settings()
    workflow = _build_workflow(cfg, extractor)
    active_ocr = ocr_service or TesseractOcrService(tesseract_cmd=cfg.tesseract_cmd)
    active_reader = reader or ImageReader(ocr_service=active_ocr, settings=cfg)

    return ImageAgent(
        workflow=workflow,
        reader=active_reader,
        ocr_service=active_ocr,
        settings=cfg,
    )


def build_voice_agent(
    settings: Settings | None = None,
    extractor: ClinicalFieldExtractor | None = None,
    transcriber: Transcriber | None = None,
    reader: AudioReader | None = None,
) -> VoiceAgent:
    """
    Constructs a fully wired VoiceAgent with dependency inversion.
    Shares the exact same ExtractionWorkflow instance type across all agents.
    """
    cfg = settings or Settings()
    workflow = _build_workflow(cfg, extractor)

    if reader is not None:
        active_reader = reader
    else:
        if transcriber is not None:
            active_transcriber = transcriber
        elif not cfg.allow_remote_audio_transcription:
            active_transcriber = _UnavailableTranscriber(
                code=ErrorCode.REMOTE_PROCESSING_DISABLED.value,
                message="Remote audio transcription is disabled by configuration (ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false).",
            )
        elif not cfg.groq_api_key or not cfg.groq_api_key.get_secret_value().strip():
            active_transcriber = _UnavailableTranscriber(
                code=ErrorCode.MISSING_API_KEY.value,
                message="Groq API key is not configured for voice transcription.",
            )
        else:
            client_wrapper = GroqClientWrapper(
                api_key=cfg.groq_api_key.get_secret_value(),
                timeout_seconds=cfg.provider_timeout_seconds,
                max_retries=cfg.provider_max_retries,
                max_backoff_seconds=cfg.max_backoff_seconds,
            )
            active_transcriber = GroqTranscriber(
                client=client_wrapper,
                model=cfg.groq_stt_model,
            )

        active_reader = AudioReader(transcriber=active_transcriber, settings=cfg)

    return VoiceAgent(
        workflow=workflow,
        reader=active_reader,
        settings=cfg,
    )
