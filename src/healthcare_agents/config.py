"""
Configuration and runtime settings for Healthcare Text Foundation.

Uses pydantic-settings to manage environment variables and defaults.
Settings are passed explicitly to components; no module reads env vars at import time.
"""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central runtime configuration for Healthcare Text Agent and workflow components.
    Defaults represent conservative prototype safety limits, not provider quotas.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq Provider Settings
    groq_api_key: SecretStr | None = Field(
        default=None,
        description="Groq API key. Redacted from repr and logs.",
    )
    groq_text_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Target Groq model. Unverified until confirmed by live provider catalog.",
    )
    groq_schema_mode: Literal["strict", "json_object"] = Field(
        default="strict",
        description="Structured outputs schema enforcement mode.",
    )
    groq_max_completion_tokens: int = Field(
        default=4096,
        ge=128,
        le=32768,
        description="Maximum completion token budget per provider call.",
    )

    @property
    def max_completion_tokens(self) -> int:
        return self.groq_max_completion_tokens

    @property
    def groq_response_mode(self) -> str:
        return self.groq_schema_mode

    # Remote Processing Safety Gate
    allow_remote_llm_processing: bool = Field(
        default=False,
        description="Safety kill-switch. When False, no outbound network requests are made.",
    )

    # Text & Workflow Budget Limits
    max_text_chars: int = Field(
        default=100_000,
        ge=1,
        description="Maximum allowed characters for a single request.",
    )
    max_llm_input_chars: int = Field(
        default=6000,
        ge=500,
        le=20_000,
        description="Maximum character length of each chunk dispatched to LLM.",
    )
    chunk_overlap_chars: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Character overlap between consecutive fragments of an oversized segment.",
    )
    max_llm_calls_per_request: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum sequential LLM calls permitted for a single request.",
    )
    max_consecutive_chunk_failures: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Stop processing if this many consecutive chunks fail.",
    )

    # Network & Retry Policy
    provider_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Client-side timeout for provider API calls.",
    )
    provider_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries for retryable status codes (429, 5xx, timeouts).",
    )
    max_backoff_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Maximum delay before aborting a rate-limited request.",
    )

    # Extraction Parsing Rules
    date_order: Literal["UNKNOWN", "DMY", "MDY"] = Field(
        default="UNKNOWN",
        description="Default interpretation for ambiguous numeric dates (e.g. 05/06/2025).",
    )
    bp_labels: list[str] = Field(
        default=["bp", "blood pressure"],
        description="Labels recognized for strict blood pressure systolic/diastolic parsing.",
    )

    @field_validator("bp_labels", mode="before")
    @classmethod
    def parse_bp_labels(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [x.strip().casefold() for x in v.split(",") if x.strip()]
        if isinstance(v, list):
            return [str(x).strip().casefold() for x in v if str(x).strip()]
        return ["bp", "blood pressure"]

    # --------------------------------------------------------------------------
    # Milestone 2: PDF & OCR Configuration
    # --------------------------------------------------------------------------
    max_pdf_bytes: int = Field(
        default=25_000_000,
        ge=1,
        description="Maximum allowed PDF file size in bytes (default 25 MB).",
    )
    max_pdf_pages: int = Field(
        default=50,
        ge=1,
        description="Maximum allowed page count per PDF document.",
    )
    max_render_pixels: int = Field(
        default=20_000_000,
        ge=1000,
        description="Maximum pixels per rendered pixmap (default 20 megapixels).",
    )
    ocr_mode: Literal["auto", "never", "always"] = Field(
        default="auto",
        description="OCR mode policy ('auto', 'never', 'always').",
    )
    ocr_languages: str = Field(
        default="eng",
        description="Tesseract OCR language string (e.g. 'eng' or 'eng+hin').",
    )
    ocr_dpi: int = Field(
        default=200,
        ge=72,
        le=600,
        description="Target rendering DPI for OCR.",
    )
    ocr_min_dpi: int = Field(
        default=100,
        ge=72,
        le=300,
        description="Minimum stepwise fallback DPI when render budget is exceeded.",
    )
    ocr_timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Per-call timeout in seconds for Tesseract OCR execution.",
    )
    max_ocr_regions_per_request: int = Field(
        default=60,
        ge=1,
        le=200,
        description="Maximum number of OCR region calls permitted per request.",
    )
    pdf_min_native_alnum: int = Field(
        default=40,
        ge=1,
        description="Minimum native alphanumeric characters required to consider page/region text-sufficient.",
    )
    pdf_image_region_min_fraction: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Minimum page area fraction for an embedded image to trigger selective OCR in auto mode.",
    )
    pdf_fullpage_raster_fraction: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Page area fraction of raster graphics triggering full-page OCR in auto mode.",
    )
    dedup_iou_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Bounding box Intersection-over-Union threshold for native/OCR duplicate suppression.",
    )
    tesseract_cmd: str | None = Field(
        default=None,
        description="Optional absolute path to Tesseract executable (overrides system PATH).",
    )

    # --------------------------------------------------------------------------
    # Milestone 3: Image Agent Configuration
    # --------------------------------------------------------------------------
    max_image_bytes: int = Field(
        default=15_000_000,
        ge=1,
        description="Maximum allowed image file size in bytes (default 15 MB).",
    )
    max_image_pixels: int = Field(
        default=30_000_000,
        ge=1000,
        description="Maximum allowed raw uncompressed image pixels (default 30 megapixels).",
    )
    image_upscale_min_side: int = Field(
        default=1000,
        ge=0,
        description="Upscale at most 2x when the shorter side is smaller than this; 0 disables upscaling.",
    )
    ocr_low_confidence: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Threshold below which OCR word confidence triggers a warning.",
    )

    # --------------------------------------------------------------------------
    # Milestone 4: Voice Agent Configuration
    # --------------------------------------------------------------------------
    allow_remote_audio_transcription: bool = Field(
        default=False,
        description="Independent safety kill-switch for remote audio transcription.",
    )
    groq_stt_model: str = Field(
        default="whisper-large-v3-turbo",
        description="Target Groq speech-to-text model (candidate: whisper-large-v3-turbo).",
    )
    max_audio_bytes: int = Field(
        default=20_000_000,
        ge=1,
        description="Maximum allowed audio file size in bytes (default 20 MB).",
    )
    max_audio_duration_seconds: float = Field(
        default=600.0,
        ge=1.0,
        description="Maximum allowed audio duration in seconds (default 10 minutes).",
    )
    ffprobe_path: str | None = Field(
        default=None,
        description="Optional absolute path to ffprobe executable (overrides system PATH).",
    )
    ffprobe_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Execution timeout in seconds for ffprobe subprocess inspection.",
    )


def get_settings() -> Settings:
    """Returns a default or cached Settings instance."""
    return Settings()
