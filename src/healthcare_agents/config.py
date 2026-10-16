"""Application configuration and default operational limits.

Uses pydantic-settings to manage environment variables, safety gates,
and configurable bounds for the healthcare extraction pipeline.
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DateOrderType = Literal["UNKNOWN", "DMY", "MDY"]
GroqResponseModeType = Literal["strict_json_schema", "json_object"]


class Settings(BaseSettings):
    """Runtime configuration settings for the text extraction foundation.

    All limits are configurable prototypes and enforce safety boundaries
    to prevent runaway LLM calls and memory exhaustion.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Groq API Configuration
    groq_api_key: str | None = Field(
        default=None,
        validation_alias="GROQ_API_KEY",
        description="API key for Groq Cloud. Required only if remote LLM processing is enabled.",
    )
    groq_text_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias="GROQ_TEXT_MODEL",
        description="Target model identifier on Groq.",
    )
    groq_response_mode: GroqResponseModeType = Field(
        default="strict_json_schema",
        validation_alias="GROQ_RESPONSE_MODE",
        description="Response formatting mode: strict_json_schema or json_object.",
    )

    # Remote Processing Safety Gate
    allow_remote_llm_processing: bool = Field(
        default=False,
        validation_alias="ALLOW_REMOTE_LLM_PROCESSING",
        description="Safety gate. If False, remote Groq API calls are rejected and skipped.",
    )

    # Date parsing order rule
    date_order: DateOrderType = Field(
        default="UNKNOWN",
        validation_alias="DATE_ORDER",
        description="Default interpretation order for ambiguous dates (UNKNOWN, DMY, or MDY).",
    )

    # Chunking & Call Budget Limits
    max_text_chars: int = Field(
        default=36000,
        validation_alias="MAX_TEXT_CHARS",
        description="Maximum allowed input characters (Unicode code points) per request.",
    )
    max_llm_chars_per_chunk: int = Field(
        default=6000,
        validation_alias="MAX_LLM_CHARS_PER_CHUNK",
        description="Maximum characters submitted in a single LLM chunk.",
    )
    max_llm_calls_per_request: int = Field(
        default=8,
        validation_alias="MAX_LLM_CALLS_PER_REQUEST",
        description="Total allowed provider attempts including retries across all chunks.",
    )
    max_completion_tokens: int = Field(
        default=2500,
        validation_alias="MAX_COMPLETION_TOKENS",
        description="Hard token generation limit per provider invocation.",
    )

    # Network Resilience & Concurrency
    provider_timeout_seconds: float = Field(
        default=30.0,
        validation_alias="PROVIDER_TIMEOUT_SECONDS",
        description="Socket and read timeout for provider calls.",
    )
    provider_max_retries: int = Field(
        default=2,
        validation_alias="PROVIDER_MAX_RETRIES",
        description="Maximum application-level retries for transient provider failures.",
    )
    provider_max_concurrency_per_request: int = Field(
        default=1,
        validation_alias="PROVIDER_MAX_CONCURRENCY_PER_REQUEST",
        description="Max concurrent LLM calls per request (kept at 1 for sequential chunking).",
    )

    @field_validator("date_order", mode="before")
    @classmethod
    def normalize_date_order(cls, v: str) -> str:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper in {"UNKNOWN", "DMY", "MDY"}:
                return v_upper
        raise ValueError("DATE_ORDER must be one of: 'UNKNOWN', 'DMY', 'MDY'")


# Default settings instance for convenient singleton import
def get_settings() -> Settings:
    """Return a cached or newly initialized Settings instance."""
    return Settings()
