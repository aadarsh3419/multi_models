"""
Standardized AgentResult, StageResult, and AuditMetadata contracts.

Zero provider or agent dependencies.
"""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from healthcare_agents.contracts.errors import StructuredError
from healthcare_agents.contracts.facts import FactCandidate


class StageResult(BaseModel):
    """Execution status and count metrics for a single workflow stage."""

    stage_name: str
    status: Literal["success", "partial", "failed", "skipped"]
    counts: dict[str, int] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def stage(self) -> str:
        return self.stage_name


class AuditMetadata(BaseModel):
    """
    Cryptographic and runtime audit trail.
    Strictly forbids raw prompt or raw text capture to prevent PHI exposure.
    """

    model_config = ConfigDict(extra="ignore")

    provider: str = Field(
        default="unavailable", description="Active extraction provider ('groq' or 'unavailable')."
    )
    model_id: str = Field(
        default="unavailable", description="Provider model identifier, or 'unavailable')."
    )
    schema_enforcement: Literal["strict", "json_object", "none"] = "none"
    prompt_version: str = "1.0.0"
    prompt_sha256: str = "unavailable"
    schema_version: str = "1.0.0"
    model_call_count: int = 0
    provider_attempt_count: int = 0
    provider_request_ids: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    date_order_rule: str = "UNKNOWN"
    python_version: str = ""
    installed_versions: dict[str, str] = Field(default_factory=dict)
    key_settings: dict[str, Any] = Field(default_factory=dict)
    ocr_settings: dict[str, Any] = Field(default_factory=dict)
    table_api_available: bool = False
    page_plans: list[dict[str, Any]] = Field(default_factory=list)
    image_details: dict[str, Any] = Field(default_factory=dict)
    audio_details: dict[str, Any] = Field(default_factory=dict)
    stt_model_id: str | None = None
    ffprobe_version: str | None = None


class AgentResult(BaseModel):
    """
    Standardized multi-agent clinical result contract.
    Shared across Text Agent (Milestone 1) and future multimodal agents (PDF, Image, Voice).
    """

    schema_version: str = "1.0.0"
    request_id: UUID
    agent_name: str
    source_type: str = "text"
    status: Literal["success", "partial", "failed"]
    sanitized_source_metadata: dict[str, Any]
    content_sha256: str
    candidates: list[FactCandidate] = Field(default_factory=list)
    stage_results: dict[str, StageResult] = Field(default_factory=dict)
    warnings: list[StructuredError] = Field(default_factory=list)
    errors: list[StructuredError] = Field(default_factory=list)
    audit: AuditMetadata
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    contains_source_text: bool = False
    debug_source_text: str | None = None

    @property
    def stage_outcomes(self) -> dict[str, StageResult]:
        return self.stage_results
