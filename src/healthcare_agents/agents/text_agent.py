"""
TextAgent: Concrete modality agent for unstructured and semi-structured text narratives.

Enforces input size limits, orchestrates paragraph reading into SourceTextBundle,
and delegates extraction to the injected ExtractionWorkflow.
Zero direct provider SDK imports.
"""

import hashlib
import logging
import platform
from datetime import UTC, datetime

from healthcare_agents.agents.base import BaseAgent
from healthcare_agents.contracts.errors import ErrorCode, StructuredError
from healthcare_agents.contracts.requests import TextAgentRequest
from healthcare_agents.contracts.results import AgentResult, AuditMetadata, StageResult
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.sources.text_reader import TextReader

logger = logging.getLogger(__name__)


class TextAgent(BaseAgent):
    """
    Ingests raw clinical narratives and produces verified FactCandidate records
    packaged in the unified AgentResult contract.
    """

    def __init__(
        self,
        workflow: ExtractionWorkflow,
        max_text_chars: int = 100_000,
    ):
        self.workflow = workflow
        self.max_text_chars = max_text_chars

    def process(self, request: TextAgentRequest) -> AgentResult:
        """
        Validates text size limits and executes extraction workflow.
        Returns failed AgentResult without raising if max_text_chars is exceeded.
        """
        # Logging: strictly PHI-free
        logger.info(
            "TextAgent: Processing request_id=%s, length=%d chars",
            request.request_id,
            len(request.text),
        )

        content_bytes = request.text.encode("utf-8")
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()

        # Enforce maximum text length limit
        if len(request.text) > self.max_text_chars:
            logger.warning(
                "TextAgent: Request %s exceeded limit (length=%d, max=%d)",
                request.request_id,
                len(request.text),
                self.max_text_chars,
            )
            audit = AuditMetadata(
                provider="unavailable",
                model_id="unavailable",
                schema_enforcement="none",
                prompt_version="1.0.0",
                prompt_sha256="unavailable",
                schema_version="1.0.0",
                python_version=platform.python_version(),
            )
            return AgentResult(
                schema_version="1.0.0",
                request_id=request.request_id,
                agent_name="text_agent",
                source_type="text",
                status="failed",
                sanitized_source_metadata={
                    "char_count": len(request.text),
                    "byte_count": len(content_bytes),
                    "content_sha256": content_sha256,
                },
                content_sha256=content_sha256,
                candidates=[],
                stage_results={
                    "source_reading": StageResult(
                        stage_name="source_reading",
                        status="failed",
                        details={"error": "Text size exceeds configured maximum."},
                    )
                },
                warnings=[],
                errors=[
                    StructuredError(
                        code=ErrorCode.INVALID_INPUT.value,
                        stage="source_reading",
                        message=f"Input text length ({len(request.text)}) exceeds configured maximum ({self.max_text_chars}).",
                        details={"length": len(request.text), "max_allowed": self.max_text_chars},
                    )
                ],
                audit=audit,
                created_at_utc=datetime.now(UTC),
                duration_ms=0.0,
            )

        # Build SourceTextBundle
        bundle = TextReader.read_bundle(request.text, request.request_id)

        # Delegate execution to injected workflow
        result = self.workflow.process_bundle(
            bundle=bundle,
            language_hint=request.language_hint,
            date_order_hint=request.date_order_hint,
            options=request.options,
            agent_name="text_agent",
            original_text=request.text,
        )

        logger.info(
            "TextAgent: Completed request_id=%s, status=%s, candidates=%d, duration=%.2fms",
            result.request_id,
            result.status,
            len(result.candidates),
            result.duration_ms,
        )

        return result
