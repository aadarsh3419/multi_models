"""Text Agent: Entrypoint for clinical text extraction requests.

Encapsulates:
- Request size and non-empty validation gating.
- Lossless SourceTextBundle construction.
- Invocation of the shared ExtractionWorkflow.
- PHI-safe logging boundaries.
"""

import logging
from datetime import UTC

from healthcare_agents.contracts import (
    AgentResult,
    AgentStatus,
    StructuredError,
    TextAgentRequest,
)
from healthcare_agents.source_text import create_source_text_bundle
from healthcare_agents.workflow import ExtractionWorkflow

logger = logging.getLogger("healthcare_agents.text_agent")


class TextAgent:
    """Agent responsible for processing clinical text inputs into structured AgentResult."""

    def __init__(self, workflow: ExtractionWorkflow) -> None:
        """Initialize TextAgent with an injected ExtractionWorkflow.

        Architecture rule: TextAgent does not instantiate Groq clients directly.
        """
        self.workflow = workflow

    def process(self, request: TextAgentRequest) -> AgentResult:
        """Process a validated TextAgentRequest and return structured AgentResult."""
        logger.info(
            "Processing request_id=%s, length=%d chars",
            request.request_id,
            len(request.text),
        )

        try:
            # Step 1: Build lossless SourceTextBundle from untouched raw text
            bundle = create_source_text_bundle(
                raw_text=request.text,
                request_id=request.request_id,
                metadata={"language_hint": request.language_hint},
            )

            # Step 2: Execute extraction workflow
            result = self.workflow.process(bundle=bundle, options=request.options)
            return result

        except Exception as err:
            logger.error(
                "Unhandled error in TextAgent processing request_id=%s: %s",
                request.request_id,
                type(err).__name__,
            )
            # Create a safe failed AgentResult
            from datetime import datetime
            from hashlib import sha256

            from healthcare_agents.contracts import AuditMetadata

            source_hash = sha256(request.text.encode("utf-8")).hexdigest()
            return AgentResult(
                schema_version="1.0.0",
                request_id=request.request_id,
                agent_name="text",
                status=AgentStatus.failed,
                source_metadata={"language_hint": request.language_hint},
                candidates=[],
                stage_outcomes={},
                warnings=["Pipeline terminated abnormally."],
                errors=[
                    StructuredError(
                        code="INTERNAL_PROCESSING_ERROR",
                        message=f"Processing failed: {type(err).__name__}",
                        stage="text_agent",
                        retryable=False,
                    )
                ],
                audit_metadata=AuditMetadata(
                    request_id=str(request.request_id),
                    source_content_hash=source_hash,
                ),
                created_at=datetime.now(UTC),
                duration_ms=0.0,
            )
