"""Clinical field extractor abstraction.

Defines the interface between the workflow engine and extraction providers
(Groq or offline test doubles), enforcing the Dependency Inversion Principle.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from healthcare_agents.contracts import ExtractionResponse
from healthcare_agents.source_text import SourceSegment


class ExtractorCallMetadata(BaseModel):
    """Metadata regarding a single provider invocation attempt."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model_id: str
    response_id: str | None = None
    prompt_version: str = "clinical-extraction-v1"
    prompt_hash: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts_made: int = 1
    duration_ms: float = 0.0


class ClinicalFieldExtractor(ABC):
    """Abstract provider interface for clinical fact extraction."""

    @abstractmethod
    def extract(
        self,
        segments: list[SourceSegment],
        language_hint: str | None = None,
    ) -> tuple[ExtractionResponse, ExtractorCallMetadata]:
        """Extract structured clinical fact candidates from a list of segments.

        Args:
            segments: Ordered list of source segments for this chunk.
            language_hint: Optional ISO language code hint.

        Returns:
            Tuple of (ExtractionResponse, ExtractorCallMetadata)
        """
        pass
