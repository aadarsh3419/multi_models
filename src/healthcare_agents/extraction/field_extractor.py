"""
ClinicalFieldExtractor abstract interface and extractor errors.

Owned by the extraction package so ExtractionWorkflow depends on an interface it owns.
Zero provider SDK imports.
"""

from abc import ABC, abstractmethod
from typing import Any

from healthcare_agents.contracts.facts import ProviderExtractionResponse
from healthcare_agents.contracts.source_text import ProviderUnit


class ExtractorUnavailable(Exception):
    """Raised when an extractor cannot execute (e.g. remote LLM disabled or missing API key)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ClinicalFieldExtractor(ABC):
    """
    Abstract interface for extracting clinical fact candidates from a single ProviderUnit.
    Implementations (such as GroqFieldExtractor or test fakes) implement this protocol.
    """

    @abstractmethod
    def extract(
        self,
        unit: ProviderUnit,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        """
        Dispatches a ProviderUnit to the underlying model/service.
        Returns:
            ProviderExtractionResponse: Validated provider-schema candidates.
            dict[str, Any]: Audit metadata (model_id, prompt_sha256, request_id, token_usage).
        """
        pass
