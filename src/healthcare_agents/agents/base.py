"""
BaseAgent abstract interface.

Defines the common execution contract for Text Agent and future multimodal agents.
Zero provider SDK imports.
"""

from abc import ABC, abstractmethod

from healthcare_agents.contracts.requests import TextAgentRequest
from healthcare_agents.contracts.results import AgentResult


class BaseAgent(ABC):
    """Abstract base class for all modality agents."""

    @abstractmethod
    def process(self, request: TextAgentRequest) -> AgentResult:
        """
        Executes ingestion, coordinate grounding, structured extraction,
        and post-processing for a single clinical request.
        """
        pass
