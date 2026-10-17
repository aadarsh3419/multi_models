"""
Extraction package owning the workflow, validator, parsers, deduplicator, and extractor protocol.

Zero direct provider SDK imports.
"""

from healthcare_agents.extraction.deduplication import CandidateDeduplicator
from healthcare_agents.extraction.evidence_validator import EvidenceValidator
from healthcare_agents.extraction.field_extractor import (
    ClinicalFieldExtractor,
    ExtractorUnavailable,
)
from healthcare_agents.extraction.value_parser import DateParser, ValueParser
from healthcare_agents.extraction.workflow import ExtractionWorkflow

__all__ = [
    "CandidateDeduplicator",
    "ClinicalFieldExtractor",
    "DateParser",
    "EvidenceValidator",
    "ExtractionWorkflow",
    "ExtractorUnavailable",
    "ValueParser",
]
