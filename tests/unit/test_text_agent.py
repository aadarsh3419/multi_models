"""
Unit tests for TextAgent: end-to-end offline pipeline, limit enforcement, and privacy.
"""

import logging
from decimal import Decimal

from healthcare_agents.agents.text_agent import TextAgent
from healthcare_agents.contracts.facts import (
    AssertionStatus,
    ExperiencerType,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    Temporality,
)
from healthcare_agents.contracts.locations import TextLocation
from healthcare_agents.contracts.requests import TextAgentRequest
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.extraction.workflow import ExtractionWorkflow


class FakeClinicalExtractor(ClinicalFieldExtractor):
    def extract(self, unit: ProviderUnit, language_hint: str | None = None):
        # Emits a valid candidate if Heart rate is mentioned
        candidates = []
        if "Heart rate 72 bpm" in unit.text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Heart rate",
                    value_raw="72",
                    unit_raw="bpm",
                    reference_range_raw="",
                    abnormal_flag_raw="",
                    date_raw="",
                    time_raw="",
                    qualifier_raw="",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    experiencer_relationship_raw="",
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(segment_id=unit.unit_id, quote="Heart rate 72 bpm")
                    ],
                )
            )

        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "fake_extractor",
            "model_id": "test_model",
            "schema_enforcement": "strict",
            "prompt_sha256": "dummy_sha",
            "attempt_count": 1,
            "token_usage": {"total_tokens": 50},
        }


def test_text_agent_end_to_end_coordinate_preservation():
    """Verify that source offsets and evidence locations survive paragraph chunking end-to-end."""
    raw_text = (
        "Admission Note:\nPatient arrived in clinic at 10 AM.\n\n"
        "Vitals Exam:\nHeart rate 72 bpm observed.\n\n"
        "Plan:\nContinue observation."
    )
    workflow = ExtractionWorkflow(extractor=FakeClinicalExtractor())
    agent = TextAgent(workflow=workflow)

    req = TextAgentRequest(text=raw_text)
    result = agent.process(req)

    assert result.status == "success"
    assert len(result.candidates) == 1

    cand = result.candidates[0]
    assert cand.label_raw == "Heart rate"
    assert cand.value_parse_result.numeric_value == Decimal("72")

    # Verify evidence record has physical location surviving chunking
    assert len(cand.evidence_records) == 1
    ev_loc = cand.evidence_records[0].location
    assert isinstance(ev_loc, TextLocation)

    # Invariant: raw_text[start:end] == quote
    assert raw_text[ev_loc.start : ev_loc.end] == "Heart rate 72 bpm"


def test_text_agent_max_chars_limit_returns_failed_without_raising():
    """Verify that input exceeding max_text_chars returns a failed AgentResult rather than raising."""
    workflow = ExtractionWorkflow(extractor=FakeClinicalExtractor())
    agent = TextAgent(workflow=workflow, max_text_chars=50)

    long_text = (
        "This text is definitely much longer than fifty characters and must fail gracefully."
    )
    req = TextAgentRequest(text=long_text)
    result = agent.process(req)

    assert result.status == "failed"
    assert len(result.candidates) == 0
    assert any(e.code == "invalid_input" for e in result.errors)


def test_text_agent_privacy_no_text_in_logs(caplog):
    """Verify that routine logging does not leak raw text or evidence quotes."""
    secret_marker = "SUPER_SECRET_PATIENT_IDENTIFIER_XYZ_999"
    workflow = ExtractionWorkflow(extractor=FakeClinicalExtractor())
    agent = TextAgent(workflow=workflow)

    with caplog.at_level(logging.DEBUG):
        req = TextAgentRequest(text=f"Clinical note: {secret_marker} has mild cough.")
        agent.process(req)

    # Marker string must not appear anywhere in captured log text
    assert secret_marker not in caplog.text
