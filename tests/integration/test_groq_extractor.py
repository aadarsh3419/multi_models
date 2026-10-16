"""Integration tests for GroqFieldExtractor: Mocked adapter behavior and optional live API tests."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from groq import AuthenticationError, RateLimitError

from healthcare_agents.config import Settings
from healthcare_agents.contracts import ExtractionResponse
from healthcare_agents.groq_extractor import GroqFieldExtractor
from healthcare_agents.source_text import SourceSegment


def make_mock_segment() -> SourceSegment:
    return SourceSegment(
        segment_id="txt_000001",
        exact_text="Heart rate is 72 bpm.",
        start_offset=0,
        end_offset=21,
    )


def test_groq_extractor_mocked_success():
    """Verify GroqFieldExtractor parses valid mock response from Groq client."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "candidates": [
                {
                    "fact_type": "measurement",
                    "label_raw": "Heart rate",
                    "value_raw": "72",
                    "unit_raw": "bpm",
                    "reference_range_raw": None,
                    "abnormal_flag_raw": None,
                    "date_raw": None,
                    "time_raw": None,
                    "qualifier_raw": None,
                    "dose_raw": None,
                    "frequency_raw": None,
                    "route_raw": None,
                    "duration_raw": None,
                    "assertion_status": "affirmed",
                    "temporality": "current",
                    "experiencer_type": "patient",
                    "experiencer_relationship_raw": None,
                    "reporting_source_role": "clinician",
                    "evidence_refs": [
                        {"segment_id": "txt_000001", "quote": "Heart rate is 72 bpm"}
                    ],
                }
            ]
        }
    )
    mock_completion = MagicMock()
    mock_completion.id = "mock_groq_id_123"
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 45
    mock_completion.usage.completion_tokens = 30
    mock_client.chat.completions.create.return_value = mock_completion

    settings = Settings(
        allow_remote_llm_processing=True,
        groq_api_key="gsk_mockkey",
        provider_max_retries=1,
    )
    extractor = GroqFieldExtractor(settings=settings, client=mock_client)

    resp, meta = extractor.extract([make_mock_segment()])

    assert len(resp.candidates) == 1
    assert resp.candidates[0].label_raw == "Heart rate"
    assert meta.provider_name == "groq"
    assert meta.response_id == "mock_groq_id_123"
    assert meta.input_tokens == 45
    assert meta.output_tokens == 30
    assert meta.attempts_made == 1


def test_groq_extractor_fatal_auth_error_not_retried():
    """Verify AuthenticationError is not retried indefinitely."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=mock_response,
        body={"error": "Invalid API Key"},
    )

    settings = Settings(
        allow_remote_llm_processing=True,
        groq_api_key="gsk_invalid",
        provider_max_retries=2,
    )
    extractor = GroqFieldExtractor(settings=settings, client=mock_client)

    with pytest.raises(AuthenticationError):
        extractor.extract([make_mock_segment()])

    # Must fail on the very first attempt without retrying
    assert mock_client.chat.completions.create.call_count == 1


def test_groq_extractor_rate_limit_with_retry_after():
    """Verify RateLimitError respects Retry-After header and retries up to provider_max_retries."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "0"}  # 0s for fast test
    rl_error = RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": "Rate limit"},
    )

    # First call raises 429, second call succeeds
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"candidates": []})
    mock_completion = MagicMock()
    mock_completion.id = "mock_after_retry"
    mock_completion.choices = [mock_choice]
    mock_completion.usage = None

    mock_client.chat.completions.create.side_effect = [rl_error, mock_completion]

    settings = Settings(
        allow_remote_llm_processing=True,
        groq_api_key="gsk_mockkey",
        provider_max_retries=1,
    )
    extractor = GroqFieldExtractor(settings=settings, client=mock_client)

    with patch("time.sleep", return_value=None):
        resp, meta = extractor.extract([make_mock_segment()])

    assert mock_client.chat.completions.create.call_count == 2
    assert meta.attempts_made == 2


def test_groq_extractor_remote_processing_disabled_rejection():
    """Verify extractor rejects execution when remote processing safety gate is False."""
    settings = Settings(
        allow_remote_llm_processing=False,
        groq_api_key=None,
    )
    extractor = GroqFieldExtractor(settings=settings)

    with pytest.raises(RuntimeError) as exc_info:
        extractor.extract([make_mock_segment()])
    assert "remote processing is disabled" in str(exc_info.value)


# ==============================================================================
# Optional Live API Test
# ==============================================================================


@pytest.mark.live_api
def test_live_groq_api_structured_extraction():
    """Live API test invoking actual Groq endpoint.

    Requires:
    - Environment variable RUN_LIVE_API_TESTS=true
    - Valid GROQ_API_KEY
    - ALLOW_REMOTE_LLM_PROCESSING=true
    """
    run_live = os.environ.get("RUN_LIVE_API_TESTS", "").lower() == "true"
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not run_live or not api_key:
        pytest.skip(
            "Live API test skipped: set RUN_LIVE_API_TESTS=true and GROQ_API_KEY to execute."
        )

    settings = Settings(
        allow_remote_llm_processing=True,
        groq_api_key=api_key,
        groq_text_model="openai/gpt-oss-20b",
        provider_timeout_seconds=30.0,
    )
    extractor = GroqFieldExtractor(settings=settings)

    test_seg = SourceSegment(
        segment_id="txt_live_01",
        exact_text="Clinical encounter: Patient blood pressure 125/82 mmHg.",
        start_offset=0,
        end_offset=56,
    )

    resp, meta = extractor.extract([test_seg])

    assert isinstance(resp, ExtractionResponse)
    assert meta.provider_name == "groq"
    assert meta.model_id == "openai/gpt-oss-20b"
    assert len(resp.candidates) >= 1
    cand = resp.candidates[0]
    assert cand.fact_type == "measurement"
    assert "125/82" in (cand.value_raw or "")
