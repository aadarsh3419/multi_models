"""
Integration and mock tests for GroqClientWrapper and GroqFieldExtractor.

Tests retry matrix, request shape, error handling, audit metadata, and schema synchronization.
Offline only: never calls live network.
"""

import json
from unittest.mock import MagicMock

import pytest
from groq import AuthenticationError, RateLimitError

from healthcare_agents.contracts.facts import (
    PROVIDER_JSON_SCHEMA,
    FactType,
    ProviderExtractionResponse,
)
from healthcare_agents.contracts.source_text import ProviderUnit
from healthcare_agents.extraction.field_extractor import ExtractorUnavailable
from healthcare_agents.providers.groq_client import GroqClientWrapper
from healthcare_agents.providers.groq_field_extractor import (
    EXTRACTION_PROMPT_SHA256,
    EXTRACTION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    GroqFieldExtractor,
)


def make_unit() -> ProviderUnit:
    return ProviderUnit(
        unit_id="s0001",
        text="Heart rate is 72 bpm.",
        parent_segment_id="s0001",
        parent_start=0,
        parent_end=21,
    )


def test_schema_sync_pydantic_and_handwritten():
    """Verify that ProviderExtractionResponse schema agrees with PROVIDER_JSON_SCHEMA."""
    pydantic_schema = ProviderExtractionResponse.model_json_schema()
    assert "candidates" in pydantic_schema["properties"]
    assert "candidates" in PROVIDER_JSON_SCHEMA["properties"]

    cand_props = PROVIDER_JSON_SCHEMA["properties"]["candidates"]["items"]["properties"]
    required_fields = set(PROVIDER_JSON_SCHEMA["properties"]["candidates"]["items"]["required"])

    # Every property defined in schema must be listed in required (Groq strict requirement)
    assert set(cand_props.keys()) == required_fields
    assert (
        PROVIDER_JSON_SCHEMA["properties"]["candidates"]["items"]["additionalProperties"] is False
    )


def test_groq_extractor_request_shape():
    """Verify request shape: source narrative only in user JSON, system prompt unchanged."""
    mock_sdk_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "candidates": [
                {
                    "fact_type": "measurement",
                    "label_raw": "Heart rate",
                    "value_raw": "72",
                    "unit_raw": "bpm",
                    "reference_range_raw": "",
                    "abnormal_flag_raw": "",
                    "date_raw": "",
                    "time_raw": "",
                    "qualifier_raw": "",
                    "assertion_status": "affirmed",
                    "temporality": "not_applicable",
                    "experiencer_type": "patient",
                    "experiencer_relationship_raw": "",
                    "reporting_source_role": "clinician",
                    "evidence_refs": [{"segment_id": "s0001", "quote": "Heart rate is 72 bpm"}],
                }
            ]
        }
    )
    mock_completion = MagicMock()
    mock_completion.id = "req_12345"
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 50
    mock_completion.usage.completion_tokens = 25
    mock_completion.usage.total_tokens = 75
    mock_sdk_client.chat.completions.create.return_value = mock_completion

    wrapper = GroqClientWrapper(api_key="gsk_test", client=mock_sdk_client)
    extractor = GroqFieldExtractor(client=wrapper, model="openai/gpt-oss-20b", schema_mode="strict")

    resp, meta = extractor.extract(make_unit(), language_hint="en")

    assert len(resp.candidates) == 1
    assert resp.candidates[0].fact_type == FactType.MEASUREMENT
    assert meta["model_id"] == "openai/gpt-oss-20b"
    assert meta["prompt_version"] == EXTRACTION_PROMPT_VERSION
    assert meta["prompt_sha256"] == EXTRACTION_PROMPT_SHA256
    assert meta["request_id"] == "req_12345"
    assert meta["token_usage"]["total_tokens"] == 75

    # Verify parameters sent to chat.completions.create
    create_call = mock_sdk_client.chat.completions.create.call_args[1]
    assert create_call["temperature"] == 0.0
    messages = create_call["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Heart rate is 72 bpm." not in messages[0]["content"]  # Never in system message!

    assert messages[1]["role"] == "user"
    user_payload = json.loads(messages[1]["content"])
    assert user_payload["units"][0]["text"] == "Heart rate is 72 bpm."


def test_groq_client_retry_rate_limit_with_retry_after():
    """Verify 429 RateLimitError retries up to max_retries and sleeps."""
    mock_sdk = MagicMock()
    mock_response = MagicMock()
    mock_response.headers = {"retry-after": "2.0"}

    # Fail on first call with 429, succeed on second call
    mock_choice = MagicMock()
    mock_choice.message.content = '{"candidates": []}'
    mock_success = MagicMock()
    mock_success.choices = [mock_choice]
    mock_success.id = "req_success"
    mock_success.usage = None

    mock_sdk.chat.completions.create.side_effect = [
        RateLimitError(message="Rate limit hit", response=mock_response, body={}),
        mock_success,
    ]

    sleeps: list[float] = []

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    wrapper = GroqClientWrapper(
        api_key="gsk_test", client=mock_sdk, max_retries=2, sleep_fn=mock_sleep
    )
    content, meta = wrapper.create_chat_completion([], "test_model", {})

    assert content == '{"candidates": []}'
    assert meta["attempt_count"] == 2
    assert sleeps == [2.0]  # Respected Retry-After header


def test_groq_client_fatal_errors_never_retried():
    """Verify 401 AuthenticationError and 400 BadRequestError are never retried."""
    mock_sdk = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_sdk.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=mock_response,
        body={},
    )

    wrapper = GroqClientWrapper(api_key="gsk_test", client=mock_sdk, max_retries=3)

    with pytest.raises(ExtractorUnavailable) as exc_info:
        wrapper.create_chat_completion([], "test_model", {})

    assert "AuthenticationError" in exc_info.value.message
    # Proves it was called exactly once and aborted immediately without retrying
    assert mock_sdk.chat.completions.create.call_count == 1


def test_groq_client_rate_limit_exceeded_fails():
    """Verify persistent 429 past max_retries raises ExtractorUnavailable."""
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.side_effect = RateLimitError(
        message="Rate limit",
        response=MagicMock(),
        body={},
    )

    wrapper = GroqClientWrapper(
        api_key="gsk_test", client=mock_sdk, max_retries=2, sleep_fn=lambda s: None
    )

    with pytest.raises(ExtractorUnavailable) as exc_info:
        wrapper.create_chat_completion([], "test_model", {})

    assert exc_info.value.code == "provider_rate_limited"
    assert mock_sdk.chat.completions.create.call_count == 3  # Initial + 2 retries
