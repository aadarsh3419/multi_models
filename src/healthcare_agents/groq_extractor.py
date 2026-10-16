"""Groq Cloud structured clinical field extractor adapter.

Implements ClinicalFieldExtractor using the official Groq Python SDK.
Configures:
- Strict JSON Schema structured outputs.
- SDK initialized with max_retries=0 to avoid nested retry multiplication.
- Application-level retries with Retry-After rate limit handling.
- Graceful handling of network timeouts, invalid credentials, and remote safety gates.
"""

import json
import logging
import time
from typing import Any

from groq import APIConnectionError, AuthenticationError, BadRequestError, Groq, RateLimitError

from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts import (
    ExtractionResponse,
)
from healthcare_agents.extractor_interface import (
    ClinicalFieldExtractor,
    ExtractorCallMetadata,
)
from healthcare_agents.prompt_templates import (
    EXTRACTION_JSON_SCHEMA,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    format_user_prompt,
    get_prompt_template_hash,
)
from healthcare_agents.source_text import SourceSegment

logger = logging.getLogger("healthcare_agents.groq_extractor")


class GroqFieldExtractor(ClinicalFieldExtractor):
    """Adapter invoking Groq models with strict structured JSON output."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: Groq | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Remote processing safety gate check
        if not self.settings.allow_remote_llm_processing and client is None:
            logger.info("Remote LLM processing disabled (ALLOW_REMOTE_LLM_PROCESSING=false).")

        self.model_id = self.settings.groq_text_model
        self.timeout = self.settings.provider_timeout_seconds
        self.max_retries = self.settings.provider_max_retries
        self.max_completion_tokens = self.settings.max_completion_tokens
        self.response_mode = self.settings.groq_response_mode

        # Initialize official Groq client with max_retries=0 to control retries in application layer
        if client is not None:
            self.client = client
        elif self.settings.allow_remote_llm_processing and self.settings.groq_api_key:
            self.client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        else:
            self.client = None

    def extract(
        self,
        segments: list[SourceSegment],
        language_hint: str | None = None,
    ) -> tuple[ExtractionResponse, ExtractorCallMetadata]:
        """Extract structured clinical fact candidates from segments using Groq."""
        if self.client is None:
            raise RuntimeError(
                "Groq client unavailable: remote processing is disabled or GROQ_API_KEY is missing."
            )

        # Prepare segment dictionary for prompt
        seg_payloads = [{"segment_id": s.segment_id, "exact_text": s.exact_text} for s in segments]
        user_message = format_user_prompt(seg_payloads, language_hint=language_hint)

        # Configure response_format according to GROQ_RESPONSE_MODE
        if self.response_mode == "strict_json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "clinical_extraction_response",
                    "strict": True,
                    "schema": EXTRACTION_JSON_SCHEMA,
                },
            }
        else:
            # json_object mode
            response_format = {"type": "json_object"}

        attempts_made = 0
        last_exception: Exception | None = None
        call_start_time = time.perf_counter()

        for attempt in range(1, self.max_retries + 2):
            attempts_made += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=response_format,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=0.0,
                )

                elapsed_ms = (time.perf_counter() - call_start_time) * 1000

                # Extract response text
                choice = response.choices[0]
                content_str = choice.message.content or ""
                decoded_json = json.loads(content_str)

                # Validate and parse into ExtractionResponse
                parsed_response = ExtractionResponse.model_validate(decoded_json)

                # Extract token usage and response ID
                in_tokens = response.usage.prompt_tokens if response.usage else None
                out_tokens = response.usage.completion_tokens if response.usage else None

                metadata = ExtractorCallMetadata(
                    provider_name="groq",
                    model_id=self.model_id,
                    response_id=response.id,
                    prompt_version=PROMPT_VERSION,
                    prompt_hash=get_prompt_template_hash(),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    attempts_made=attempts_made,
                    duration_ms=elapsed_ms,
                )

                return parsed_response, metadata

            except AuthenticationError as auth_err:
                # Fatal: do not retry invalid credentials
                logger.error("Groq authentication failed: %s", type(auth_err).__name__)
                raise

            except BadRequestError as bad_req:
                # Fatal: schema or model syntax error
                logger.error("Groq bad request error: %s", type(bad_req).__name__)
                raise

            except RateLimitError as rl_err:
                last_exception = rl_err
                # Check for Retry-After header
                wait_seconds = 2.0
                if hasattr(rl_err, "response") and rl_err.response is not None:
                    retry_header = rl_err.response.headers.get("retry-after")
                    if retry_header and retry_header.isdigit():
                        wait_seconds = max(1.0, float(retry_header))
                logger.warning(
                    "Rate limit hit on attempt %d/%d. Waiting %.1f seconds...",
                    attempt,
                    self.max_retries + 1,
                    wait_seconds,
                )
                if attempt <= self.max_retries:
                    time.sleep(wait_seconds)

            except (APIConnectionError, TimeoutError) as net_err:
                last_exception = net_err
                logger.warning(
                    "Network/timeout error on attempt %d/%d: %s",
                    attempt,
                    self.max_retries + 1,
                    type(net_err).__name__,
                )
                if attempt <= self.max_retries:
                    time.sleep(1.0 * attempt)

            except Exception as other_err:
                last_exception = other_err
                logger.warning(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt,
                    self.max_retries + 1,
                    type(other_err).__name__,
                )
                if attempt <= self.max_retries:
                    time.sleep(1.0 * attempt)

        # Retries exhausted
        raise RuntimeError(
            f"Groq extraction failed after {attempts_made} attempts: {type(last_exception).__name__}"
        ) from last_exception
