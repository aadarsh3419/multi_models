"""
Groq client wrapper with single-owner retry and backoff policy.

Creates official groq SDK client with client-internal retries disabled (max_retries=0).
Owns retries strictly for 429, 5xx, timeouts, and network connection errors.
400, 401, 403, 404, 422 are fatal and never retried.
"""

import re
import time
from collections.abc import Callable
from typing import Any

import groq
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from healthcare_agents.contracts.errors import ErrorCode
from healthcare_agents.extraction.field_extractor import ExtractorUnavailable


class GroqClientWrapper:
    """
    Manages communication with Groq Chat Completions API with explicit retry policies
    and injectable sleep mechanism for unit testing.
    """

    NON_RETRYABLE_ERRORS = (
        AuthenticationError,
        PermissionDeniedError,
        BadRequestError,
        NotFoundError,
        UnprocessableEntityError,
    )

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        max_backoff_seconds: float = 30.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        client: groq.Groq | None = None,
    ):
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds
        self.sleep = sleep_fn

        # SDK client created with its own internal retries explicitly disabled
        if client is not None:
            self._client = client
        else:
            self._client = groq.Groq(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,  # Single-owner retry requirement
            )

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        response_format: dict[str, Any],
        max_completion_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        """
        Executes chat completion with deterministic retry policy.
        Returns:
            tuple[content_str, call_metadata]
        """
        attempts = 0
        backoff = 1.0

        while True:
            attempts += 1
            try:
                completion = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                )

                content = completion.choices[0].message.content or "{}"
                call_meta = {
                    "request_id": getattr(completion, "id", None),
                    "attempt_count": attempts,
                    "token_usage": {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0)
                        if completion.usage
                        else 0,
                        "completion_tokens": (
                            getattr(completion.usage, "completion_tokens", 0)
                            if completion.usage
                            else 0
                        ),
                        "total_tokens": getattr(completion.usage, "total_tokens", 0)
                        if completion.usage
                        else 0,
                    },
                }
                return content, call_meta

            except self.NON_RETRYABLE_ERRORS as err:
                # Fatal authentication or bad request errors -> never retried
                code = (
                    ErrorCode.PROVIDER_AUTHENTICATION_FAILED.value
                    if isinstance(err, (AuthenticationError, PermissionDeniedError))
                    else ErrorCode.PROVIDER_RESPONSE_INVALID.value
                )
                raise ExtractorUnavailable(
                    code, f"Non-retryable Groq client error: {type(err).__name__}"
                ) from err

            except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as err:
                if attempts > self.max_retries:
                    code = (
                        ErrorCode.PROVIDER_RATE_LIMITED.value
                        if isinstance(err, RateLimitError)
                        else ErrorCode.PROVIDER_TIMEOUT.value
                    )
                    raise ExtractorUnavailable(
                        code, f"Exceeded max retries ({self.max_retries}): {err}"
                    ) from err

                # Determine sleep time (honor Retry-After if provided)
                wait_time = self._extract_retry_after(err) or backoff
                if wait_time > self.max_backoff_seconds:
                    raise ExtractorUnavailable(
                        ErrorCode.PROVIDER_RATE_LIMITED.value,
                        f"Retry delay ({wait_time}s) exceeds max allowed backoff ({self.max_backoff_seconds}s).",
                    ) from err

                self.sleep(wait_time)
                backoff = min(backoff * 2.0, self.max_backoff_seconds)

    def create_transcription(
        self,
        file: tuple[str, bytes],
        model: str = "whisper-large-v3-turbo",
        response_format: str = "verbose_json",
        timestamp_granularities: list[str] | None = None,
        language: str | None = None,
        temperature: float = 0.0,
    ) -> Any:
        """
        Executes Groq audio transcription with deterministic single-owner retry policy.
        Error mapping:
          - 413: FILE_TOO_LARGE
          - 401/403: PROVIDER_AUTHENTICATION_FAILED
          - 429: PROVIDER_RATE_LIMITED
          - 400/unknown: AUDIO_TRANSCRIPTION_FAILED
        """
        attempts = 0
        backoff = 1.0
        kwargs: dict[str, Any] = {
            "file": file,
            "model": model,
            "response_format": response_format,
            "temperature": temperature,
        }
        if timestamp_granularities:
            kwargs["timestamp_granularities"] = timestamp_granularities
        if language:
            kwargs["language"] = language

        while True:
            attempts += 1
            try:
                return self._client.audio.transcriptions.create(**kwargs)
            except self.NON_RETRYABLE_ERRORS as err:
                if isinstance(err, (AuthenticationError, PermissionDeniedError)):
                    code = ErrorCode.PROVIDER_AUTHENTICATION_FAILED.value
                else:
                    code = ErrorCode.AUDIO_TRANSCRIPTION_FAILED.value
                raise ExtractorUnavailable(
                    code, f"Non-retryable Groq transcription error: {type(err).__name__}: {err}"
                ) from err
            except (RateLimitError, APIStatusError, APITimeoutError, APIConnectionError) as err:
                status_code = getattr(err, "status_code", None)
                if status_code == 413:
                    raise ExtractorUnavailable(
                        ErrorCode.FILE_TOO_LARGE.value,
                        f"Audio file exceeds provider upload limit (413 Payload Too Large): {err}",
                    ) from err
                if attempts > self.max_retries:
                    code = (
                        ErrorCode.PROVIDER_RATE_LIMITED.value
                        if isinstance(err, RateLimitError)
                        else ErrorCode.PROVIDER_TIMEOUT.value
                    )
                    raise ExtractorUnavailable(
                        code, f"Exceeded max retries ({self.max_retries}) for transcription: {err}"
                    ) from err
                wait_time = self._extract_retry_after(err) or backoff
                if wait_time > self.max_backoff_seconds:
                    raise ExtractorUnavailable(
                        ErrorCode.PROVIDER_RATE_LIMITED.value,
                        f"Retry delay ({wait_time}s) exceeds max allowed backoff ({self.max_backoff_seconds}s).",
                    ) from err
                self.sleep(wait_time)
                backoff = min(backoff * 2.0, self.max_backoff_seconds)

    def _extract_retry_after(self, err: Exception) -> float | None:
        """Parses Retry-After seconds from HTTP response headers or error message if available."""
        if hasattr(err, "response") and err.response is not None:
            retry_header = err.response.headers.get("retry-after")
            if retry_header:
                try:
                    return float(retry_header)
                except ValueError:
                    pass

        # Check for message pattern (e.g. 'try again in 2.5s')
        msg = str(err)
        m = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        return None
