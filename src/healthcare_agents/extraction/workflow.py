"""
ExtractionWorkflow: Synchronous multi-stage orchestration for clinical fact extraction.

Coordinates chunking, provider dispatch, evidence verification, value parsing, and deduplication.
Zero direct provider SDK imports.
"""

import platform
import time
from datetime import UTC, datetime
from typing import Literal

from healthcare_agents.contracts.errors import ErrorCode, StructuredError
from healthcare_agents.contracts.facts import FactCandidate
from healthcare_agents.contracts.requests import (
    ImageAgentOptions,
    PDFAgentOptions,
    TextAgentOptions,
    VoiceAgentOptions,
)
from healthcare_agents.contracts.results import AgentResult, AuditMetadata, StageResult
from healthcare_agents.contracts.source_text import SourceTextBundle
from healthcare_agents.extraction.deduplication import CandidateDeduplicator
from healthcare_agents.extraction.evidence_validator import EvidenceValidator
from healthcare_agents.extraction.field_extractor import (
    ClinicalFieldExtractor,
    ExtractorUnavailable,
)
from healthcare_agents.extraction.value_parser import DateParser, ValueParser
from healthcare_agents.sources.chunker import TextChunker


class ExtractionWorkflow:
    """
    Coordinates clinical text extraction across sequential chunk calls with budget enforcement,
    safe failure isolation, and rigorous audit tracking.
    """

    def __init__(
        self,
        extractor: ClinicalFieldExtractor,
        validator: EvidenceValidator | None = None,
        value_parser: ValueParser | None = None,
        chunker: TextChunker | None = None,
        deduplicator: CandidateDeduplicator | None = None,
        date_order: str = "UNKNOWN",
        max_llm_calls_per_request: int = 20,
        max_consecutive_chunk_failures: int = 2,
    ):
        self.extractor = extractor
        self.validator = validator or EvidenceValidator()
        self.value_parser = value_parser or ValueParser()
        self.chunker = chunker or TextChunker()
        self.deduplicator = deduplicator or CandidateDeduplicator()
        self.date_order = date_order
        self.max_llm_calls = max_llm_calls_per_request
        self.max_consecutive_chunk_failures = max_consecutive_chunk_failures

    def process_bundle(
        self,
        bundle: SourceTextBundle,
        language_hint: str | None = None,
        date_order_hint: str | None = None,
        options: (
            TextAgentOptions | PDFAgentOptions | ImageAgentOptions | VoiceAgentOptions | None
        ) = None,
        agent_name: str = "text_agent",
        original_text: str | None = None,
    ) -> AgentResult:
        """
        Executes extraction pipeline across all segments in the SourceTextBundle.
        """
        start_time = time.perf_counter()
        effective_date_order = date_order_hint or self.date_order
        effective_options = options or TextAgentOptions()

        all_candidates: list[FactCandidate] = []
        all_warnings: list[StructuredError] = []
        all_errors: list[StructuredError] = []
        stage_results: dict[str, StageResult] = {}

        # ----------------------------------------------------------------------
        # Stage 1: Source Segmentation & Chunking
        # ----------------------------------------------------------------------
        units = self.chunker.chunk_bundle(bundle)
        stage_results["source_reading"] = StageResult(
            stage_name="source_reading",
            status="success",
            counts={"segments": len(bundle.ordered_segments), "units": len(units)},
        )

        # Build quick lookup of parent segments by segment_id
        segment_map = {seg.segment_id: seg for seg in bundle.ordered_segments}

        # Audit accumulators
        model_call_count = 0
        provider_attempt_count = 0
        provider_request_ids: list[str] = []
        token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        active_provider_name = "unavailable"
        active_model_id = "unavailable"
        schema_enforcement: Literal["strict", "json_object", "none"] = "none"
        prompt_version = "1.0.0"
        prompt_sha256 = "unavailable"

        consecutive_failures = 0
        unprocessed_unit_ids: list[str] = []

        # ----------------------------------------------------------------------
        # Stage 2: Sequential Provider Calls & Validation
        # ----------------------------------------------------------------------
        if self.extractor is None:
            stage_results["extraction"] = StageResult(
                stage_name="extraction",
                status="skipped",
                counts={"units": len(units)},
            )
            extractor_skip_err = StructuredError(
                code="no_active_extractor",
                stage="extraction",
                message="no active extractor available",
            )
            all_warnings.append(extractor_skip_err)
            all_errors.append(extractor_skip_err)
            unprocessed_unit_ids.extend([u.unit_id for u in units])
            units = []

        for idx, unit in enumerate(units):
            # Enforce sequential LLM call budget
            if model_call_count >= self.max_llm_calls:
                unprocessed = [u.unit_id for u in units[idx:]]
                unprocessed_unit_ids.extend(unprocessed)
                budget_warn = StructuredError(
                    code=ErrorCode.LLM_CALL_BUDGET_EXCEEDED.value,
                    stage="clinical_extraction",
                    message=f"Call budget exhausted ({self.max_llm_calls}); stopping further chunk processing.",
                    details={"unprocessed_unit_ids": unprocessed},
                )
                budget_err = StructuredError(
                    code="CALL_BUDGET_EXHAUSTED",
                    stage="clinical_extraction",
                    message=f"Call budget exhausted ({self.max_llm_calls}); stopping further chunk processing.",
                    details={"unprocessed_unit_ids": unprocessed},
                )
                all_warnings.append(budget_warn)
                all_warnings.append(budget_err)
                all_errors.append(budget_err)
                break

            # Enforce consecutive chunk failure limit
            if consecutive_failures >= self.max_consecutive_chunk_failures:
                unprocessed = [u.unit_id for u in units[idx:]]
                unprocessed_unit_ids.extend(unprocessed)
                all_warnings.append(
                    StructuredError(
                        code=ErrorCode.PROVIDER_RESPONSE_INVALID.value,
                        stage="clinical_extraction",
                        message=f"Consecutive chunk failure limit ({self.max_consecutive_chunk_failures}) reached.",
                        details={"unprocessed_unit_ids": unprocessed},
                    )
                )
                break

            parent_seg = segment_map.get(unit.parent_segment_id)
            if not parent_seg:
                # Fallback to first segment if mapping fails
                parent_seg = bundle.ordered_segments[0]

            try:
                provider_response, call_meta = self.extractor.extract(
                    unit=unit,
                    language_hint=language_hint,
                )
                model_call_count += 1
                consecutive_failures = 0

                # Accumulate audit metadata
                active_provider_name = call_meta.get("provider", "groq")
                active_model_id = call_meta.get("model_id", "unavailable")
                schema_enforcement = call_meta.get("schema_enforcement", "strict")
                prompt_sha256 = call_meta.get("prompt_sha256", "unavailable")
                provider_attempt_count += call_meta.get("attempt_count", 1)

                req_id_str = call_meta.get("request_id")
                if req_id_str:
                    provider_request_ids.append(req_id_str)

                unit_tokens = call_meta.get("token_usage", {})
                for k, v in unit_tokens.items():
                    token_usage[k] = token_usage.get(k, 0) + v

                # Evidence Grounding & Coordinate Verification
                provenance_data = {
                    "model_id": active_model_id,
                    "prompt_version": prompt_version,
                }
                candidates, val_warnings = self.validator.validate_candidates(
                    provider_candidates=provider_response.candidates,
                    unit=unit,
                    parent_segment=parent_seg,
                    content_sha256=bundle.safe_metadata.get("content_sha256", ""),
                    provenance=provenance_data,
                )
                all_warnings.extend(val_warnings)

                # Deterministic Value & Date Parsing
                for cand in candidates:
                    cand.value_parse_result = self.value_parser.parse_value(
                        cand.value_raw, cand.label_raw
                    )
                    cand.parse_status = cand.value_parse_result.parse_status

                    if cand.reference_range_raw:
                        range_res = self.value_parser.parse_reference_range(
                            cand.reference_range_raw
                        )
                        cand.parsed_reference_range_low = range_res.range_low
                        cand.parsed_reference_range_high = range_res.range_high

                    if cand.date_raw:
                        p_date, d_status, _ = DateParser.parse_date(
                            cand.date_raw, effective_date_order
                        )
                        cand.parsed_date = p_date
                        cand.date_parse_status = d_status

                all_candidates.extend(candidates)

            except ExtractorUnavailable as err:
                # Fatal account-wide / configuration error (e.g. remote processing disabled)
                all_errors.append(
                    StructuredError(
                        code=err.code,
                        stage="clinical_extraction",
                        message=err.message,
                        details={"unit_id": unit.unit_id},
                    )
                )
                unprocessed_unit_ids.extend([u.unit_id for u in units[idx:]])
                break

            except Exception as err:
                # Chunk-local provider error
                consecutive_failures += 1
                all_warnings.append(
                    StructuredError(
                        code=ErrorCode.PROVIDER_RESPONSE_INVALID.value,
                        stage="clinical_extraction",
                        message=f"Chunk extraction error on unit '{unit.unit_id}': {type(err).__name__}",
                        details={"unit_id": unit.unit_id},
                    )
                )

        stage_results["clinical_extraction"] = StageResult(
            stage_name="clinical_extraction",
            status="success" if not unprocessed_unit_ids and not all_errors else "partial",
            counts={
                "model_calls": model_call_count,
                "unprocessed_units": len(unprocessed_unit_ids),
            },
        )
        stage_results["evidence_validation"] = StageResult(
            stage_name="evidence_validation",
            status="success",
            counts={"accepted_candidates": len(all_candidates)},
        )

        # ----------------------------------------------------------------------
        # Stage 3: Post-Processing Deduplication & Conflicts
        # ----------------------------------------------------------------------
        final_candidates, dedup_warnings = self.deduplicator.deduplicate_and_detect_conflicts(
            all_candidates
        )
        all_warnings.extend(dedup_warnings)

        stage_results["post_processing"] = StageResult(
            stage_name="post_processing",
            status="success",
            counts={
                "initial_candidates": len(all_candidates),
                "final_candidates": len(final_candidates),
                "conflicts_flagged": len(dedup_warnings),
            },
        )

        # Overall Status
        final_status: Literal["success", "partial", "failed"]
        if all_errors or unprocessed_unit_ids:
            final_status = "partial"
        else:
            final_status = "success"

        # ----------------------------------------------------------------------
        # Stage 4: Audit Metadata Assembly
        # ----------------------------------------------------------------------
        audit = AuditMetadata(
            provider=active_provider_name,
            model_id=active_model_id,
            schema_enforcement=schema_enforcement,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            schema_version="1.0.0",
            model_call_count=model_call_count,
            provider_attempt_count=provider_attempt_count,
            provider_request_ids=provider_request_ids,
            token_usage=token_usage,
            date_order_rule=effective_date_order,
            python_version=platform.python_version(),
            installed_versions={
                "pydantic": "2.12.5",
                "pydantic-settings": "2.13.1",
                "groq": "1.7.0",
            },
            key_settings={
                "max_llm_input_chars": self.chunker.max_input_chars,
                "max_llm_calls_per_request": self.max_llm_calls,
                "max_consecutive_chunk_failures": self.max_consecutive_chunk_failures,
            },
        )

        duration = (time.perf_counter() - start_time) * 1000.0

        return AgentResult(
            schema_version="1.0.0",
            request_id=bundle.request_id,
            agent_name=agent_name,
            source_type=bundle.source_type,
            status=final_status,
            sanitized_source_metadata=bundle.safe_metadata,
            content_sha256=bundle.safe_metadata.get("content_sha256", ""),
            candidates=final_candidates,
            stage_results=stage_results,
            warnings=all_warnings,
            errors=all_errors,
            audit=audit,
            created_at_utc=datetime.now(UTC),
            duration_ms=round(duration, 3),
            contains_source_text=effective_options.include_source_text_in_result,
            debug_source_text=original_text
            if effective_options.include_source_text_in_result
            else None,
        )

    def process(
        self,
        bundle: SourceTextBundle,
        language_hint: str | None = None,
        date_order_hint: str | None = None,
        options: (
            TextAgentOptions | PDFAgentOptions | ImageAgentOptions | VoiceAgentOptions | None
        ) = None,
        agent_name: str = "text_agent",
        original_text: str | None = None,
    ) -> AgentResult:
        """Alias for process_bundle for backward compatibility."""
        return self.process_bundle(
            bundle=bundle,
            language_hint=language_hint,
            date_order_hint=date_order_hint,
            options=options,
            agent_name=agent_name,
            original_text=original_text,
        )
