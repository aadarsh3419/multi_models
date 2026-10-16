"""Shared clinical extraction workflow.

Synchronous pipeline engine that coordinates:
1. Input/segment bounds validation.
2. Bounded chunking.
3. Provider invocation through the ClinicalFieldExtractor interface.
4. Evidence verification and source coordinate mapping.
5. Deterministic value and date parsing.
6. Safe candidate deduplication and defensible conflict detection.
7. Call budget enforcement and partial result handling.

Follows strict architectural boundaries: zero direct Groq SDK imports.
"""

import time
from datetime import UTC, datetime

from healthcare_agents.chunker import TextChunk, TextChunker
from healthcare_agents.contracts import (
    AgentResult,
    AgentStatus,
    AssertionStatus,
    AuditMetadata,
    ExperiencerType,
    ExtractionCandidateRaw,
    FactCandidate,
    FactType,
    ParseStatus,
    ReportingSourceRole,
    StageOutcome,
    StructuredError,
    Temporality,
    TextAgentOptions,
)
from healthcare_agents.deduplicator import (
    CandidateDeduplicator,
    generate_candidate_id,
)
from healthcare_agents.evidence_validator import EvidenceValidator
from healthcare_agents.extractor_interface import (
    ClinicalFieldExtractor,
    ExtractorCallMetadata,
)
from healthcare_agents.prompt_templates import (
    PROMPT_VERSION,
    get_prompt_template_hash,
)
from healthcare_agents.source_text import SourceTextBundle
from healthcare_agents.value_parser import DateParser, ValueParser


class ExtractionWorkflow:
    """Core orchestration engine for healthcare text extraction."""

    def __init__(
        self,
        extractor: ClinicalFieldExtractor | None = None,
        chunker: TextChunker | None = None,
        date_order: str = "UNKNOWN",
        max_llm_calls_per_request: int = 8,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker or TextChunker()
        self.date_order = date_order
        self.max_calls = max_llm_calls_per_request

    def process(
        self,
        bundle: SourceTextBundle,
        options: TextAgentOptions | None = None,
    ) -> AgentResult:
        """Execute the end-to-end extraction workflow on a SourceTextBundle."""
        start_time = time.perf_counter()
        opts = options or TextAgentOptions()
        date_order = opts.date_order or self.date_order

        stage_outcomes: dict[str, StageOutcome] = {}
        warnings: list[str] = list(bundle.processing_warnings)
        errors: list[StructuredError] = []
        raw_candidates_collected: list[ExtractionCandidateRaw] = []

        total_provider_attempts = 0
        last_call_meta: ExtractorCallMetadata | None = None

        # Stage 1: Segmentation outcome
        stage_outcomes["segmentation"] = StageOutcome(
            stage="segmentation",
            status="success",
            message=f"Segments created: {len(bundle.segments)}",
        )

        # Stage 2: Chunking
        chunk_t0 = time.perf_counter()
        chunks: list[TextChunk] = self.chunker.chunk_bundle(bundle)
        chunk_duration = (time.perf_counter() - chunk_t0) * 1000
        stage_outcomes["chunking"] = StageOutcome(
            stage="chunking",
            status="success",
            duration_ms=chunk_duration,
            message=f"Created {len(chunks)} bounded chunk(s)",
        )

        # Stage 3: Extraction via Provider Interface
        extract_t0 = time.perf_counter()
        extraction_status = "success"
        workflow_overall_status = AgentStatus.success

        if self.extractor is None:
            # Extractor not configured or remote processing disabled
            stage_outcomes["extraction"] = StageOutcome(
                stage="extraction",
                status="skipped",
                duration_ms=0.0,
                message="Extractor not configured or remote LLM processing disabled.",
            )
            warnings.append("Extraction stage skipped: no active extractor available.")
            workflow_overall_status = AgentStatus.partial
        else:
            unprocessed_chunks: list[str] = []
            for idx, chunk in enumerate(chunks):
                # Check call budget before each attempt
                if total_provider_attempts >= self.max_calls:
                    unprocessed_chunks = [c.chunk_id for c in chunks[idx:]]
                    msg = (
                        f"Call budget exhausted ({total_provider_attempts} attempts >= {self.max_calls}). "
                        f"Unprocessed chunks: {unprocessed_chunks}"
                    )
                    warnings.append(msg)
                    errors.append(
                        StructuredError(
                            code="CALL_BUDGET_EXHAUSTED",
                            message=msg,
                            stage="extraction",
                            retryable=False,
                        )
                    )
                    extraction_status = "partial"
                    workflow_overall_status = AgentStatus.partial
                    break

                try:
                    resp, meta = self.extractor.extract(
                        segments=chunk.segments,
                        language_hint=opts.language_hint,
                    )
                    last_call_meta = meta
                    total_provider_attempts += meta.attempts_made
                    raw_candidates_collected.extend(resp.candidates)

                except Exception as err:
                    # Capture provider failure safely without leaking PHI
                    err_msg = f"Provider invocation failed on chunk {chunk.chunk_id}: {type(err).__name__}"
                    warnings.append(err_msg)
                    errors.append(
                        StructuredError(
                            code="PROVIDER_ERROR",
                            message=err_msg,
                            stage="extraction",
                            retryable=False,
                        )
                    )
                    unprocessed_chunks = [c.chunk_id for c in chunks[idx:]]
                    extraction_status = "partial"
                    workflow_overall_status = AgentStatus.partial
                    break

            extract_duration = (time.perf_counter() - extract_t0) * 1000
            stage_outcomes["extraction"] = StageOutcome(
                stage="extraction",
                status=extraction_status,
                duration_ms=extract_duration,
                message=f"Extracted {len(raw_candidates_collected)} raw candidate(s) across {total_provider_attempts} attempt(s)",
            )

        # Stage 4: Evidence Validation, Value Parsing, and Candidate Construction
        val_t0 = time.perf_counter()
        domain_candidates: list[FactCandidate] = []

        for raw_cand in raw_candidates_collected:
            # 4a. Validate evidence quotes against SourceTextBundle
            val_refs, ev_status = EvidenceValidator.validate_candidate_evidence(raw_cand, bundle)

            # 4b. Parse value deterministically
            val_parse = ValueParser.parse(
                raw_value=raw_cand.value_raw,
                field_context=raw_cand.label_raw,
            )

            # 4c. Parse date deterministically
            date_parse = DateParser.parse(
                raw_date=raw_cand.date_raw,
                date_order=date_order,
            )

            # Determine aggregate parse status
            if val_parse.parse_status == ParseStatus.parsed and (
                date_parse.parse_status in {ParseStatus.parsed, ParseStatus.not_stated}
            ):
                aggregate_parse_status = ParseStatus.parsed
            elif (
                val_parse.parse_status == ParseStatus.ambiguous
                or date_parse.parse_status == ParseStatus.ambiguous
            ):
                aggregate_parse_status = ParseStatus.ambiguous
            elif (
                val_parse.parse_status == ParseStatus.invalid
                or date_parse.parse_status == ParseStatus.invalid
            ):
                aggregate_parse_status = ParseStatus.invalid
            elif val_parse.parse_status == ParseStatus.not_applicable:
                aggregate_parse_status = ParseStatus.not_applicable
            else:
                aggregate_parse_status = ParseStatus.not_stated

            # Safe enum parsing
            fact_type = self._parse_enum(FactType, raw_cand.fact_type, FactType.other)
            assertion = self._parse_enum(
                AssertionStatus, raw_cand.assertion_status, AssertionStatus.uncertain
            )
            temporality = self._parse_enum(Temporality, raw_cand.temporality, Temporality.uncertain)
            experiencer = self._parse_enum(
                ExperiencerType, raw_cand.experiencer_type, ExperiencerType.unknown
            )
            reporting_role = self._parse_enum(
                ReportingSourceRole,
                raw_cand.reporting_source_role,
                ReportingSourceRole.unknown,
            )

            # Extract coordinates for candidate ID generation
            evidence_spans = [(r.segment_id, r.start_offset, r.end_offset) for r in val_refs]

            candidate_id = generate_candidate_id(
                source_content_hash=bundle.source_content_hash,
                fact_type=fact_type.value,
                label_raw=raw_cand.label_raw,
                value_raw=raw_cand.value_raw,
                unit_raw=raw_cand.unit_raw,
                assertion_status=assertion.value,
                temporality=temporality.value,
                experiencer_type=experiencer.value,
                evidence_spans=evidence_spans,
            )

            cand = FactCandidate(
                candidate_id=candidate_id,
                fact_type=fact_type,
                label_raw=raw_cand.label_raw,
                value_raw=raw_cand.value_raw,
                unit_raw=raw_cand.unit_raw,
                reference_range_raw=raw_cand.reference_range_raw,
                abnormal_flag_raw=raw_cand.abnormal_flag_raw,
                date_raw=raw_cand.date_raw,
                time_raw=raw_cand.time_raw,
                qualifier_raw=raw_cand.qualifier_raw,
                dose_raw=raw_cand.dose_raw,
                frequency_raw=raw_cand.frequency_raw,
                route_raw=raw_cand.route_raw,
                duration_raw=raw_cand.duration_raw,
                assertion_status=assertion,
                temporality=temporality,
                experiencer_type=experiencer,
                experiencer_relationship_raw=raw_cand.experiencer_relationship_raw,
                reporting_source_role=reporting_role,
                evidence_refs=val_refs,
                evidence_status=ev_status,
                value_parsed=val_parse,
                date_parsed=date_parse,
                parse_status=aggregate_parse_status,
            )
            domain_candidates.append(cand)

        val_duration = (time.perf_counter() - val_t0) * 1000
        stage_outcomes["evidence_and_parsing"] = StageOutcome(
            stage="evidence_and_parsing",
            status="success",
            duration_ms=val_duration,
            message=f"Validated and parsed {len(domain_candidates)} candidate(s)",
        )

        # Stage 5: Deduplication and Conflict Detection
        dedup_t0 = time.perf_counter()
        final_candidates, conflict_warnings = (
            CandidateDeduplicator.deduplicate_and_detect_conflicts(domain_candidates)
        )
        warnings.extend(conflict_warnings)
        dedup_duration = (time.perf_counter() - dedup_t0) * 1000
        stage_outcomes["deduplication"] = StageOutcome(
            stage="deduplication",
            status="success",
            duration_ms=dedup_duration,
            message=f"Final candidates: {len(final_candidates)} (conflicts flagged: {len(conflict_warnings)})",
        )

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Build Audit Metadata
        token_usage_dict: dict[str, int] | None = None
        if last_call_meta and (
            last_call_meta.input_tokens is not None or last_call_meta.output_tokens is not None
        ):
            token_usage_dict = {
                "input_tokens": last_call_meta.input_tokens or 0,
                "output_tokens": last_call_meta.output_tokens or 0,
            }

        audit = AuditMetadata(
            project_version="0.1.0",
            schema_version="1.0.0",
            request_id=str(bundle.request_id),
            source_content_hash=bundle.source_content_hash,
            provider_name=last_call_meta.provider_name if last_call_meta else None,
            model_id=last_call_meta.model_id if last_call_meta else None,
            prompt_version=PROMPT_VERSION,
            prompt_template_hash=get_prompt_template_hash(),
            provider_schema_version="1.0.0",
            provider_response_id=last_call_meta.response_id if last_call_meta else None,
            token_usage=token_usage_dict,
            actual_provider_attempt_count=total_provider_attempts,
            configured_date_order=date_order,
            stage_outcomes=stage_outcomes,
        )

        return AgentResult(
            schema_version="1.0.0",
            request_id=bundle.request_id,
            agent_name="text",
            status=workflow_overall_status,
            source_metadata=bundle.source_metadata,
            candidates=final_candidates,
            stage_outcomes=stage_outcomes,
            warnings=warnings,
            errors=errors,
            audit_metadata=audit,
            created_at=datetime.now(UTC),
            duration_ms=total_duration_ms,
        )

    @staticmethod
    def _parse_enum(enum_cls, val: str | None, default_val):
        if not val:
            return default_val
        try:
            return enum_cls(val.lower().strip())
        except (ValueError, KeyError):
            return default_val
