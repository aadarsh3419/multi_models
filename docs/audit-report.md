# Healthcare Text Agent Audit Report (Milestone 1)

**Audit Date**: 2026-10-16  
**Auditor**: Senior Python Code Reviewer & AI Test Engineer  
**Scope**: In-place verification of codebase, test suite, benchmark evaluation, CLI, architecture boundaries, and scope constraints.

---

## 1. Executive Summary

An exhaustive in-place audit of the Milestone 1 repository was conducted. Every claimed feature, test suite, benchmark script, and architectural boundary was executed and verified against actual Python 3.14 runtime behavior.

### High-Level Findings:
1. **Reproducibility**: All 46 offline unit and integration tests pass cleanly (1 live API test appropriately skipped). Code formatting and linting pass all Ruff rules without error.
2. **Deterministic Architecture**: The core Clean Architecture / Hexagonal structure is genuinely implemented. AST inspection tests confirm `contracts.py`, `workflow.py`, and `text_agent.py` contain zero direct imports of the Groq SDK.
3. **Evidence Grounding**: The mathematical invariant `raw_text[start:end] == quote` is strictly verified. The pipeline never trusts model-provided offsets, counts Python Unicode code points, and explicitly marks repeated quotations in the same segment as `ambiguous_match` (as proven by case 032).
4. **Benchmark Verification**: The reported `1.0000` precision and recall is reproducible, but represents an **offline software verification of the workflow engine and evidence validator using a deterministic test double**. It does **not** represent live Groq model extraction accuracy.
5. **Scope Deviation**: A localhost web server (`scripts/serve_local.py`) was introduced during development. It uses custom regex extraction rather than the LLM workflow and binds to `0.0.0.0` with external CDN dependencies. This represents a confirmed scope deviation and minor security risk.

---

## 2. Executed Commands & Actual Results

| Command | Exit Code | Results / Output Summary | Verification Status |
|---|---|---|---|
| `python -m pytest tests/ -v` | `0` | **46 passed, 1 skipped** in 12.99s. Zero failures. | **VERIFIED** |
| `python -m ruff check .` | `0` | `All checks passed!` across 29 files. | **VERIFIED** |
| `python -m ruff format --check .` | `0` | `29 files already formatted`. Zero drift. | **VERIFIED** |
| `python run.py` | `0` | Step 1 Demo: 7 candidates extracted & coordinates verified (1.43ms).<br>Step 2 Benchmark: 38 cases evaluated (P: 1.0, R: 1.0, F1: 1.0). | **VERIFIED** |
| `python scripts/evaluate_text_extraction.py` | `0` | 38 cases evaluated. 41 TP, 0 FP, 0 FN. Grounded TP: 40, Grounded FN: 1. | **VERIFIED** |
| `healthcare-text extract-text --stdin` | `3` | Piped input accepted; exits with `3` (`partial`) because remote LLM is safely disabled. Zero PHI leaked to stderr. | **VERIFIED** |

---

## 3. Claim-versus-Evidence Traceability Matrix

| README Claim / Specification | Implementation File | Key Class / Function | Proving Test | Status |
|---|---|---|---|---|
| **TextAgent calls ExtractionWorkflow** | `src/healthcare_agents/text_agent.py` | `TextAgent.process()` | `test_text_agent_full_pipeline` (`test_gate_b.py`) | **VERIFIED** |
| **ExtractionWorkflow calls ClinicalFieldExtractor interface** | `src/healthcare_agents/workflow.py` | `ExtractionWorkflow.process()` | `test_workflow_with_fake_extractor_success` (`test_gate_b.py`) | **VERIFIED** |
| **AST boundary decoupling (no groq in workflow/contracts)** | `src/healthcare_agents/workflow.py`, `contracts.py` | AST module inspection | `test_workflow_has_no_direct_groq_import`, `test_contracts_has_no_provider_or_agent_dependencies` (`test_architecture.py`) | **VERIFIED** |
| **Official Groq SDK adapter & error handling** | `src/healthcare_agents/groq_extractor.py` | `GroqFieldExtractor.extract()` | `test_groq_extractor_mocked_success`, `test_groq_extractor_fatal_auth_error_not_retried`, `test_groq_extractor_rate_limit_with_retry_after` (`test_groq_extractor.py`) | **VERIFIED** |
| **Offline safety gate (no outbound calls when disabled)** | `src/healthcare_agents/groq_extractor.py`, `config.py` | `allow_remote_llm_processing=False` | `test_groq_extractor_remote_processing_disabled_rejection` (`test_groq_extractor.py`), `test_workflow_provider_skipped_reports_partial` (`test_gate_b.py`) | **VERIFIED** |
| **Evidence quote verification & code-point offset mapping** | `src/healthcare_agents/evidence_validator.py` | `EvidenceValidator.validate_single_reference()` | `test_evidence_validator_exact_match`, `test_evidence_validator_whitespace_normalized_fallback` (`test_gate_a.py`) | **VERIFIED** |
| **Repeated quotes marked ambiguous (no arbitrary selection)** | `src/healthcare_agents/evidence_validator.py` | `EvidenceValidator._find_all_substring_indices()` | `test_evidence_validator_duplicate_quote_marked_ambiguous` (`test_gate_a.py`), Case 032 in benchmark | **VERIFIED** |
| **ValueParser (scalars, comparators, ranges, BP allowlist)** | `src/healthcare_agents/value_parser.py` | `ValueParser.parse()` | `test_value_parser_scalars`, `test_value_parser_comparators`, `test_value_parser_ranges`, `test_value_parser_blood_pressure_allowlist`, `test_value_parser_ambiguous_commas` (`test_gate_a.py`) | **VERIFIED** |
| **DateParser (ISO, calendar check, DMY/MDY, no year guessing)** | `src/healthcare_agents/value_parser.py` | `DateParser.parse()` | `test_date_parser_iso_standard`, `test_date_parser_invalid_calendar_date`, `test_date_parser_unambiguous_day_gt_12`, `test_date_parser_ambiguous_date_under_unknown`, `test_date_parser_missing_year` (`test_gate_a.py`) | **VERIFIED** |
| **Exact candidate deduplication** | `src/healthcare_agents/deduplicator.py` | `CandidateDeduplicator.deduplicate_and_detect_conflicts()` | `test_candidate_deduplication_exact_match` (`test_gate_a.py`) | **VERIFIED** |
| **Defensible conflict detection (retaining both observations)** | `src/healthcare_agents/deduplicator.py` | `CandidateDeduplicator.deduplicate_and_detect_conflicts()` | `test_candidate_conflict_detection_same_date_differing_values` (`test_gate_a.py`), Case 033 in benchmark | **VERIFIED** |
| **Clinical schema semantics (experiencer, temporality, roles)** | `src/healthcare_agents/contracts.py` | `FactCandidate`, `AssertionStatus`, `Temporality`, `ExperiencerType` | Validated in Gate A models and fixture cases 006, 010, 011, 018 | **VERIFIED** |
| **TextChunker zero-drop coverage & fragment tracking** | `src/healthcare_agents/chunker.py` | `TextChunker.chunk_bundle()`, `_fragment_large_segment()` | `test_chunker_small_text_single_chunk`, `test_chunker_large_text_multi_fragments_no_character_drops` (`test_gate_b.py`) | **VERIFIED** |
| **Call budget enforcement & partial results** | `src/healthcare_agents/workflow.py` | `ExtractionWorkflow.process()` | `test_workflow_call_budget_exhaustion` (`test_gate_b.py`) | **VERIFIED** |
| **Prompt injection defense (adversarial source narrative)** | `src/healthcare_agents/prompt_templates.py`, `workflow.py` | `SYSTEM_PROMPT` untrusted rule | `test_prompt_injection_adversarial_handling` (`test_gate_b.py`), Case 034 in benchmark | **VERIFIED** |
| **CLI exit codes (0, 1, 2, 3)** | `src/healthcare_agents/cli.py` | `run_extract_text()`, `main()` | `test_cli_help`, `test_cli_extract_text_stdin_success`, `test_cli_nonexistent_file`, `test_cli_empty_file`, `test_cli_oversized_file` (`test_cli.py`) | **VERIFIED** |
| **PHI protection in logs & exception representations** | `src/healthcare_agents/contracts.py`, `source_text.py` | `__repr__`, `__str__` | `test_text_agent_request_phi_safe_repr`, `test_text_agent_request_size_limit` (`test_gate_a.py`) | **VERIFIED** |
| **Audit metadata generation** | `src/healthcare_agents/contracts.py`, `workflow.py` | `AuditMetadata`, `StageOutcome` | Verified in `test_text_agent_full_pipeline`, `demo_extraction.py`, CLI JSON output | **VERIFIED** |

---

## 4. Confirmed Defects & Technical Debt

### Defect 1: Insecure Network Interface Binding in `scripts/serve_local.py`
- **Location**: `scripts/serve_local.py:578` (`server_address = ("", port)`)
- **Severity**: Medium
- **Description**: Passing `""` to `HTTPServer` binds to `0.0.0.0` (all network interfaces), exposing the unauthenticated local HTTP server to the local area network (LAN/Wi-Fi). In a clinical or developer setting, this could allow third parties on the same subnet to submit queries or observe traffic.
- **Remediation**: Bind strictly to loopback: `server_address = ("127.0.0.1", port)`.

### Defect 2: Mocked Regex Extractor Divergence in `scripts/serve_local.py`
- **Location**: `scripts/serve_local.py:34-180` (`LocalInteractiveExtractor`)
- **Severity**: Low (Architectural Consistency)
- **Description**: `serve_local.py` implements an ad-hoc regex extractor (`re.search(r"(?:BP|Blood Pressure)...")`) directly inside the script rather than utilizing the shared pipeline's `GroqFieldExtractor` (when API keys are present) or the benchmark test double. As a result, the web UI behaves differently from the real Text Agent when arbitrary text is entered.
- **Remediation**: Refactor `serve_local.py` to use `build_text_agent()` from `healthcare_agents.cli`, maintaining strict architectural consistency.

### Defect 3: External CDN Dependency in Offline Web Interface
- **Location**: `scripts/serve_local.py:200` (`<script src="https://cdn.tailwindcss.com"></script>`)
- **Severity**: Low (Offline Integrity)
- **Description**: The web interface in `serve_local.py` attempts to fetch Tailwind CSS from an external CDN. If the developer or auditor runs offline without internet access, CSS styling will fail to load.
- **Remediation**: Use inline vanilla CSS or the allowlisted static asset bundle.

---

## 5. Potential Defects & Edge Cases Remaining Unverified

### Potential Defect A: Cross-Fragment Evidence Quotes
- **Context**: When a continuous paragraph exceeds 6,000 characters, `TextChunker` fragments it into smaller segments with parent coordinate tracking.
- **Risk**: If a model cites a multi-word quotation that straddles the boundary between fragment $N$ and fragment $N+1$, `EvidenceValidator` evaluates each fragment independently. The quotation will be reported as `unmatched` in both fragments rather than reassembling the parent span.
- **Current Mitigation**: Overlap of 200 characters is used when boundary splits cannot be found, reducing but not mathematically eliminating boundary straddling.

### Potential Defect B: Malformed LLM Responses from Groq
- **Context**: `GroqFieldExtractor.extract()` executes `json.loads(content_str)` followed by `ExtractionResponse.model_validate()`.
- **Risk**: If Groq terminates generation prematurely (e.g. `finish_reason: length`) resulting in truncated/unparseable JSON, `json.loads` raises `json.JSONDecodeError`. Currently, this is caught by the generic `except Exception` block and retried. However, there is no specialized test in `tests/integration/test_groq_extractor.py` verifying behavior under truncated JSON.

---

## 6. Scope Deviations & Audit of Web UI

### Assessment of `scripts/serve_local.py`:
- **Milestone 1 Scope Constraint**: Section 2 explicitly stated: *"Do not implement: FastAPI, frontend, JWT or authentication."*
- **Origin of Script**: `scripts/serve_local.py` was introduced in turn 6 to satisfy a direct user instruction (*"local host mai chalo"*).
- **Compliance Status**: It avoids third-party web frameworks (FastAPI) by relying entirely on Python standard library `http.server`. However, it introduces an auxiliary web frontend and an independent regex extractor not specified in the milestone requirements.
- **Recommendation**: Retain `scripts/serve_local.py` for developer demonstration, but isolate it in documentation as an **optional developer tool**. Do not package it into the production core distribution.

---

## 7. Honest Benchmark Evaluation Analysis

The README advertises **1.0000 Precision and 1.0000 Recall** across 38 cases. An auditor must ask: *Where do these numbers come from?*

### The Truth of the Metrics:
1. **Extractor Used**: When `scripts/evaluate_text_extraction.py` runs without `--live-groq`, it instantiates `SyntheticBenchmarkFakeExtractor`. This test double reads the expected facts from the gold fixture and submits them to the pipeline.
2. **What the 1.0 Metric Truly Proves**:
   - It proves that the **downstream pipeline** (SourceTextBundle segmentation, EvidenceValidator quote matching, coordinate calculation, ValueParser, DateParser, and CandidateDeduplicator) losslessly parses, validates, and retains 100% of validly formatted clinical facts without dropping attributes or mutating values.
   - It proves that the **ExtractionScorer** correctly implements strict deterministic matching rules (fact type, NFC label, exact unit, assertion, temporality, experiencer).
3. **What the 1.0 Metric Does NOT Prove**:
   - It does **not** prove that a live Groq LLM (e.g. `openai/gpt-oss-20b`) achieves 1.0 precision/recall on clinical text.
   - It does **not** prove clinical accuracy or real-world patient safety.
4. **Why 38 Cases Equal 41 True Positives**:
   - Several cases contain multiple distinct facts:
     - `case_006`: 2 negated symptoms (`chest pain` and `shortness of breath`).
     - `case_033`: 2 conflicting measurements (`Fasting Glucose 95` and `Fasting Glucose 145`).
     - `case_035`: 2 Hindi facts (`तेज बुखार` symptom and `रक्तचाप 130/85` measurement).
     - `case_036`: 2 Hinglish facts (`dry cough` symptom and `Paracetamol 650 mg` medication).
     - `case_034`: Prompt injection case with **0 expected facts** (yielding $TP=0, FP=0, FN=0 \implies \text{Precision: N/A}$).
   - Total expected facts across all 38 cases: $33 \times 1 + 4 \times 2 + 1 \times 0 = 41\text{ facts}$. When all 41 facts are extracted and validated, $TP = 41$.

---

## 8. Prioritized Defect Repair Recommendations

| Priority | Issue | Affected File(s) | Proposed Remediation |
|---|---|---|---|
| **P1** | **Insecure Network Binding** | `scripts/serve_local.py` | Change `("", port)` to `("127.0.0.1", port)` to prevent external network exposure. |
| **P2** | **External CDN Dependency** | `scripts/serve_local.py` | Remove external Tailwind script tag; replace with self-contained CSS styles for offline resilience. |
| **P3** | **Architecture Consistency in Localhost Server** | `scripts/serve_local.py` | Replace custom regex extraction logic with standard `build_text_agent()` and `ExtractionWorkflow`. |
| **P4** | **Missing Truncated JSON Test** | `tests/integration/test_groq_extractor.py` | Add unit test verifying that truncated/malformed JSON from the LLM raises a retryable or structured error. |
| **P5** | **README Metric Qualification** | `README.md` | Add explicit callout in the README summary clarifying that the 1.0 benchmark metric reflects the offline pipeline verification test double, not live Groq performance. |

---

*Report prepared and certified by Senior Python Code Reviewer & AI Test Engineer.*

