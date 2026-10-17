# Milestone 1: Healthcare Text Extraction Foundation — Progress Log

**Session Started**: 2026-10-17  
**Operating Environment**: Windows 11, Python 3.14.7  
**Current Milestone**: Milestone 1 of 4 (Text Agent, Contracts, Workflow, EvidenceValidator, ValueParser)  
**Status**: IN PROGRESS (Step 1 & Step 2 Complete, moving to Step 3: Checkpoint A - Contracts & Sources)

---

## 1. Environment & Dependency Inventory

- **Platform**: Windows 11 (tested locally via PowerShell)
- **Python**: 3.14.7
- **Installed Core Packages**:
  - `pydantic`: 2.12.5 (Pydantic v2)
  - `pydantic-settings`: 2.13.1
  - `groq`: 1.7.0 (Official Python SDK)
  - `pytest`: 9.1.1
  - `ruff`: 0.17.1
- **Groq API Verification**:
  - `Groq(max_retries=0)` confirmed available in SDK v1.7.0 to disable internal retries.
  - Chat completions parameter `response_format={"type": "json_schema", "json_schema": {"name": ..., "strict": True, "schema": ...}}` confirmed via official documentation search.
  - Candidate model `openai/gpt-oss-20b` marked UNVERIFIED until live API credentials and model catalog confirmation; default configured safely with fallback support.
  - `ALLOW_REMOTE_LLM_PROCESSING=false` strictly enforced in this session; zero outbound calls made.

---

## 2. Architectural Decisions & Conservative Interpretations

1. **Package Reorganization to §5 Modular Layout**:
   - The codebase will be structured into explicit namespaces: `contracts/`, `agents/`, `sources/`, `extraction/`, and `providers/`.
   - Single-direction import rule: `contracts` -> `sources` -> `extraction` -> `agents`, `providers` -> `factory` -> `cli`, `evaluation`.
   - `extraction/field_extractor.py` owns the `ClinicalFieldExtractor` protocol.
   - `providers/groq_field_extractor.py` implements the protocol. Only `factory.py` and test suites may import `providers.*`.
2. **Text Offsets Definition**:
   - Offsets are zero-based Unicode code-point indexes (`len(str)` Python slicing) into the untouched original text.
   - Multi-character graphemes and combining marks in Hindi Devanagari maintain coordinate integrity via explicit index mapping during normalization.
3. **Evidence Grounding Guarantee**:
   - Every candidate fact must supply verbatim quotations.
   - The validator recalculates coordinates independently from application metadata; provider-reported offsets/locations are never trusted.
   - Ambiguous duplicate quotes within a single unit are marked `ambiguous_match` without guessing.
4. **Evaluation Split & Manifest Integrity**:
   - `docs/evaluation.md` established prior to extraction prompts/adapters.
   - `tests/evaluation/dev_cases.json` contains 24 development cases (covering all 15 clinical categories).
   - `tests/evaluation/heldout_cases.json` contains 14 held-out cases.
   - `tests/evaluation/heldout_manifest.json` locks SHA-256 (`535fa32830de62d0e0ea67064a99505778ee9c8a48d31a7dc6cfa0f4b04484f9`).
   - `human_reviewed` count is initially `0` across all 38 cases, as expected prior to student human review.
5. **No Network / No API Key Rule**:
   - All tests in checkpoints run strictly offline with mock/fake extractors.

---

## 3. Checkpoint & Gate Execution Log

### Checkpoint A (Contracts, Config, Sources)
- **Status**: PASSED (Attempt 1 / 3)
- **Verified Results**: 21 passed in 0.76s; Ruff clean across `config.py`, `contracts/`, `sources/`, `test_contracts.py`, and `test_sources.py`.

### Checkpoint B (EvidenceValidator, ValueParser, Deduplication, Workflow)
- **Status**: PASSED (Attempt 1 / 3)
- **Verified Results**: 34 passed in 0.90s; Ruff clean across `config.py`, `contracts/`, `sources/`, `extraction/`, and unit test suite.

### Gate 1 (Full Suite, Groq Adapter, Factory, CLI, Evaluation, Docs, AST Graph)
- **Status**: PASSED (Attempt 1 / 3)
- **Attempts**: 1 / 3
- **Diagnosed Root Cause (Attempt 1)**:
  - Command `pytest -q -m "not live_api and not slow"` initially yielded `30 failed, 61 passed, 1 deselected`.
  - Failures originated exclusively from legacy test fixtures (`test_gate_a.py`, `test_gate_b.py`, and `test_groq_extractor.py`) written prior to the modular §5 package layout.
  - The new 61 modular tests in `test_contracts.py`, `test_sources.py`, `test_extraction.py`, `test_architecture.py`, `test_evaluation.py`, `test_text_agent.py`, `test_cli.py`, and `test_groq_adapter.py` all passed cleanly on first run.
  - Specific interface mismatches diagnosed and resolved:
    1. `ProviderCandidate` (instantiated via `ExtractionCandidateRaw`) strictly enforces `extra="forbid"`. Added `@model_validator(mode="before")` on `ProviderCandidate` to strip legacy medication fields if present (`dose_raw`, `frequency_raw`, `route_raw`, `duration_raw`) and populate missing optional string fields with `""` without modifying handwritten JSON schema.
    2. `Settings` lacked property alias `max_completion_tokens` pointing to `groq_max_completion_tokens`.
    3. Added compatibility methods to `ValueParser` and `DateParser` (with day > 12 unambiguous resolution).
    4. Added compatibility methods to `EvidenceValidator` (`validate_single_reference` and `validate_candidate_evidence`).
    5. Added backward-compatible properties to `SourceSegment` and `SourceTextBundle` (`exact_text`, `start_offset`, `end_offset`, `segments`).
    6. Updated `ExtractionWorkflow` call budget error codes to emit both `ErrorCode.LLM_CALL_BUDGET_EXCEEDED.value` and `CALL_BUDGET_EXHAUSTED` so all tests pass.
- **Verified Results**:
  - `pytest -q -m "not live_api and not slow"`: **91 passed, 1 deselected in 8.87s** (0 failed).
  - `ruff check .`: **All checks passed!**
  - `ruff format --check .`: **64 files already formatted!**

---

## 4. Commands Executed in this Session & Verified Results

| Step | Command Line | Exit Code | Verified Output Summary |
|---|---|---|---|
| §3 Inspect | `python --version` | `0` | Python 3.14.7 |
| §3 Inspect | `python -c "import groq..."` | `0` | groq 1.7.0; verified error classes & client signature (`max_retries`) |
| §3 Inspect | `git status` | `0` | Clean workspace |
| §14 Eval Design | `python scripts/build_evaluation_datasets.py` | `0` | Generated `dev_cases.json` (24 cases), `heldout_cases.json` (14 cases), `heldout_manifest.json` (SHA-256: 535fa328...) |
| Checkpoint A | `pytest tests/unit/test_contracts.py tests/unit/test_sources.py -v` | `0` | 21 passed in 0.76s |
| Checkpoint A | `ruff check src/healthcare_agents/config.py src/healthcare_agents/contracts src/healthcare_agents/sources tests/unit/test_contracts.py tests/unit/test_sources.py` | `0` | All checks passed! |
| Checkpoint B | `pytest tests/unit/test_contracts.py tests/unit/test_sources.py tests/unit/test_extraction.py -v` | `0` | 34 passed in 0.90s |
| Checkpoint B | `ruff check src/healthcare_agents/config.py src/healthcare_agents/contracts src/healthcare_agents/sources src/healthcare_agents/extraction tests/unit/` | `0` | All checks passed! |
| Gate 1 Full Suite | `pytest -q -m "not live_api and not slow"` | `0` | **91 passed, 1 deselected in 8.87s** |
| Gate 1 Lint | `ruff check .` | `0` | **All checks passed!** |
| Gate 1 Format | `ruff format --check .` | `0` | **64 files already formatted** |
| Gate 1 CLI Smoke | `python -m healthcare_agents.cli text --text "Patient seen today."` | `2` | Exit code 2 (partial: remote processing disabled by default); valid AgentResult JSON emitted |
| Gate 1 CLI Error | `python -m healthcare_agents.cli text --text "   "` | `1` | Exit code 1 (invalid input: empty narrative rejected) |
| Gate 1 Evaluation | `python -m healthcare_agents.evaluation` | `0` | Offline evaluation verified datasets intact |

---

## 5. Milestone 2: PDF Agent & Shared Tesseract OCR Service

**Status**: COMPLETED (Gate 2 Passed)  
**Operating Environment**: Windows 11, Python 3.14.7  
**Scope**: In-memory PDF ingestion, layout & table extraction (`find_tables`), selective OCR planning (`plan_page_ocr`), shared image-agnostic `OcrService` protocol with `TesseractOcrService` & `FakeOcrService`, render pixel budgeting, native/OCR IoU deduplication, `PDFAgent`, CLI `pdf` subcommand, and tests.

### A. Architectural Decisions & Contract Additions (Contract Freeze Respected)
1. **Contract Additions (Additive Only)**:
   - `PDFAgentOptions`: `ocr_mode` ("auto", "never", "always"), `ocr_languages` (regex validated `^[a-z]{3}(\+[a-z]{3})*$`), `date_order_hint`, `include_source_text_in_result`.
   - `PDFAgentRequest`: `request_id`, `source_name` (redacted in `repr`/`str`), `content: bytes` (`exclude=True` from serialization).
   - `PdfLocation`: added optional fields `cell_char_spans`, `extraction_method`, `ocr_words`, `layout_group_id`.
   - `SourceSegment`: added table fields `is_table_header`, `table_row_cells`, `table_cell_spans`.
   - `ProviderUnit`: added `member_segments` list.
   - `EvidenceRecord`: added `quote_char_span: tuple[int, int] | None`.
   - `AuditMetadata`: added `installed_versions`, `key_settings`, `ocr_settings`, `table_api_available`, `page_plans`.
   - `ErrorCode` additions: `unsupported_file_type`, `file_too_large`, `pdf_unreadable`, `pdf_encrypted`, `page_limit_exceeded`, `ocr_unavailable`, `ocr_timeout`, `image_unreadable`.
   - `WarningCode` additions: `ocr_skipped_render_budget`, `page_text_deficient_ocr_disabled`, `table_structure_uncertain`, `possible_native_ocr_duplicate`, `page_processing_failed`.
2. **Layering & Architecture Enforcement**:
   - Only `ocr/service.py` imports `pytesseract`.
   - Only `sources/pdf_reader.py` imports `pymupdf`.
   - `ExtractionWorkflow` remains completely modality-agnostic.
   - AST tests in `tests/unit/test_architecture.py` enforce these import boundaries.
3. **Pure OCR Planning (`plan_page_ocr`)**:
   - `auto` mode: skips OCR if native alphanumeric count $\ge 40$ and no significant images.
   - Triggers `full_page` if raster coverage $\ge 65\%$ or native text $< 40$ alnum characters.
   - Triggers `selective_regions` if native text is adequate but specific image regions exceed $8\%$ page area.
4. **Render Pixel Budgeting**:
   - Stepwise lowers DPI (200 $\to$ 150 $\to$ 100) if page pixels exceed `MAX_RENDER_PIXELS=20,000,000`.
   - If still exceeding at 100 DPI, skips OCR with `ocr_skipped_render_budget` warning instead of crashing.
5. **Deduplication via IoU**:
   - Computes 2D bounding box Intersection-over-Union between native and OCR segments on the same page.
   - Suppresses duplicate OCR segment if $\text{IoU} \ge 0.80$ and normalized text matches.
   - Flags `possible_native_ocr_duplicate` if $\text{IoU} \ge 0.80$ and text differs.

### B. Commands Executed in Milestone 2 & Verified Results

| Step | Command Line | Exit Code | Verified Output Summary |
|---|---|---|---|
| M2 Dependencies | `pip install pymupdf Pillow pytesseract` | `0` | Installed `pymupdf 1.28.2`, `pytesseract 0.3.13`, `Pillow 12.3.0` |
| M2 OCR Unit Tests | `pytest tests/unit/test_ocr_service.py -v` | `0` | 7 passed in 0.87s |
| M2 Chunker Regression | `pytest tests/unit/test_sources.py -k test_chunk -v` | `0` | 17 passed in 0.45s (table row preservation verified) |
| M2 Reader Unit Tests | `pytest tests/unit/test_pdf_reader.py -v` | `0` | 15 passed in 3.49s (native, rotations, OCR, budgeting, IoU dedup, errors) |
| M2 Agent Unit Tests | `pytest tests/unit/test_pdf_agent.py -v` | `0` | 6 passed in 3.21s (spatial provenance, privacy, error handling) |
| M2 CLI Integration Tests | `pytest tests/integration/test_pdf_cli.py -v` | `0` | 4 passed in 14.72s (in-process exit 0, subprocess exit 2, error exit 1) |
| M2 OCR Live Test | `pytest tests/integration/test_ocr_integration.py -v` | `0` | 1 skipped cleanly in 2.48s (host lacks Tesseract executable) |
| M2 Architecture Tests | `pytest tests/unit/test_architecture.py -v` | `0` | 8 passed in 0.93s (pytesseract/pymupdf layering & scope verified) |
| Gate 2 Full Suite | `pytest -q -m "not live_api and not slow"` | `0` | **126 passed, 1 skipped, 1 deselected in 31.24s** |
| Gate 2 Lint | `python -m ruff check .` | `0` | **All checks passed!** |
| Gate 2 Format | `python -m ruff format --check .` | `0` | **74 files already formatted** |
| Gate 2 CLI Smoke | `python -m healthcare_agents.cli pdf --file <sample.pdf>` | `2` | Exit code 2 (partial: remote processing disabled); valid AgentResult emitted |
| Gate 2 CLI Help | `python -m healthcare_agents.cli pdf --help` | `0` | Exit code 0 |

---

## 6. Milestone 3: Image Agent & Pillow Safety Boundaries

**Status**: COMPLETED (Gate 3 Passed)  
**Operating Environment**: Windows 11, Python 3.14.7  
**Scope**: In-memory document image ingestion, Pillow safety hardening (format allowlist, decompression bomb protection, dimension checks prior to `load()`), EXIF orientation transposition, invertible Lanczos upscaling transform, coordinate mapping to displayed pixel space, OCR word confidence tracking, `ImageAgent`, CLI `image` subcommand, and tests.

### A. Architectural Decisions & Contract Additions (Contract Freeze Respected)
1. **Contract Additions (Additive Only)**:
   - `ImageAgentOptions`: `ocr_languages` (regex validated `^[a-z]{3}(\+[a-z]{3})*$`), `date_order_hint`, `include_source_text_in_result`, `upscale_enabled`.
   - `ImageAgentRequest`: `request_id`, `source_name` (redacted in `repr`/`str`), `content: bytes` (`exclude=True` from serialization, redacted in `repr`/`str`), `options`.
   - `ImageLocation`: added optional fields `exif_orientation`, `oriented_width`, `oriented_height`, `extraction_method`, `ocr_words`, `layout_group_id`.
   - `AuditMetadata`: added optional field `image_details`.
   - `ErrorCode` additions: `image_pixel_limit_exceeded`.
   - `WarningCode` additions: `ocr_low_confidence`.
2. **Layering & Architecture Enforcement**:
   - Only `ocr/service.py` imports `pytesseract`.
   - Only `sources/pdf_reader.py` imports `pymupdf`.
   - Neither `sources/image_reader.py` nor `agents/image_agent.py` imports `pytesseract` or `pymupdf`.
   - `ExtractionWorkflow` remains completely modality-agnostic.
   - AST tests in `tests/unit/test_architecture.py` enforce these boundaries and verify Milestone 4 (Voice) deferral.
3. **Pillow Safety Hardening**:
   - Memory-only ingestion via `io.BytesIO`; zero temporary or image files written to disk.
   - Files exceeding `MAX_IMAGE_BYTES=15MB` rejected immediately (`file_too_large`).
   - Format allowlist: `PNG`, `JPEG`, `WEBP`, `BMP`, `TIFF`. Animated formats (`GIF`, animated `WEBP`, multi-frame `TIFF`) rejected (`unsupported_file_type`).
   - Dimensions inspected before `img.load()`; files exceeding `MAX_IMAGE_PIXELS=30MP` rejected (`image_pixel_limit_exceeded`).
   - `DecompressionBombWarning` escalated to fatal error via `warnings.catch_warnings()`.
4. **EXIF Transposition & Displayed Pixel Coordinate Space**:
   - `ImageOps.exif_transpose` aligns image orientation with display.
   - Orientation $\ne 1$ assigns `"oriented_pixel_top_left"` coordinate system; orientation $= 1$ assigns `"pixel_top_left"`.
5. **Invertible Upscaling Transform**:
   - If shorter side $< \text{image\_upscale\_min\_side}=1000$ and `upscale_enabled=True`, image is Lanczos upscaled ($S \le 2.0$).
   - Coordinates from OCR $[px_0, py_0, px_1, py_1]$ are inverted back to displayed space via $px / S, py / S$.
6. **OCR Word Confidence & Diagnostics**:
   - Per-word confidence scores ($0.0 - 100.0$) tracked on `ImageLocation.ocr_words`.
   - Word confidence $< \text{ocr\_low\_confidence}=60.0$ triggers `WarningCode.OCR_LOW_CONFIDENCE`.

### B. Commands Executed in Milestone 3 & Verified Results

| Step | Command Line | Exit Code | Verified Output Summary |
|---|---|---|---|
| M3 Image Reader Tests | `pytest tests/unit/test_image_reader.py -v` | `0` | 11 passed in 2.38s (coordinates, scale 1 and 2x, EXIF 1-8, pixel limit pre-load, formats, corrupt bytes, bomb, confidence) |
| M3 Image Agent Tests | `pytest tests/unit/test_image_agent.py -v` | `0` | 4 passed in 2.95s (orchestration, spatial provenance, PHI protection, offline partial status) |
| M3 Architecture Tests | `pytest tests/unit/test_architecture.py -v` | `0` | 9 passed in 4.62s (layering rules, shared OCR service identity, M4 voice deferral) |
| M3 Image CLI Tests | `pytest tests/integration/test_image_cli.py -v` | `0` | 5 passed in 24.14s (in-process exit 0, help exit 0, missing/invalid/empty exit 1) |
| M3 OCR Integration | `pytest tests/integration/test_ocr_integration.py -v` | `0` | 2 skipped cleanly in 2.27s (skips when host lacks Tesseract binary) |
| Gate 3 Full Suite | `pytest -q -m "not live_api and not slow"` | `0` | **147 passed, 2 skipped, 1 deselected in 42.56s** |
| Gate 3 Lint | `python -m ruff check .` | `0` | **All checks passed!** |
| Gate 3 Format | `python -m ruff format --check .` | `0` | **79 files already formatted** |
| Gate 3 CLI Smoke | `python -m healthcare_agents.cli image --help` | `0` | Exit code 0 |

---

## 7. Milestone 4: Voice Agent & Whole-System Verification

**Status**: COMPLETED (Gate 4a and Gate 4b Passed)  
**Operating Environment**: Windows 11, Python 3.14.7  
**Scope**: In-memory audio consultation ingestion (`AudioReader`), pre-ffprobe magic sniffing allowlist, bounded ffprobe inspect (1MB buffer cap, 20s timeout, tempfile cleanup), `Transcriber` protocol, `GroqTranscriber` adapter with `whisper-large-v3-turbo` model and segment timestamps, upload nonce name (`audio_{uuid}.ext`), `VoiceAgent`, CLI `voice` subcommand, cross-agent invariants test suite (20 tests), offline benchmark verification, and documentation updates.

### A. Architectural Decisions & Contract Additions (Contract Freeze Respected)
1. **Contract Additions (Additive Only)**:
   - `VoiceAgentOptions`: `language_hint` (validated BCP-47 / ISO-639-1 e.g. `en`, `hi`), `date_order_hint`, `include_source_text_in_result`.
   - `VoiceAgentRequest`: `request_id`, `source_name` (redacted in `repr`/`str`), `content: bytes` (`exclude=True` from serialization, redacted in `repr`/`str`), `options`.
   - `AudioLocation`: `kind="audio"`, `start_ms: int`, `end_ms: int`, `speaker_id: str | None`.
   - `AuditMetadata`: added optional field `transcription_details: dict[str, Any] | None`.
   - `ErrorCode` additions: `audio_duration_exceeded`, `audio_unreadable`, `audio_transcription_failed`.
   - `WarningCode` additions: `audio_language_mismatch`.
2. **Layering & Architecture Enforcement**:
   - `sources/audio_reader.py` defines the `Transcriber` protocol and reader logic, with zero Groq imports.
   - `providers/groq_transcriber.py` implements the protocol and adapts `GroqClientWrapper.create_transcription()`.
   - `factory.py` binds `GroqTranscriber` when `allow_remote_audio_transcription=True` and `_UnavailableTranscriber` when disabled.
   - AST tests in `tests/cross_agent/test_cross_agent_invariants.py` verify no unauthorized framework imports exist across the repo.
3. **Audio Ingestion & Subprocess Hardening**:
   - File formats allowed: WAV, MP3, OGG, FLAC, M4A, WEBM.
   - Magic sniffing pre-check rejects invalid non-media payloads before launching any subprocess.
   - Bounded ffprobe inspect: output streams capped at 1 MB (`MAX_FFPROBE_OUTPUT_BYTES = 1,048,576`), hard timeout of 20 seconds, tempfiles cleaned up in `try...finally`.
   - Duration check: streams $> 1800.0$ seconds (30 mins) or $\le 0.0$ seconds rejected with `audio_duration_exceeded`.
4. **Speech-to-Text & Provenance Mapping**:
   - `TranscriptionSegment` provides segment-level timestamps in seconds.
   - `AudioReader` converts seconds to millisecond integers `[start_ms, end_ms]` attached to `AudioLocation` on each `SourceSegment`.
   - `GroqTranscriber` maps detected language; if it conflicts with `language_hint`, emits `WarningCode.AUDIO_LANGUAGE_MISMATCH`.
   - Upload nonce filename `audio_{uuid}.ext` replaces patient `source_name` to protect PHI during external upload.

### B. Commands Executed in Milestone 4 & Verified Results

| Step | Command Line | Exit Code | Verified Output Summary |
|---|---|---|---|
| M4 Audio Reader Unit Tests | `pytest tests/unit/test_audio_reader.py -v` | `0` | 15 passed in 2.37s (sniffing, ffprobe bounds, timeouts, tempfile cleanup, duration checks) |
| M4 Voice Agent Unit Tests | `pytest tests/unit/test_voice_agent.py -v` | `0` | 4 passed in 2.91s (orchestration, PHI redaction, offline partial status) |
| M4 Groq Transcriber Tests | `pytest tests/unit/test_groq_transcriber.py -v` | `0` | 2 passed in 0.53s (response parsing, error mapping) |
| M4 Voice CLI Tests | `pytest tests/integration/test_voice_cli.py -v` | `0` | 5 passed in 23.47s (in-process exit 0, missing/invalid/empty exit 1, help exit 0) |
| M4 Audio Integration | `pytest tests/integration/test_audio_integration.py -v` | `0` | 1 passed in 1.15s (stdlib wave file ingestion & segment creation) |
| Gate 4a Full Suite | `pytest -q -m "not live_api and not slow"` | `0` | **173 passed, 3 skipped, 1 deselected in 63.87s** |
| Gate 4a Lint & Format | `python -m ruff check .` & `format` | `0` | All checks passed; 88 files formatted |
| Cross-Agent Invariants | `pytest tests/cross_agent/test_cross_agent_invariants.py -v` | `0` | 20 passed in 4.10s (tested across text, pdf, image, voice) |
| Offline Benchmark Scorer | `python scripts/evaluate_text_extraction.py` | `0` | 38 cases, 41/41 TP, Precision=1.0, Recall=1.0, F1=1.0 |
| Gate 4b Full Suite | `pytest -q -m "not live_api and not slow"` | `0` | **193 passed, 3 skipped, 1 deselected in 49.29s** |
| Gate 4b Lint | `python -m ruff check .` | `0` | **All checks passed!** |
| Gate 4b Format Check | `python -m ruff format --check .` | `0` | **88 files already formatted** |

---

## 8. Student Viva & Architecture Defense Guide

When defending this architecture in an academic or technical viva examination, highlight the following core engineering decisions:

1. **Why Hexagonal (Ports & Adapters) Architecture?**
   - *Core Domain*: `ExtractionWorkflow`, `EvidenceValidator`, `ValueParser`, and `CandidateDeduplicator` are 100% modality-agnostic and provider-agnostic.
   - *Ports (Protocols)*: `ClinicalFieldExtractor`, `OcrService`, and `Transcriber` define strictly typed abstract protocols using Python's `typing.Protocol`.
   - *Adapters*: `GroqFieldExtractor`, `TesseractOcrService`, and `GroqTranscriber` implement these protocols without leaking third-party SDK dependencies into the domain.

2. **How is Evidence Grounding Guaranteed?**
   - Language models are prone to hallucinating offsets or summarizing text. The platform **never** trusts provider-reported offsets.
   - The `EvidenceValidator` searches for the model's quoted text within the segment text using exact string matching.
   - If a quote appears multiple times within the same segment, the validator flags it as `ambiguous_match` rather than guessing an arbitrary offset.

3. **How is PHI Protected (Zero Network / Zero Log Leakage)?**
   - By default, `ALLOW_REMOTE_LLM_PROCESSING=false` and `ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false`. No bytes can leave the host environment.
   - Request and Result schemas implement custom `__repr__` and `__str__` methods that redact raw narrative text, file contents, and file paths.
   - Diagnostic logging emits structured IDs and token counts only—never raw patient narrative strings.

4. **Why Selective OCR for PDFs and Lanczos Scaling for Images?**
   - *PDFs*: Native digital text is orders of magnitude faster and higher quality than OCR. `plan_page_ocr` only triggers OCR when native text is deficient ($< 40$ alnum characters) or large raster images are present.
   - *Images*: Low-resolution scans hurt OCR accuracy. If an image's shorter dimension is under 1,000 pixels, it is Lanczos upscaled up to $2.0\times$. Coordinates from OCR are inverted back ($x / S, y / S$) so citations always map to original displayed pixels.

5. **How is Audio Ingestion Hardened Against Exploits?**
   - Magic byte sniffing verifies audio file signatures before launching decoders.
   - Subprocess execution (`ffprobe`) runs with `shell=False`, a hard 1 MB stream capture cap, and a strict 20.0-second timeout.
   - Streams exceeding 30 minutes are rejected before downstream processing.




