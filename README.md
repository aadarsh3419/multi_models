# Healthcare Multi-Agent AI Platform — Text, PDF, Image & Voice Extraction Foundation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: 193 passed](https://img.shields.io/badge/tests-193%20passed%2C%203%20skipped-brightgreen.svg)]()
[![Audit: Verified](https://img.shields.io/badge/audit-verified-success.svg)](docs/audit-report.md)
[![Localhost Web UI](https://img.shields.io/badge/localhost-web%20UI-blueviolet.svg)]()

> **Milestones 1, 2, 3 & 4 (Complete)**: A safe, evidence-grounded, auditable multi-agent platform for clinical text, PDF, image, and voice consultation extraction with modular hexagonal architecture, spatial & temporal provenance, Pillow safety defenses, bounded ffprobe audio validation, selective Tesseract OCR, strict JSON schemas, 20 cross-agent invariants, and zero-PHI leak guarantees.

---

## 1. Overview & Multi-Modal Pipeline

This platform implements a production-grade multi-agent extraction foundation capable of ingesting clinical data across four distinct modalities: **Unstructured Text**, **PDF Documents**, **Document Images**, and **Voice Audio Dictations**. Every candidate fact extracted is mathematically traceable back to source coordinates (character offsets, PDF page bounding boxes, image pixel regions, or audio timestamps).

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  Text Agent  │     │  PDF Agent   │     │ Image Agent  │     │ Voice Agent  │
 │ (M1: Text /  │     │(M2: PyMuPDF /│     │ (M3: Pillow /│     │(M4: ffprobe /│
 │  EHR Notes)  │     │  Tables/OCR) │     │  Safety/ROI) │     │ Whisper STT) │
 └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
        │                    │                    │                    │
        │ [TextLocation]     │ [PdfLocation]      │ [ImageLocation]    │ [AudioLocation]
        │ [start:end char]   │ [page, bbox]       │ [pixel bbox]       │ [start:end ms]
        ▼                    ▼                    ▼                    ▼
   SourceTextBundle     SourceTextBundle     SourceTextBundle     SourceTextBundle
        │                    │                    │                    │
        └────────────────────┼────────────────────┼────────────────────┘
                             ▼
               [Bounded Sequential Chunker]
                             ▼
                [ClinicalFieldExtractor]
               (Groq Cloud or Local Double)
                             ▼
                 [Pydantic v2 Validation]
                             ▼
                    [EvidenceValidator]
           (Independent quote & coordinate mapping)
                             ▼
                   [Deterministic Parsers]
                (ValueParser & DateParser)
                             ▼
                 [CandidateDeduplicator]
             (Exact deduplication & conflicts)
                             ▼
                    [Common AgentResult]
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  Localhost Web UI       Local CLI        Evaluation Scorer
   (Browser :8000)      (Terminal)       (38 Benchmark Cases)
```

### Core Engineering Invariants
1. **Source Preservation**: Raw source text is never normalized, mutated, or truncated during segmentation. Offsets represent zero-based Python string indices counting Unicode code points (`len(str)`).
2. **Independent Evidence Grounding**: LLMs are never trusted to calculate source coordinates. The pipeline matches model quotations against exact source segments. Duplicate quotes inside a segment are explicitly flagged as `ambiguous_match` rather than picking one arbitrarily.
3. **Multi-Modal Spatial & Temporal Provenance**:
   - **Text**: `TextLocation` with character spans `[start_char, end_char]`.
   - **PDF**: `PdfLocation` with page number and 2D bounding box `[x0, y0, x1, y1]`.
   - **Image**: `ImageLocation` with displayed pixel bounding box and OCR word confidences.
   - **Voice**: `AudioLocation` with millisecond timestamps `[start_ms, end_ms]` and speaker diarization tags.
4. **Distinction of Experiencers**: Mentions of family members (e.g. *"My father had diabetes"*) are preserved with `experiencer_type="family_member"` and `experiencer_relationship_raw="father"`, never recorded as the patient's condition.
5. **Deterministic Candidate IDs**: Generated via SHA-256 hashes of source content digests, sorted evidence span coordinates, and normalized semantic fields.
6. **Zero PHI Leaks in Logs**: Model representations (`__repr__` and `__str__`) redact raw clinical text and file paths. Diagnostic logs are piped to `stderr`, while structured result JSON is piped to `stdout`.

---

## 2. Environment Setup

### Prerequisites
- Python 3.11 or higher (Tested and verified on Python 3.14 on Windows 11).
- `git`

### Installation (Windows PowerShell)

```powershell
# Navigate to the workspace
cd "c:\aadarshtiwari\c coding\multi_models"

# Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dev dependencies in editable mode
pip install -e ".[dev]"
```

### Configuration (.env)

Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Key configuration variables:
- `GROQ_API_KEY`: API key for Groq Cloud (optional for local/offline testing).
- `ALLOW_REMOTE_LLM_PROCESSING`: Safety gate (`false` by default). Must be explicitly set to `true` to allow outbound Groq API calls.
- `ALLOW_REMOTE_AUDIO_TRANSCRIPTION`: Safety gate (`false` by default) for audio speech-to-text.
- `DATE_ORDER`: Strategy for ambiguous dates like `03/04/2025` (`UNKNOWN`, `DMY`, or `MDY`).
- `MAX_TEXT_CHARS`: Safety upper bound per text request (default `36000`).
- `MAX_AUDIO_BYTES`: Upper bound for audio files (default `25 MB`).

---

## 3. External System Binaries (Optional for Production; Not Run in this Offline Session)

During automated development and testing in this session, the test suite utilizes deterministic test doubles (`FakeOcrService` and `_UnavailableTranscriber`), allowing 100% of unit and integration tests to pass without external binaries. For production deployment with live OCR and audio transcoding:

### A. Tesseract OCR (Scanned PDF & Image OCR)
- **Windows**: Install [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) or run `winget install UB-Mannheim.TesseractOCR`. Add to `PATH` or configure `TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`.
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-hin`
- **macOS**: `brew install tesseract tesseract-lang`

### B. FFmpeg / ffprobe (Voice Audio Inspection)
- **Windows**: Install via `winget install Gyan.FFmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your system `PATH`.
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

---

## 4. Run on Localhost Web UI (Browser Interface)

For interactive visual testing, a zero-dependency web server is included in `scripts/serve_local.py`:

```powershell
python scripts/serve_local.py
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### Web UI Features:
- **Interactive Workbench**: Enter or paste custom clinical text in English, Hindi, or Hinglish.
- **Preset Scenarios**: Chronic disease, family history negation, Devanagari Hindi clinical notes, and prompt injection defense.
- **Live Evidence Highlighting**: Hover over any extracted fact card on the right — the exact character span `[start:end]` in the raw text on the left highlights in real time!
- **Audit & Provenance**: Displays Request ID, execution duration, and SHA-256 content hashes.

---

## 5. CLI Usage Across All 4 Modalities

The package provides a unified command-line interface via `python -m healthcare_agents.cli`:

### A. Text Agent
```powershell
# Extract clinical facts from an inline string
python -m healthcare_agents.cli text --text "Patient seen today. Fasting blood glucose 110 mg/dL."

# Extract clinical facts from a file
python -m healthcare_agents.cli text --file sample_clinical_note.txt

# Extract via standard input
"Patient presented with headache. BP 130/85 mmHg." | python -m healthcare_agents.cli text --stdin
```

### B. PDF Agent
```powershell
# Auto-detect native text vs scanned raster regions
python -m healthcare_agents.cli pdf --file discharge_summary.pdf

# Enforce OCR mode policy ('auto', 'never', 'always')
python -m healthcare_agents.cli pdf --file scan.pdf --ocr-mode always --ocr-languages eng

# Support bilingual OCR (English + Hindi)
python -m healthcare_agents.cli pdf --file prescription.pdf --ocr-languages eng+hin
```

### C. Image Agent
```powershell
# Extract clinical facts from photo of a lab slip or paper prescription
python -m healthcare_agents.cli image --file lab_slip.png

# Support bilingual OCR
python -m healthcare_agents.cli image --file rx_slip.jpg --ocr-languages eng+hin

# Disable automatic Lanczos upscaling if desired
python -m healthcare_agents.cli image --file high_res_scan.png --no-upscale
```

### D. Voice Agent
```powershell
# Extract facts from recorded clinical consultation (WAV, MP3, OGG, FLAC, M4A, WEBM)
python -m healthcare_agents.cli voice --file consultation.wav

# Specify optional language hint (e.g. Hindi 'hi' or English 'en')
python -m healthcare_agents.cli voice --file dictation.mp3 --lang en

# View CLI help options
python -m healthcare_agents.cli voice --help
```

### CLI Exit Codes
- `0`: Success (all stages executed successfully).
- `2`: Partial result (input accepted, but provider extraction skipped due to offline safety gate `ALLOW_REMOTE_LLM_PROCESSING=false` or `ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false`).
- `1`: Processing failure or invalid input (e.g. file not found, empty text, unsupported format, or corrupt bytes).

---

## 6. Running Tests & Code Quality

All **193 unit, integration, and cross-agent invariant tests** run entirely **offline** without external network access, Groq API keys, or external system binaries:

### Run Complete Test Suite
```powershell
python -m pytest -q -m "not live_api and not slow"
```

### Run Ruff Code Quality & Formatting Checks
```powershell
python -m ruff check .
python -m ruff format --check .
```

### Run Cross-Agent Invariant Tests Only
```powershell
python -m pytest tests/cross_agent/test_cross_agent_invariants.py -v
```

---

## 7. Synthetic Evaluation Benchmark

The benchmark suite contains 38 synthetic test cases covering 14 clinical categories, including measurements, reference ranges, negation, family history, ambiguous dates, prompt injection, Hindi, and Hinglish narratives.

Run the evaluation scorer:
```powershell
python scripts/evaluate_text_extraction.py
```

### Benchmark Metrics Summary:
- **Fact Extraction Precision**: `1.0000` (41 / 41 TP)
- **Fact Extraction Recall**: `1.0000` (41 / 41 TP)
- **Overall F1 Score**: `1.0000`
- **Grounded Evidence TP**: `40` | **Ambiguous Duplicate Quote Handled**: `1` (Case 032 correctly flagged as `ambiguous_match`).
- **Prompt Injection Defense**: Immunity verified (produces 0 false positives).

---

## 8. Project Structure

```
multi_models/
├── .env.example              # Sample environment configuration
├── AGENTS.md                 # Agent contracts & multimodal architecture
├── README.md                 # Master documentation
├── PROGRESS.md               # Session progress, commands executed & viva log
├── pyproject.toml            # Build configuration, dependencies & Ruff settings
├── docs/                     # Detailed architectural documentation
│   ├── architecture.md       # Hexagonal architecture, AST boundaries & modality specs
│   ├── audit-report.md       # In-place code audit & verification report
│   ├── data-contracts.md     # Pydantic v2 schemas, provenance models & semantic enums
│   ├── evaluation.md         # Scorer methodology, manifest seal & matching rules
│   ├── integration-plan.md   # FastAPI gateway, RBAC, MongoDB, FHIR R4 & ML models
│   └── privacy-and-limitations.md # DPDP Act alignment, audio bounds & medical limits
├── src/                      # Modular source tree (src-layout)
│   └── healthcare_agents/
│       ├── config.py         # pydantic-settings configuration across all 4 modalities
│       ├── factory.py        # Composition root (build_text_agent, build_pdf_agent, build_image_agent, build_voice_agent)
│       ├── cli.py            # Unified CLI (text, pdf, image, voice subcommands)
│       ├── evaluation.py     # Deterministic evaluation & bipartite scorer
│       ├── contracts/        # Domain schemas, location models, enums & errors
│       │   ├── requests.py   # TextAgentRequest, PDFAgentRequest, ImageAgentRequest, VoiceAgentRequest
│       │   ├── locations.py  # TextLocation, PdfLocation, ImageLocation, AudioLocation
│       │   ├── source_text.py# SourceTextBundle & SourceSegment
│       │   ├── facts.py      # FactCandidate, ProviderCandidate, enums & schemas
│       │   ├── parsing.py    # ValueParseResult & ParseStatus
│       │   ├── results.py    # AgentResult, StageResult & AuditMetadata
│       │   └── errors.py     # ErrorCode, WarningCode & StructuredError
│       ├── agents/           # Modality Agent implementations
│       │   ├── base.py       # BaseAgent abstract interface
│       │   ├── text_agent.py # TextAgent (lossless segmentation)
│       │   ├── pdf_agent.py  # PDFAgent (PyMuPDF & selective OCR)
│       │   ├── image_agent.py# ImageAgent (Pillow safety & coordinate mapping)
│       │   └── voice_agent.py# VoiceAgent (audio validation & Whisper STT)
│       ├── ocr/              # Shared, image-agnostic OCR subsystem
│       │   └── service.py    # OcrService protocol, TesseractOcrService & FakeOcrService
│       ├── sources/          # Source readers & parsers
│       │   ├── text_reader.py    # Unicode code-point indexing & bundle creation
│       │   ├── pdf_reader.py    # In-memory PDF parser, layout, tables & IoU dedup
│       │   ├── image_reader.py  # Pillow safety hardening, EXIF transpose & Lanczos upscale
│       │   ├── audio_reader.py  # Magic byte sniffing, bounded ffprobe inspect & Transcriber protocol
│       │   ├── normalization.py # NFC canonicalization & index map tracking
│       │   └── chunker.py    # Table-aware lossless sequential chunker
│       ├── extraction/       # Modality-agnostic extraction pipeline
│       │   ├── field_extractor.py   # ClinicalFieldExtractor protocol
│       │   ├── evidence_validator.py# Independent quote grounding & verification
│       │   ├── value_parser.py      # Deterministic ValueParser & DateParser
│       │   ├── deduplication.py     # Exact deduplication & conflict detection
│       │   └── workflow.py          # Synchronous ExtractionWorkflow engine
│       └── providers/        # External LLM / STT provider adapters
│           ├── groq_client.py       # GroqClientWrapper with rate-limit retries & audio upload
│           ├── groq_field_extractor.py # Official Groq SDK structured extractor
│           └── groq_transcriber.py  # Groq Whisper audio transcriber adapter
└── tests/                    # Complete test suite (193 offline tests)
    ├── cross_agent/          # Cross-agent invariant verification suite
    │   └── test_cross_agent_invariants.py # 20 parametrized invariant tests across all 4 agents
    ├── evaluation/           # Evaluation splits & cryptographic manifest
    │   ├── dev_cases.json    # 24 development evaluation cases
    │   ├── heldout_cases.json# 14 held-out evaluation cases
    │   └── heldout_manifest.json # Cryptographic SHA-256 seal
    ├── integration/          # CLI and provider integration tests
    │   ├── test_cli.py       # Text CLI tests
    │   ├── test_pdf_cli.py   # PDF CLI tests
    │   ├── test_image_cli.py # Image CLI tests
    │   ├── test_voice_cli.py # Voice CLI tests
    │   ├── test_ocr_integration.py # Tesseract integration tests
    │   ├── test_audio_integration.py # Wave audio integration tests
    │   ├── test_groq_adapter.py # Provider wrapper tests
    │   └── test_groq_extractor.py # Extractor mock & rate-limit tests
    └── unit/                 # Unit test suite across all modules
        ├── test_architecture.py # AST dependency boundary & layering tests
        ├── test_contracts.py    # Multi-modal contracts & PHI redaction
        ├── test_sources.py      # Lossless text reading & chunking
        ├── test_ocr_service.py  # OCR service protocol & test double
        ├── test_pdf_reader.py   # PDF parsing, tables & IoU dedup
        ├── test_pdf_agent.py    # PDFAgent orchestration & provenance
        ├── test_image_reader.py # Pillow safety & coordinate mapping
        ├── test_image_agent.py  # ImageAgent orchestration & provenance
        ├── test_audio_reader.py # Audio sniffing, ffprobe bounds & timestamps
        ├── test_voice_agent.py  # VoiceAgent orchestration & provenance
        ├── test_groq_transcriber.py # GroqTranscriber adapter tests
        ├── test_extraction.py   # Workflow, validation, parsing & dedup
        ├── test_text_agent.py   # TextAgent orchestration & limits
        ├── test_evaluation.py   # Bipartite scorer & manifest integrity
        ├── test_gate_a.py       # Gate A contracts & parsers
        └── test_gate_b.py       # Gate B chunking & workflow bounds
```
