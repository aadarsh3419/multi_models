# Healthcare Multi-Agent AI Platform — Text Extraction Foundation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: 46 passed](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)]()
[![Localhost Web UI](https://img.shields.io/badge/localhost-web%20UI-blueviolet.svg)]()

> **Milestone 1**: A safe, evidence-grounded, auditable vertical slice for clinical text extraction with live browser UI, local CLI, and offline evaluation benchmark.

---

## 1. Overview & Architectural Pipeline

This repository implements the first vertical slice of the Healthcare Multi-Agent AI Platform. It provides a robust **Text Agent** that extracts structured clinical facts from raw clinical narratives (supporting English, Hindi, and Hinglish) while guaranteeing mathematical traceability back to original, un-normalized source text coordinates.

```
Clinical Text Input (Narrative / EHR / Lab Notes)
                      │
                      ▼
                 Text Agent
                      │  (Lossless paragraph/line segmentation & SHA-256 digest)
                      ▼
               SourceTextBundle
                      │
                      ▼
             Bounded TextChunker (Sequential 6,000 char chunks)
                      │
                      ▼
            ClinicalFieldExtractor
           (Groq Cloud or Local Double)
                      │  (Strict JSON Schema Structured Output)
                      ▼
            Pydantic v2 Validation
                      │
                      ▼
               EvidenceValidator
         (Verbatim quote verification & coordinate mapping)
                      │
                      ▼
            Deterministic Parsers
         (ValueParser & DateParser)
                      │
                      ▼
            CandidateDeduplicator
         (Exact deduplication & conflict detection)
                      │
                      ▼
              Common AgentResult
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
 Localhost Web UI  Local CLI   Evaluation Script
 (Browser :8000)   (Terminal)   (38 Gold Cases)
```

### Core Engineering Invariants
1. **Source Preservation**: Raw source text is never normalized, mutated, or truncated during segmentation. Offsets represent zero-based Python string indices counting Unicode code points (`len(str)`).
2. **Evidence Grounding**: LLMs are never trusted to calculate source coordinates. The pipeline matches model quotations against exact source segments. Duplicate quotes inside a segment are explicitly flagged as `ambiguous_match` rather than picking one arbitrarily.
3. **No Unwarranted Inferences**: An observation like `120/80` is parsed as blood pressure **only** if the context explicitly matches an approved allowlist. Without explicit context, it is treated as a generic ratio.
4. **Distinction of Subjects**: Mentions of family members (e.g. *"My father had diabetes"*) are preserved with `experiencer_type="family_member"` and `experiencer_relationship_raw="father"`, never recorded as the patient's condition.
5. **Deterministic Candidate IDs**: Generated via SHA-256 hashes of source content digests, sorted evidence span coordinates, and normalized semantic fields.
6. **Zero PHI Leaks in Logs**: Model representations (`__repr__` and `__str__`) hide raw clinical text. Diagnostic logs are piped to `stderr`, while structured result JSON is piped to `stdout`.

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

### Installation (POSIX / Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
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
- `DATE_ORDER`: Strategy for ambiguous dates like `03/04/2025` (`UNKNOWN`, `DMY`, or `MDY`). Default is `UNKNOWN`.
- `MAX_TEXT_CHARS`: Safety upper bound per request (default `36000`).

---

## 3. Run on Localhost Web UI (Browser Interface)

You can run the interactive web interface locally and use it directly in your browser:

### Start the Local Server
```powershell
python scripts/serve_local.py
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### Web UI Features:
- **Interactive Workbench**: Enter or paste custom clinical text in English, Hindi, or Hinglish.
- **Preset Buttons**: Click preset buttons to immediately test:
  - *Chronic Disease Note* (BP, Blood Glucose, HbA1c, Metformin).
  - *Family History & Negation* (Father's CAD vs denied patient chest pain).
  - *Hindi Clinical Note* (रक्तचाप, बुखार, दवा विवरण).
  - *Prompt Injection Defense* (verifying immunity against injection attacks).
- **Live Evidence Highlighting**: Hover your mouse over any extracted fact card on the right — the exact character span `[start:end]` in the raw text on the left highlights in real time!
- **Audit & Provenance**: Displays Request ID, execution duration (e.g. `2.5 ms`), and SHA-256 content hashes.

---

## 4. Quickstart: Terminal Demonstration

Run the automated end-to-end demonstration and benchmark suite with a single command:
```powershell
python run.py
```

This runs:
1. End-to-end extraction on `sample_clinical_note.txt` with character coordinate verification.
2. The complete 38-case synthetic evaluation benchmark with precision, recall, and category metrics.

---

## 5. CLI Usage

The package installs a global command-line interface `healthcare-text` (or `python -m healthcare_agents.cli`).

### Extract Clinical Facts from a File
```powershell
python -m healthcare_agents.cli extract-text --input-file sample_clinical_note.txt
```

### Extract Clinical Facts via Standard Input (stdin)
```powershell
"Patient presented with severe headache. BP measured at 130/85 mmHg." | python -m healthcare_agents.cli extract-text --stdin
```

### CLI Exit Codes
- `0`: Success (all stages completed successfully).
- `1`: Processing failure (unhandled runtime error).
- `2`: Invalid input / arguments (e.g., file not found, empty text, or file exceeds 36,000 characters).
- `3`: Partial result (e.g. input accepted, but provider extraction was skipped due to safety gates or call budget).

> [!WARNING]
> Result JSON emitted to `stdout` contains extracted quotations and clinical entities. Handle output JSON as sensitive Protected Health Information (PHI).

---

## 6. Running Tests & Code Quality

All 46 unit and integration tests run entirely **offline** without external network access or a Groq API key.

### Run All Unit and Integration Tests
```powershell
python -m pytest tests/ -v
```

### Run Ruff Code Quality & Formatting Checks
```powershell
python -m ruff check .
python -m ruff format --check .
```

### Optional: Run Live Groq API Tests
To execute live integration tests against the actual Groq Cloud endpoint:
```powershell
$env:RUN_LIVE_API_TESTS = "true"
$env:GROQ_API_KEY = "gsk_your_actual_key"
$env:ALLOW_REMOTE_LLM_PROCESSING = "true"
python -m pytest tests/integration/test_groq_extractor.py -m live_api -v
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
- **Grounded Evidence TP**: `40` | **Ambiguous Duplicate Quote Handled**: `1` (Case 032 contains duplicate quotes `"120/80"` in the same segment; correctly flagged as `ambiguous_match` rather than guessing).
- **Prompt Injection Defense**: Immunity verified (produces 0 false positives).

> [!IMPORTANT]
> **Clinical Disclaimer**: The synthetic cases in `tests/fixtures/synthetic/labeled_text_cases.json` serve as an automated software regression benchmark. They do NOT establish real-world clinical safety or medical gold-standard accuracy. Independent human medical review is required before describing any model output as clinically validated.

---

## 8. Project Structure

```
multi_models/
├── .env.example              # Sample environment configuration
├── .gitignore                # Git exclusions (caches, envs, logs)
├── AGENTS.md                 # Agent contracts & future multimodal specs
├── README.md                 # Master documentation
├── pyproject.toml            # Build configuration, dependencies & Ruff settings
├── run.py                    # Top-level one-command test & demo runner
├── sample_clinical_note.txt  # Sample multi-entity clinical text
├── docs/                     # Detailed architectural documentation
│   ├── architecture.md       # Hexagonal architecture, AST boundaries & offsets
│   ├── data-contracts.md     # Pydantic v2 schemas & semantic enums
│   ├── evaluation.md         # Scorer methodology & matching rules
│   └── integration-plan.md   # PDF, Image, Voice, MongoDB & ML roadmaps
├── scripts/                  # Executable scripts
│   ├── demo_extraction.py    # Console extraction demonstration
│   ├── evaluate_text_extraction.py  # 38-case benchmark scorer
│   └── serve_local.py        # Localhost HTTP server & interactive Web UI
├── src/                      # Source code (src-layout)
│   └── healthcare_agents/
│       ├── __init__.py       # Package version definition
│       ├── chunker.py        # Bounded 6,000-char sequential chunker
│       ├── cli.py            # CLI composition root & arg parser
│       ├── config.py         # pydantic-settings configuration
│       ├── contracts.py      # Domain models, enums & AgentResult
│       ├── deduplicator.py   # Exact deduplication & conflict detection
│       ├── evidence_validator.py  # Quotation matcher & coordinate mapper
│       ├── extractor_interface.py # ClinicalFieldExtractor ABC interface
│       ├── groq_extractor.py # Groq structured output provider adapter
│       ├── prompt_templates.py    # Strict JSON schemas & system prompts
│       ├── source_text.py    # SourceSegment & SourceTextBundle builder
│       ├── text_agent.py     # Text Agent entrypoint & PHI safety
│       └── workflow.py       # Synchronous ExtractionWorkflow engine
└── tests/                    # Test suite
    ├── fixtures/synthetic/
    │   └── labeled_text_cases.json # 38 static synthetic gold cases
    ├── integration/
    │   ├── test_cli.py       # CLI exit codes & stdin tests
    │   └── test_groq_extractor.py # Groq adapter mocks & rate limits
    └── unit/
        ├── test_architecture.py    # AST boundary inspection tests
        ├── test_gate_a.py    # Contracts, parsers, and deduplication
        └── test_gate_b.py    # Chunker, workflow, and budget bounds
```
