# Privacy, Security, and Clinical Limitations Specification

**Document Version**: 4.0.0  
**Platform Scope**: Milestones 1–4 (Text, PDF, Image, and Voice Extraction Foundation)  
**Status**: Authoritative Reference  

---

## 1. Information Extraction Scope & Medical Disclaimers

### A. Non-Diagnostic Information Extraction Only
The Healthcare Multi-Agent AI Platform is strictly an **information extraction engine**. Its purpose is to parse unstructured or semi-structured clinical narratives, document layouts, scanned forms, and audio dictations into structured fact candidate records (`FactCandidate`).

> [!WARNING]
> **NOT A MEDICAL DEVICE OR DIAGNOSTIC TOOL**  
> 1. This software does **not** provide clinical diagnosis, clinical decision support, medical advice, drug interaction checking, or treatment recommendations.
> 2. All outputs emitted by any agent (`TextAgent`, `PDFAgent`, `ImageAgent`, `VoiceAgent`) are **candidates** (`FactCandidate` records with `ReviewStatus.UNREVIEWED`), never verified clinical data.
> 3. Extracted candidates must be reviewed, corroborated, and approved by qualified, licensed human healthcare practitioners before being used in any patient care workflow.
> 4. Performance on automated regression benchmarks does not equate to clinical efficacy or diagnostic accuracy on real patient populations.

### B. Unit & Value Neutrality
The `ValueParser` and `DateParser` engines perform **pure syntactic parsing**:
- Numerical scalars, blood pressure ratios, and ranges are extracted into standard numeric representations (`Decimal`).
- The system **never** performs automatic unit conversions (e.g., converting mg/dL to mmol/L or Fahrenheit to Celsius) or clinical evaluations (e.g., labeling a value "critical" or "stage 2 hypertension").
- No clinical judgments, triage algorithms, or diagnostic inference models are executed autonomously.

---

## 2. Protected Health Information (PHI) & Data Privacy Guarantees

### A. Untouched Source Text Isolation
1. **Zero Text Retention in Outputs by Default**:
   All agent requests and `AgentResult` enforce strict isolation. By default (`options.include_source_text_in_result = False`), the original source narrative or transcribed speech is discarded from the returned JSON payload. Only cryptographic digests (`content_sha256`) and sanitized metadata are preserved.
2. **Safe String Representation (`__repr__` and `__str__`)**:
   `TextAgentRequest`, `PDFAgentRequest`, `ImageAgentRequest`, and `VoiceAgentRequest` implement custom redaction methods:
   - Raw clinical text is redacted: `[REDACTED text_len=N sha256=...]`.
   - File bytes and media streams are excluded from string formatting and serialization.
   - `source_name` (e.g. `patient_john_doe_ecg.pdf`) is redacted in `__repr__` and `__str__` to prevent PHI leakage into logs or terminal crash dumps.
3. **Structured Diagnostic Logging**:
   All log statements within readers, chunkers, validators, workflow, and provider adapters emit static messages or non-PHI identifiers (e.g. `segment_id`, `unit_id`, `error_code`, byte lengths). Patient names, narrative text, quotes, and transcriptions are strictly barred from logger format strings.

### B. Offline Default Gate & Zero Network Leakage
1. **`ALLOW_REMOTE_LLM_PROCESSING=false` & `ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false`**:
   The default platform configuration strictly disables all outbound remote network calls across all modalities.
2. If remote processing is disabled, the system completes local ingestion and parsing stages cleanly, emitting non-fatal diagnostics (`remote_processing_disabled`) and returning `status="partial"` without transmitting any bytes across the network.
3. Live Groq API calls require explicit, opt-in configuration (`ALLOW_REMOTE_LLM_PROCESSING=true` / `ALLOW_REMOTE_AUDIO_TRANSCRIPTION=true`) and valid API keys supplied via environment variables.

### C. Groq API Provider Boundary
When remote LLM processing is explicitly enabled:
1. **Inert JSON Transmission**: The source text is framed strictly as an inert, quoted JSON string within the `user` turn (`{"units": [{"unit_id": ..., "text": ...}]}`).
2. **Prompt Injection Hardening**: System instructions mandate treating narrative content as untrusted patient text, discarding any embedded command overrides or role reversals.
3. **Strict Schema Constraints**: Provider output is constrained via `strict_json_schema` with `additionalProperties: false`. Hallucinated keys or out-of-schema attributes are rejected immediately at validation time.

---

## 3. Regulatory Alignment (India DPDP Act & HIPAA)

### A. Digital Personal Data Protection (DPDP) Act 2023 Alignment
> [!NOTE]
> *(User-supplied, unverified legal reference)*: The Digital Personal Data Protection Act 2023 (DPDP Act) was enacted by the Parliament of India in August 2023. Notice is hereby given that the enforcement of various provisions of the Act occurs through staggered government notifications, and subordinate implementation rules are subject to ongoing public consultation. This document provides architectural design alignment only and does NOT constitute formal legal advice or certification.

Architectural safeguards designed to facilitate DPDP compliance:
1. **Purpose Limitation**: Patient health data ingested into the agents is processed solely for the specific clinical entity extraction requested by the invoking clinician or authorized system.
2. **Data Minimization**: Auditing captures only cryptographic digests (`content_sha256`, `prompt_sha256`), execution durations, and non-PHI counts—never raw patient narratives or recordings.
3. **Storage Limitation & Ephemeral Processing**: Core pipeline execution occurs strictly in-memory. Ingested bytes and temporary files are immediately deallocated upon completion.
4. **Data Fiduciary Governance**: Cleanly separated role-based contracts ensure clear segregation of duties between clinicians, auditors, and system administrators.

### B. Academic Prototype Disclaimer
> [!IMPORTANT]
> This platform is an educational and academic research prototype developed as part of a multi-agent engineering project. It has NOT been certified under US HIPAA Safe Harbor, European GDPR, Indian DPDP Act 2023, or Central Drugs Standard Control Organisation (CDSCO) Medical Device Rules 2017. Clinical use on real patients without prior institutional ethics board (IRB) approval is strictly prohibited.

---

## 4. Engineering Limitations & Concurrency Assumptions

1. **Synchronous Execution Model**:
   - `agent.process(request) -> AgentResult` is a **blocking, synchronous call** across all agents.
   - When integrating into web services (e.g. FastAPI), dispatch agent invocations to thread pools (`asyncio.to_thread`) to prevent blocking the async event loop.
2. **Deterministic Regression Benchmark**:
   - Benchmark metrics reported in `docs/evaluation.md` measure algorithmic regression against synthetic test cases.
   - They do **not** represent clinical performance on real-world electronic health record (EHR) data.

---

## 5. Milestone 2: PDF Ingestion & OCR Limitations

### A. PyMuPDF Dual Licensing Warning
- PyMuPDF (`pymupdf`) is dual-licensed under **GNU AGPL-3.0** (or commercial license from Artifex Software).
- Any organization deploying, distributing, or offering this software as a network service must evaluate their licensing compliance before deployment.

### B. In-Process PDF Parsing & Resource Boundaries
- `MAX_PDF_BYTES` (default: 25 MB).
- `MAX_PDF_PAGES` (default: 50 pages). Documents exceeding this limit are rejected before page processing.
- Documents are opened strictly in memory (`stream=content, filetype="pdf"`); no temporary PDF files are written to disk.

### C. Selective OCR & Render Pixel Budgeting
- Stepwise lowers rendering DPI (200 $\to$ 150 $\to$ 100 DPI) if a page exceeds `MAX_RENDER_PIXELS` (default: 20,000,000 pixels). If the budget is still exceeded, OCR is cleanly skipped with warning `ocr_skipped_render_budget` to prevent Out-Of-Memory (OOM) crashes.
- OCR execution enforces a strict per-call timeout (`OCR_TIMEOUT_SECONDS=60`) and request cap (`MAX_OCR_REGIONS_PER_REQUEST=60`).

---

## 6. Milestone 3: Image Agent Safety Boundaries & Pillow Protections

### A. Non-Diagnostic Document OCR Scope
- Engineered exclusively to read printed alphanumeric text from document images (e.g. paper lab reports, printed prescription slips).
- Not a diagnostic computer-vision or radiology engine.

### B. Pillow Safety & Decompression Bomb Defenses
- Files exceeding `MAX_IMAGE_BYTES=15MB` are rejected before parsing.
- Format allowlist: `PNG`, `JPEG`, `WEBP`, `BMP`, and `TIFF`. Animated formats (`GIF`, animated `WEBP`, multi-frame `TIFF`) are rejected with `unsupported_file_type`.
- Image dimensions are inspected prior to `img.load()`. Files exceeding `MAX_IMAGE_PIXELS=30,000,000` are rejected with `image_pixel_limit_exceeded`.
- Pillow's `DecompressionBombWarning` is treated as a fatal error via `warnings.catch_warnings()`.

---

## 7. Milestone 4: Voice Agent Safety Boundaries & Audio Ingestion Hardening

### A. Pre-ffprobe Magic Byte Sniffing
To prevent passing arbitrary shell commands, corrupted streams, or malicious payloads to media decoders, `AudioReader` checks initial file bytes against a strict magic number allowlist:
- **WAV**: `RIFF....WAVE`
- **MP3**: `ID3` tag or sync word `0xFF 0xFB / 0xFA / 0xF3 / 0xF2`
- **OGG**: `OggS`
- **FLAC**: `fLaC`
- **M4A / MP4**: `....ftyp`
- **WEBM / MKV**: `0x1A 0x45 0xDF 0xA3` (EBML header)

Payloads failing magic sniffing are rejected immediately with `ErrorCode.UNSUPPORTED_FILE_TYPE` before launching any subprocess.

### B. Bounded Subprocess Isolation
When invoking `ffprobe` to inspect media streams:
- Subprocess execution runs with `shell=False` passing explicit argument arrays.
- Hard output stream limit: standard output and standard error reads are capped at **1 MB** (`MAX_FFPROBE_OUTPUT_BYTES = 1,048,576`) to prevent memory exhaustion from verbose container metadata.
- Execution timeout: Hard timeout of **20.0 seconds** (`FFPROBE_TIMEOUT_SECONDS = 20.0`). Processes exceeding this timeout are forcefully terminated (`proc.kill()`).
- File lifecycle: Ephemeral temporary files are cleaned up reliably within `try...finally` blocks.

### C. Audio Bounds & Duration Constraints
- **File Size**: Files exceeding `MAX_AUDIO_BYTES = 25,000,000` (25 MB) are rejected immediately.
- **Duration**: Audio streams with duration $\le 0.0$ or $> \text{MAX\_AUDIO\_SECONDS} = 1800.0$ (30 minutes) are rejected with `ErrorCode.AUDIO_DURATION_EXCEEDED`.

### D. Audio Transcription Privacy & Filename Redaction
- Audio transcription requires explicit opt-in (`ALLOW_REMOTE_AUDIO_TRANSCRIPTION=true`).
- When uploading to external providers (e.g. Groq Whisper), the original patient/hospital file name (`source_name`) is discarded and replaced with a cryptographically generated nonce (`audio_{uuid}.ext`).
- Raw audio streams are never stored in log records, error messages, or interactive shells.
