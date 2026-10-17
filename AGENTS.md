# Multi-Modal Agent System Architecture & Contracts

This document outlines the agent contracts for the Healthcare Multi-Agent AI Platform across all four operational modalities: **Text**, **PDF**, **Image**, and **Voice**.

---

## 1. Unified Agent Contract (`BaseAgent`)

All agents implement the common synchronous `BaseAgent[TRequest]` interface and return a standardized `AgentResult`:

```python
class BaseAgent(ABC, Generic[TRequest]):
    @abstractmethod
    def process(self, request: TRequest) -> AgentResult:
        """Process a modality-specific request into validated, evidence-grounded FactCandidates."""
        raise NotImplementedError
```

Every agent orchestrates a modality-specific `Reader` to generate a `SourceTextBundle`, which is then passed to the shared, modality-agnostic `ExtractionWorkflow`.

---

## 2. Implemented Modality Agents

### A. Text Agent (`TextAgent`)
- **Modality**: Unstructured or semi-structured clinical narratives (English, Hindi, Hinglish).
- **Ingestion**: `TextReader` performing lossless paragraph/line segmentation and Unicode code-point indexing.
- **Contract**:
  - Request: `TextAgentRequest(text: str, options: TextAgentOptions)`
  - Provenance: `TextLocation(kind="text", start_char=int, end_char=int)`
  - Result: `AgentResult` with `agent_name="text_agent"`

### B. PDF Agent (`PDFAgent`)
- **Modality**: Native and scanned clinical documents (discharge summaries, laboratory reports, multi-column forms).
- **Ingestion**: `PdfReader` with PyMuPDF in-memory parsing, table extraction (`find_tables`), render pixel budgeting, selective OCR planning (`plan_page_ocr`), and native/OCR IoU deduplication.
- **Contract**:
  - Request: `PDFAgentRequest(content: bytes, source_name: str, options: PDFAgentOptions)`
  - Provenance: `PdfLocation(kind="pdf", page_number=int, bbox=[x0, y0, x1, y1], ocr_words=..., cell_char_spans=...)`
  - Result: `AgentResult` with `agent_name="pdf_agent"`

### C. Image Agent (`ImageAgent`)
- **Modality**: Document photographs, scanned prescription slips, paper lab reports (PNG, JPEG, WEBP, BMP, TIFF).
- **Ingestion**: `ImageReader` with Pillow safety hardening (15MB cap, 30MP pre-load check, decompression bomb defense), EXIF transpose, invertible Lanczos upscale, and OCR word confidence scoring.
- **Contract**:
  - Request: `ImageAgentRequest(content: bytes, source_name: str, options: ImageAgentOptions)`
  - Provenance: `ImageLocation(kind="image", bbox=[x0, y0, x1, y1], coordinate_system="pixel_top_left", ocr_words=...)`
  - Result: `AgentResult` with `agent_name="image_agent"`

### D. Voice Agent (`VoiceAgent`)
- **Modality**: Dictations, telehealth consultations, recorded doctor-patient audio (WAV, MP3, OGG, FLAC, M4A, WEBM).
- **Ingestion**: `AudioReader` with pre-ffprobe magic byte sniffing, bounded subprocess inspect (1MB output cap, 20s timeout, tempfile cleanup), duration checks ($\le 30$ mins), and Groq Whisper transcription (`whisper-large-v3-turbo`).
- **Contract**:
  - Request: `VoiceAgentRequest(content: bytes, source_name: str, options: VoiceAgentOptions)`
  - Provenance: `AudioLocation(kind="audio", start_ms=int, end_ms=int, speaker_id=...)`
  - Result: `AgentResult` with `agent_name="voice_agent"`

---

## 3. Core Architectural Guarantees

1. **Untouched Source Coordinates**:
   Offsets and locations represent exact zero-based indices (Unicode code points, PDF points, image display pixels, or audio milliseconds).
2. **Strict Independent Grounding**:
   LLMs are never trusted to output coordinates. `EvidenceValidator` verifies verbatim quotes independently against raw source segments.
3. **Deterministic Candidate Deduplication**:
   Exact duplicate candidates are merged while retaining distinct citations. Conflicting facts (e.g. contradictory blood glucose readings) are preserved with distinct candidate IDs.
4. **PHI Isolation by Default**:
   `ALLOW_REMOTE_LLM_PROCESSING=false` and `ALLOW_REMOTE_AUDIO_TRANSCRIPTION=false` ensure zero network transmission by default. All requests and results redact sensitive strings and byte payloads in `__repr__` and `__str__`.

---

## 4. Platform Infrastructure Connections

The platform architecture defines standard connections to enterprise services (detailed in `docs/integration-plan.md`):

1. **Storage & Persistence**:
   - **MongoDB**: Encrypted storage of `AgentResult` JSON, encounter aggregates, and candidate review histories.
   - **Object Storage (S3/MinIO)**: Encrypted storage of raw binaries outside operational database collections.
2. **FastAPI Gateway & Orchestrator**:
   - JWT Role-Based Access Control (`clinician`, `radiologist`, `patient`, `auditor`, `admin`).
   - MIME type inspection and dispatch to the matching modality agent.
3. **FHIR R4 Interoperability**:
   - `FactCandidate` mapped to FHIR `Observation`, `Condition`, `MedicationStatement`, `AllergyIntolerance`, and `Provenance`.
4. **Predictive Clinical Models**:
   - Validated candidate features feed into calibrated risk models (ASCVD 10-year risk score, diabetic triage).
5. **Doctor Review Dashboard**:
   - Interactive evidence linking with real-time text highlighting, PDF bounding box overlays, image ROI boxes, and audio citation playback.
