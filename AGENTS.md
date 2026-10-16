# Agent System Architecture & Future Multimodal Contracts

This document outlines the agent contracts for the Healthcare Multi-Agent AI Platform.
Milestone 1 implements the foundational **Text Agent**. Future agents are documented here to specify interface boundaries and integration expectations.

---

## 1. Implemented Agent: Text Agent

- **Role**: Ingests unstructured or semi-structured clinical narratives (English, Hindi, Hinglish), preserves raw character coordinates, and produces grounded, validated clinical fact candidates.
- **Contract**:
  - Input: `TextAgentRequest` (contains raw string `text`, optional `language_hint`, validated `TextAgentOptions`).
  - Output: `AgentResult` (status, source metadata, validated `FactCandidate` records, audit metadata, structured errors).
- **Core Guarantees**:
  - Untouched source text preservation: offsets represent zero-based Python string indices (`len(str)` code points).
  - Strict evidence verification: facts must cite verbatim quotations from source segments.
  - Deterministic candidate IDs and exact deduplication.

---

## 2. Deferred Multimodal Agents (Future Milestones)

The platform architecture defines standard contracts for future modalities. These agents will build upon the same shared `ExtractionWorkflow`, `ClinicalFieldExtractor`, and `AgentResult` contracts.

### A. PDF Agent
- **Future Modality**: Native and scanned clinical documents (discharge summaries, laboratory reports, insurance forms).
- **Future Ingestion Tools**: PyMuPDF (`fitz`), pdfplumber.
- **Contract Specification**:
  - Request: `PdfAgentRequest(file_path: Path | bytes, ocr_fallback: bool = True)`
  - Modality Source Mapping: Maps text spans to page numbers and bounding boxes `[x0, y0, x1, y1]`.
  - Extension Point: `SourceSegment.multimodal_location = {"page": int, "bbox": [float, float, float, float]}`.
  - Result: Common `AgentResult` with `agent_name="pdf"`.

### B. Image Agent
- **Future Modality**: Scanned prescriptions, photographs of lab slips, mobile camera document captures.
- **Future Ingestion Tools**: Pillow (`PIL`), Tesseract OCR / Vision Models.
- **Contract Specification**:
  - Request: `ImageAgentRequest(image_bytes: bytes, mime_type: str, preprocess: bool = True)`
  - Modality Source Mapping: Bounding box region of interest (ROI) and polygon coordinates.
  - Extension Point: `SourceSegment.multimodal_location = {"roi_polygon": list[tuple[int, int]]}`.
  - Result: Common `AgentResult` with `agent_name="image"`.

### C. Voice Agent
- **Future Modality**: Recorded doctor-patient consultations, dictated physician notes, telehealth voice audio.
- **Future Ingestion Tools**: FFmpeg, Whisper / Speech-to-Text models.
- **Contract Specification**:
  - Request: `VoiceAgentRequest(audio_stream: bytes, sample_rate: int, language_hint: str | None)`
  - Modality Source Mapping: Audio segment timestamps `[start_ms, end_ms]` and speaker diarization labels.
  - Extension Point: `SourceSegment.multimodal_location = {"timestamp_ms": [int, int], "speaker": str}`.
  - Result: Common `AgentResult` with `agent_name="voice"`.

---

## 3. Future Platform Infrastructure Connections

In future milestones, the outputs of all modality agents will connect to shared enterprise infrastructure:

1. **Storage & Database Layer**:
   - **MongoDB**: Storage of structured `AgentResult` JSON, patient encounter documents, and candidate revisions.
   - **Object Storage (S3/MinIO)**: Encrypted, HIPAA-compliant raw document and audio storage (storing source content securely away from operational databases).
2. **Orchestrator**:
   - Routes incoming patient inputs (MIME type inspection) to the appropriate modality agent (`text`, `pdf`, `image`, `voice`).
   - Merges cross-modal extraction results into an encounter-level longitudinal graph.
3. **ML Model Registry & Disease Inference**:
   - Consumes validated `FactCandidate` records (e.g., fasting glucose, blood pressure, HbA1c) to trigger specialized predictive models (e.g. cardiovascular risk scores, diabetic retinopathy risk).
4. **Doctor Dashboard & Patient Summaries**:
   - Generates read-only, evidence-grounded summaries with interactive citation hovering linking back to original PDF pages, text spans, or audio playback timestamps.

