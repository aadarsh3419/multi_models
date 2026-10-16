# Platform Integration Plan & Future Milestones

This document outlines the evolutionary roadmap for integrating the Text Extraction Foundation into the full Healthcare Multi-Agent AI Platform.

---

## 1. Modality Agent Integrations

```
                     ┌──────────────────┐
                     │ Multi-Modal Ingest│
                     └────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    [Text Agent]        [PDF Agent]         [Voice Agent]
    (Milestone 1)      (Milestone 2)        (Milestone 4)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                  [Shared Extraction Workflow]
                  - Provider Adapter
                  - Evidence Validator
                  - Parsers & Deduplicator
                              │
                              ▼
                  [Common AgentResult]
                              │
                              ▼
                  [Platform Orchestrator]
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
 [MongoDB / Data Lake]                 [Predictive ML Models]
 - Encrypted encounter storage         - Cardiovascular risk
 - Longitudinal patient graph          - Diabetes progression
```

### Milestone 2: PDF Agent
- **Ingestion**: Utilizes PyMuPDF (`fitz`) and pdfplumber to extract text, tables, and embedded forms from PDF documents.
- **Multimodal Coordinates**: Maps extracted words to PDF page numbers and bounding boxes `[x0, y0, x1, y1]`.
- **Extension Point**: `SourceSegment.multimodal_location = {"page": int, "bbox": [float, float, float, float]}`.
- **Scanned PDF Fallback**: Triggers optical character recognition (OCR) when native text is absent.

### Milestone 3: Image Agent
- **Ingestion**: Pillow (`PIL`) and Tesseract / Vision model adapters.
- **Multimodal Coordinates**: Extracts text with polygon coordinates mapping to image regions of interest (ROI).
- **Extension Point**: `SourceSegment.multimodal_location = {"roi_polygon": list[tuple[int, int]]}`.

### Milestone 4: Voice Agent
- **Ingestion**: FFmpeg for audio normalization, resampling, and format transcoding; OpenAI Whisper or local STT models for transcription.
- **Multimodal Coordinates**: Audio segment timestamps `[start_ms, end_ms]` and speaker diarization tags.
- **Extension Point**: `SourceSegment.multimodal_location = {"timestamp_ms": [int, int], "speaker": str}`.

---

## 2. Infrastructure & Persistence Connections

### Encrypted Storage Architecture
1. **Raw Document Storage**:
   - Stored in an S3-compatible, HIPAA-compliant object store (e.g. MinIO / AWS S3) with AES-256 server-side encryption.
   - Raw patient files and media streams are stored outside the primary application database.
2. **Operational Database (MongoDB)**:
   - Stores structured `AgentResult` records, patient profiles, and longitudinal encounters.
   - Indexes:
     - `request_id` (unique index)
     - `patient_id` + `encounter_date`
     - `candidates.fact_type` + `candidates.label_raw`

---

## 3. Orchestration & Longitudinal Clinical Graph

- **Routing Engine**: Inspects input payload MIME types (`text/plain`, `application/pdf`, `image/png`, `audio/wav`) and delegates to the appropriate agent.
- **Encounter Aggregation**: Merges facts across multiple documents within an encounter, performing cross-document deduplication while retaining distinct evidence trails.

---

## 4. Downstream ML & Clinical Interfaces

### Predictive Disease Models
- Structured fact candidates (e.g. `fasting glucose`, `systolic BP`, `total cholesterol`, `smoking status`) serve as typed feature inputs for standardized clinical risk calculators (e.g., ASCVD 10-year risk, Framingham risk score).
- Raw unparsed text is never fed into statistical models.

### Doctor Dashboard & Patient Summaries
- Clinical review portal displaying extracted facts alongside their verification status.
- Interactive evidence inspection: Clicking on an extracted fact highlights the exact text span in the original document, jumps to the PDF bounding box, or plays the corresponding audio slice.

