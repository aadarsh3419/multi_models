# Platform Integration Plan & Future Architecture Specification

This document outlines the end-to-end integration architecture for connecting all four modality agents (**Text**, **PDF**, **Image**, and **Voice**) into an enterprise-grade Healthcare Multi-Agent AI Platform.

---

## 1. System Architecture Diagram

```
                             ┌────────────────────────┐
                             │ Healthcare Client Apps │
                             │ (Web UI, EHR, Mobile)  │
                             └───────────┬────────────┘
                                         │ (HTTPS / TLS 1.3)
                                         ▼
                             ┌────────────────────────┐
                             │   FastAPI Gateway      │
                             │ (JWT Auth, RBAC, Rate) │
                             └───────────┬────────────┘
                                         │
                                         ▼
                             ┌────────────────────────┐
                             │  Platform Orchestrator │
                             │ (MIME Router & Tracing)│
                             └─────┬───┬───┬───┬──────┘
             ┌─────────────────────┘   │   │   └─────────────────────┐
             ▼                         ▼   ▼                         ▼
      ┌──────────────┐         ┌──────────┐ ┌───────────┐     ┌──────────────┐
      │  Text Agent  │         │ PDF Agent│ │Image Agent│     │ Voice Agent  │
      │ (Milestone 1)│         │(Milestone│ │(Milestone │     │ (Milestone 4)│
      │              │         │    2)    │ │    3)     │     │              │
      └──────┬───────┘         └────┬─────┘ └─────┬─────┘     └──────┬───────┘
             │                      │             │                  │
             └──────────────────────┼─────────────┼──────────────────┘
                                    ▼
                      ┌───────────────────────────┐
                      │ Shared Extraction Workflow│
                      │  - Field Extractor Adapters│
                      │  - Evidence Validator     │
                      │  - Value & Date Parsers   │
                      │  - Candidate Deduplicator │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                      ┌───────────────────────────┐
                      │    Common AgentResult     │
                      └─────────────┬─────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
  │ Encrypted Storage  │ │ FHIR R4 Bridge     │ │ ML Model Registry  │
  │ - MinIO / S3 Raw   │ │ - Observation      │ │ - ASCVD Risk       │
  │ - MongoDB Results  │ │ - Condition        │ │ - Diabetes Triage  │
  │ - Audit Trail      │ │ - MedicationStmt   │ │ - Framingham Score │
  └────────────────────┘ └────────────────────┘ └────────────────────┘
```

---

## 2. Modality Agents & Provenance Contracts

All four agents conform to the unified `BaseAgent` interface and produce a standardized `AgentResult`. Each attaches modality-specific spatial or temporal provenance:

| Modality Agent | Ingestion Source | Preprocessing & Safety | Coordinate Provenance (`EvidenceRecord.location`) |
|---|---|---|---|
| **Text Agent** | Raw clinical text (English, Hindi, Hinglish) | Lossless paragraph/line segmentation, NFC normalization, zero truncation | `TextLocation`: Unicode code-point character slice `[start_char, end_char]` |
| **PDF Agent** | PDF documents (native & scanned) | In-memory PyMuPDF stream, table extraction (`find_tables`), render pixel budgeting, selective OCR | `PdfLocation`: `page_number`, bounding box `[x0, y0, x1, y1]`, optional OCR words & cell spans |
| **Image Agent** | Printed document images (PNG, JPEG, WEBP, BMP, TIFF) | Pillow safety gates (15MB cap, 30MP pre-load check, bomb defense), EXIF transpose, invertible Lanczos upscale | `ImageLocation`: `bbox [x0, y0, x1, y1]`, `coordinate_system` (displayed vs oriented), OCR word confidences |
| **Voice Agent** | Dictations & recorded consultations (WAV, MP3, OGG, FLAC, M4A, WEBM) | Pre-ffprobe magic byte sniffing, bounded ffprobe inspect (1MB cap, 20s timeout), Whisper STT | `AudioLocation`: timestamps `[start_ms, end_ms]`, optional `speaker_id` |

---

## 3. Platform Orchestrator & API Layer

### A. FastAPI Asynchronous Gateway
Because agent parsing and OCR/transcription routines are CPU- and I/O-intensive, the platform utilizes FastAPI with thread pool delegation to avoid blocking the event loop:

```python
# Conceptual Gateway Endpoint
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import OAuth2PasswordBearer
import asyncio
from healthcare_agents.factory import (
    build_text_agent,
    build_pdf_agent,
    build_image_agent,
    build_voice_agent,
)

app = FastAPI(title="Healthcare Multi-Agent Platform API", version="1.0.0")


@app.post("/v1/encounter/extract", response_model=AgentResult)
async def extract_encounter_document(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(require_role(["clinician", "auditor"])),
):
    content_bytes = await file.read()
    mime_type = file.content_type or sniff_mime_type(content_bytes)

    # Dispatch blocking agent execution to worker thread pool
    if mime_type == "text/plain":
        agent = build_text_agent()
        req = TextAgentRequest(text=content_bytes.decode("utf-8"))
        result = await asyncio.to_thread(agent.process, req)
    elif mime_type == "application/pdf":
        agent = build_pdf_agent()
        req = PDFAgentRequest(content=content_bytes, source_name=file.filename)
        result = await asyncio.to_thread(agent.process, req)
    elif mime_type.startswith("image/"):
        agent = build_image_agent()
        req = ImageAgentRequest(content=content_bytes, source_name=file.filename)
        result = await asyncio.to_thread(agent.process, req)
    elif mime_type.startswith("audio/"):
        agent = build_voice_agent()
        req = VoiceAgentRequest(content=content_bytes, source_name=file.filename)
        result = await asyncio.to_thread(agent.process, req)
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported MIME type: {mime_type}")

    return result
```

### B. JWT Role-Based Access Control (RBAC)
User identities and permissions are enforced via cryptographically signed JWT tokens (RS256):

| Role | Permissions |
|---|---|
| `clinician` | Ingest documents, extract facts, review and affirm/reject candidates, view raw PHI |
| `radiologist` / `specialist` | Review image and lab fact candidates, edit clinical interpretations |
| `auditor` | View un-redacted evidence spans, inspect cryptographic SHA-256 hashes, verify audit logs |
| `patient` | Access affirmed summaries and personal health records (read-only) |
| `system_worker` | Internal background processing and FHIR synchronization |

### C. Orchestrator MIME Routing Matrix
The orchestrator sniffs file headers prior to dispatch to prevent MIME spoofing:
- `0x25 0x50 0x44 0x46` (`%PDF`) $\to$ `PDFAgent`
- `0x89 0x50 0x4E 0x47` (`.PNG`), `0xFF 0xD8 0xFF` (`JPEG`), `RIFF....WEBP` $\to$ `ImageAgent`
- `RIFF....WAVE`, `ID3` / `0xFF 0xFB` (`MP3`), `OggS`, `fLaC`, `ftyp` (`M4A`) $\to$ `VoiceAgent`
- Valid UTF-8 clinical narrative without binary magic $\to$ `TextAgent`

---

## 4. Encrypted Storage Architecture & MongoDB Schemas

### A. Raw Media Store (S3 / MinIO)
- Raw PDF files, high-resolution scans, and dictation audio streams are **never** stored in operational database documents.
- Encrypted at rest via AES-256-GCM / KMS in an S3-compatible object store:
  - Key schema: `s3://hipaa-clinical-media/{tenant_id}/{encounter_id}/{sha256_digest}.bin`
  - Access restricted via short-lived, pre-signed URLs (TTL $\le 15$ minutes).

### B. MongoDB Document Schemas & Indexes
The operational database stores structured `AgentResult` records and candidate review histories:

#### 1. `encounters` Collection
```json
{
  "_id": "enc_8f7b2a19-4c3e-4d2a-8b1c-9e2d3f4a5b6c",
  "patient_id": "pat_109283",
  "tenant_id": "tenant_apollo_delhi",
  "encounter_date": "2026-10-17T09:30:00Z",
  "encounter_type": "ambulatory_consultation",
  "document_refs": [
    {
      "request_id": "req_01HCXYZ...",
      "modality": "voice",
      "source_sha256": "5b9a8...",
      "s3_uri": "s3://hipaa-clinical-media/apollo/enc_8f7b.../audio_5b9a8.bin"
    }
  ],
  "created_at": "2026-10-17T09:35:12Z"
}
```

#### 2. `agent_results` Collection
```json
{
  "_id": "res_req_01HCXYZ...",
  "request_id": "req_01HCXYZ...",
  "encounter_id": "enc_8f7b2a19...",
  "modality": "voice",
  "status": "success",
  "candidate_count": 3,
  "candidates": [
    {
      "candidate_id": "cand_a1b2c3d4...",
      "fact_type": "symptom",
      "label_raw": "cough",
      "value_raw": "persistent",
      "assertion_status": "affirmed",
      "temporality": "current",
      "experiencer_type": "patient",
      "evidence_records": [
        {
          "segment_id": "seg_001",
          "quote": "persistent cough for 3 days",
          "location": {
            "kind": "audio",
            "start_ms": 1200,
            "end_ms": 3400,
            "speaker_id": "patient"
          }
        }
      ]
    }
  ],
  "audit": {
    "engine_version": "1.0.0",
    "provider": "groq",
    "model_id": "whisper-large-v3-turbo",
    "content_sha256": "5b9a8...",
    "duration_ms": 420.5
  }
}
```

#### 3. Database Indexes
```javascript
db.agent_results.createIndex({ "request_id": 1 }, { unique: true });
db.agent_results.createIndex({ "encounter_id": 1 });
db.agent_results.createIndex({ "candidates.candidate_id": 1 });
db.agent_results.createIndex({ "candidates.fact_type": 1, "candidates.label_raw": 1 });
db.agent_results.createIndex({ "audit.content_sha256": 1 });
```

---

## 5. FHIR R4 Interoperability Mapping

The platform provides a bidirectional FHIR R4 transformation layer to allow seamless exchange with standard Electronic Health Record (EHR) systems (e.g., Epic, Cerner, ABDM Ayushman Bharat Digital Mission):

| `FactCandidate.fact_type` | Target FHIR R4 Resource | Field Mapping Specification |
|---|---|---|
| `measurement` | **`Observation`** | - `Observation.status = 'preliminary'` (or `'final'` if reviewed)<br>- `Observation.code`: Coded via LOINC (e.g. LOINC `8480-6` for Systolic BP, `2339-0` for Glucose)<br>- `Observation.valueQuantity`: `value_numeric`, `unit_raw` mapped to UCUM (`mg/dL`, `mm[Hg]`)<br>- `Observation.referenceRange`: `reference_range_numeric` min/max<br>- `Observation.interpretation`: `abnormal_flag_raw` (e.g. `H` for high)<br>- `Observation.effectiveDateTime`: `date_raw` |
| `condition` | **`Condition`** | - `Condition.clinicalStatus`: `assertion_status` (`affirmed` $\to$ `active`, `negated` $\to$ `resolved`/`refuted`)<br>- `Condition.verificationStatus`: `assertion_status` (`uncertain` $\to$ `differential`/`provisional`)<br>- `Condition.code`: SNOMED CT term mapped from `label_raw`<br>- `Condition.recordedDate`: `date_raw` |
| `symptom` | **`Condition`** or **`Observation`** | - Category: `symptom-sign`<br>- `Condition.code`: Mapped SNOMED CT symptom code<br>- `Condition.evidence.detail`: Reference to source `DocumentReference` |
| `medication` | **`MedicationStatement`** | - `MedicationStatement.status`: `active` or `intended`<br>- `MedicationStatement.medicationCodeableConcept`: RxNorm code mapped from `label_raw`<br>- `MedicationStatement.dosage[0].text`: `value_raw` + `qualifier_raw` |
| `allergy` | **`AllergyIntolerance`** | - `AllergyIntolerance.clinicalStatus`: `active`<br>- `AllergyIntolerance.code`: Allergen substance code (SNOMED / RxNorm)<br>- `AllergyIntolerance.reaction.manifestation`: Reaction details |
| Any modality provenance | **`Provenance`** / **`DocumentReference`** | - `DocumentReference.content[0].attachment`: Pre-signed URI to source artifact<br>- `Provenance.target`: UUID of created FHIR resource<br>- `Provenance.entity[0].what`: Citation coordinate (`page`, `bbox`, or `timestamp_ms`) |

---

## 6. Downstream ML & Clinical Predictive Inference

Extracted, validated `FactCandidate` records are assembled into feature vectors for predictive clinical models. Raw unparsed text is **never** fed directly into statistical models.

### A. Cardiovascular Risk Calculator (ASCVD 10-Year Risk)
- **Features Extracted**:
  - Systolic Blood Pressure (`measurement`, `label_raw="blood pressure"`, systolic `Decimal`)
  - Total Cholesterol / HDL Cholesterol (`measurement`, `mg/dL`)
  - Diabetic Status (`condition`, `label_raw="diabetes"`, `assertion_status="affirmed"`)
  - Smoking Status (`fact_type="other"`, `label_raw="smoking"`, `assertion_status`)
  - Age / Gender (Encounter demographics)
- **Calculation**: Executes standard pooled cohort equations to output a calibrated 10-year percentage risk score.

### B. Diabetic Retinopathy & Glycemic Triage
- **Features Extracted**:
  - Fasting Plasma Glucose (`measurement`, `mg/dL`)
  - Glycated Hemoglobin HbA1c (`measurement`, `%`)
  - Duration of Diabetes (`condition`, `temporality`, `onset`)
- **Triage Recommendation**: Flags patient for annual dilated fundus examination if $HbA1c \ge 8.0\%$ or disease duration $> 5$ years.

---

## 7. Interactive Doctor Dashboard: Evidence Highlighting

The Doctor Review Portal enables human clinicians to verify candidates with single-click citation tracing:

```
┌───────────────────────────────────────┬────────────────────────────────────────┐
│ Original Source Document (Viewer)     │ Extracted Fact Candidates (Review)     │
├───────────────────────────────────────┼────────────────────────────────────────┤
│ [Patient Consultation Audio Player]   │ [Card 1: Symptom]                      │
│ 00:01.2 ────────●────────────── 00:03.4│ • Label: Cough                         │
│ "Patient reports persistent cough..." │ • Value: Persistent                    │
│                                       │ • Location: 00:01.2 - 00:03.4 (Audio)  │
│ [PDF Page 1 - Discharge Summary]      │                                        │
│ ┌───────────────────────────────────┐ │ [Card 2: Measurement]                  │
│ │ Blood Pressure: [ 130/85 mmHg ]   │ │ • Label: Blood Pressure                │
│ │ (Highlighted in yellow on hover!) │ │ • Value: 130/85 mmHg                   │
│ └───────────────────────────────────┘ │ • Coordinates: Page 1, bbox [45, 120]  │
│                                       │ [Approve]   [Edit]   [Reject]          │
└───────────────────────────────────────┴────────────────────────────────────────┘
```

1. **Text Modality**: Hovering over a fact highlights the character span `[start_char, end_char]` in the original raw text view.
2. **PDF Modality**: Hovering draws a highlighted SVG rectangle over the PDF page at bounding box `[x0, y0, x1, y1]`.
3. **Image Modality**: Hovering projects the oriented region-of-interest box onto the image canvas.
4. **Voice Modality**: Clicking "Play Citation" scrubs the HTML5 audio player directly to `start_ms` and plays through `end_ms`.
