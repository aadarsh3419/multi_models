# Data Contracts & Clinical Semantic Definitions

This document details the shared Pydantic v2 schemas and controlled enumerations for Milestone 1.

---

## 1. Controlled Clinical Enumerations

In clinical information extraction, ambiguous terminology leads to dangerous misunderstandings. The platform defines strict semantic boundaries for all categorical fields:

### AssertionStatus
- `affirmed`: The condition or finding is explicitly stated as present or confirmed.
- `negated`: The finding is explicitly stated as absent or denied (e.g. *"denies fever"*).
- `uncertain`: The source expresses clinical suspicion, possibility, or unconfirmed status (e.g. *"possible appendicitis"*).

### Temporality
- `current`: Actively present at the time of the encounter.
- `past`: Occurred in the past and resolved (e.g. *"childhood asthma at age 10"*).
- `future`: Planned or scheduled for the future (e.g. *"scheduled for surgery next month"*).
- `recurrent`: Episodes that repeat over time (e.g. *"recurrent migraines"*).
- `historical_unspecified`: Mentioned as historical, but without a precise date or status.
- `uncertain`: Temporal relationship is ambiguous from the text.
- `not_stated`: The source text makes no reference to the timing of the observation.
- `not_applicable`: Temporality does not apply to this fact type.

### ExperiencerType
- `patient`: The subject experiencing the finding is the patient.
- `family_member`: The finding pertains to a relative (e.g. *"father had coronary disease"*).
- `other_person`: Pertains to an unrelated individual (e.g. *"roommate tested positive for COVID-19"*).
- `unknown`: Experiencer cannot be determined from context.
- `not_applicable`: Finding does not pertain to an individual subject.

### ReportingSourceRole
- `patient`: Communicated directly by the patient.
- `caregiver`: Communicated by a family member or caregiver.
- `clinician`: Documented by a physician, nurse, or medical provider.
- `laboratory`: Emitted by a laboratory diagnostic test.
- `device`: Recorded by an automated monitoring instrument (e.g. Holter monitor, automated BP cuff).
- `document_author`: Formal author of the clinical narrative.
- `third_party`: Employer, insurer, or external reporter.
- `unknown`: Role cannot be determined.
- `not_applicable`: Role does not apply.

### Crucial Semantic Distinctions
The pipeline strictly enforces the differences between the following values:
- `not_stated`: The source text simply does not provide this information.
- `not_applicable`: The field is logically irrelevant for this fact category.
- `unknown`: The information could apply, but the available text does not reveal it.
- `uncertain`: The source document itself expresses explicit clinical uncertainty.

> [!CAUTION]
> Missing source information must **never** be coerced into a numerical zero, boolean `false`, or assumed to represent a "normal" finding.

---

## 2. Core Domain Models

### TextAgentRequest
The validated entrypoint contract for the Text Agent:
```python
class TextAgentRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, repr=False)
    language_hint: str | None = None
    options: TextAgentOptions = Field(default_factory=TextAgentOptions)
```
- Non-empty validation enforces at least one non-whitespace character.
- Text length is validated against `options.max_text_chars` (default 36,000 code points).
- Safe representation (`repr=False`) prevents raw clinical text from leaking into logs or exceptions.

### FactCandidate
The comprehensive internal domain model representing an extracted fact:
- `candidate_id`: Deterministic hash derived from source content digest, evidence coordinates, and normalized fields.
- `fact_type`: Controlled `FactType` enum.
- `label_raw`, `value_raw`, `unit_raw`: Verbatim text elements from source.
- `reference_range_raw`, `abnormal_flag_raw`: Preserved lab metadata without synthetic flags.
- `assertion_status`, `temporality`, `experiencer_type`, `reporting_source_role`: Controlled enums.
- `evidence_refs`: List of `ValidatedEvidenceRecord` containing verified segment IDs and start/end coordinates.
- `value_parsed`: Optional `ValueParseResult` (scalar, range, ratio, comparator).
- `date_parsed`: Optional `DateParseResult` (ISO date, rule identifier).
- `conflict_warning_refs`: Candidate IDs of conflicting observations.
- `review_status`: Defaults to `unreviewed`. Clinical verification requires explicit human physician review.

### Common AgentResult
The standardized contract emitted across the platform:
- `schema_version`: Semantic schema version (`1.0.0`).
- `request_id`: Request identifier.
- `agent_name`: Name of the emitting agent (`text`).
- `status`: Overall outcome (`success`, `partial`, or `failed`).
- `candidates`: List of fully validated `FactCandidate` objects.
- `stage_outcomes`: Dictionary of stage execution statuses and execution durations.
- `warnings`: Operational and clinical conflict warnings.
- `errors`: Machine-readable `StructuredError` records.
- `audit_metadata`: Complete runtime provenance, prompt version, prompt template hash, token usage, and system versioning.

---

## 3. Milestone 2: PDF Agent Contracts & Provenance Locations

Milestone 2 introduces additive contracts for ingesting and spatially locating clinical facts within PDF documents. All Milestone 1 contracts remain frozen and fully compatible.

### A. PDFAgentRequest & PDFAgentOptions

```python
class PDFAgentOptions(BaseModel):
    ocr_mode: Literal["auto", "never", "always"] = "auto"
    ocr_languages: str = "eng"  # strictly validated by regex ^[a-z]{3}(\+[a-z]{3})*$
    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = None
    include_source_text_in_result: bool = False


class PDFAgentRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    source_name: str | None = Field(default=None, repr=False)
    content: bytes = Field(..., exclude=True, repr=False)
    options: PDFAgentOptions = Field(default_factory=PDFAgentOptions)
```

- **PHI Security**: `content` bytes are marked with `exclude=True` and `repr=False`. Raw file bytes are never emitted in logs, exception traces, or JSON serialization.
- **Source Privacy**: `source_name` (e.g. filename) is kept internally and redacted from `__repr__`, `__str__`, and default audit logs to avoid leaking patient identifiers embedded in filenames.

### B. PdfLocation Contract (Spatial Provenance)

When facts are extracted from PDF documents, their `evidence_records[].location` contains a `PdfLocation`:

```python
class PdfLocation(BaseModel):
    kind: Literal["pdf"] = "pdf"
    page_number: int = Field(ge=1, description="1-indexed page number.")
    bbox: list[float] = Field(description="[x0, y0, x1, y1] bounding box in PDF points.")
    coordinate_system: Literal["pdf_points_top_left"] = "pdf_points_top_left"
    page_width: float
    page_height: float
    rotation: int = 0  # 0, 90, 180, or 270 degrees
    block_id: str | None = None
    line_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    cell_id: str | None = None
    cell_char_spans: list[tuple[int, int]] | None = None
    extraction_method: str | None = None  # e.g. "pymupdf:native" or "ocr:tesseract"
    ocr_words: list[dict[str, Any]] | None = None
    layout_group_id: str | None = None
```

- **Top-Left Unrotated Point Coordinates**: Regardless of original display rotation, coordinates are normalized to the unrotated page coordinate space ($72\text{ points per inch}$).
- **Evidence Quote Span**: `EvidenceRecord.quote_char_span: tuple[int, int] | None` records the start and end character offsets within the referenced segment's text.

### C. Shared OcrService Contracts

The image-agnostic OCR service defines standard tokens and lines:

```python
class OCRWord(BaseModel):
    text: str
    bbox: list[int]  # [x0, y0, x1, y1] pixel coordinates
    confidence: float | None = None  # 0.0 - 100.0


class OCRLine(BaseModel):
    text: str
    bbox: list[int]  # [x0, y0, x1, y1] pixel bounding box enclosing line
    words: list[OCRWord] = Field(default_factory=list)
    block_id: int = 1
    par_id: int = 1
    line_id: int = 1


class OCRResult(BaseModel):
    text: str
    lines: list[OCRLine] = Field(default_factory=list)
    words: list[OCRWord] = Field(default_factory=list)
    language: str = "eng"
    engine_version: str = "tesseract"
```

---

## 4. Milestone 3: Image Agent Contracts & Pixel Provenance

Milestone 3 introduces additive contracts for ingesting document images (photos of prescriptions, scanned lab slips, clinic notes). All M1 and M2 contracts remain frozen and fully compatible.

### A. ImageAgentRequest & ImageAgentOptions

```python
class ImageAgentOptions(BaseModel):
    ocr_languages: str = "eng"  # strictly validated by regex ^[a-z]{3}(\+[a-z]{3})*$
    date_order_hint: Literal["UNKNOWN", "DMY", "MDY"] | None = None
    include_source_text_in_result: bool = False
    upscale_enabled: bool = True  # Enable Lanczos upscaling when shorter side < 1000px


class ImageAgentRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    source_name: str | None = Field(default=None, repr=False)
    content: bytes = Field(..., exclude=True, repr=False)
    options: ImageAgentOptions = Field(default_factory=ImageAgentOptions)
```

- **PHI Security**: `content` raw bytes are marked `exclude=True` and `repr=False`. Raw image bytes are never serialized or logged.
- **Source Privacy**: `source_name` is redacted in string representations to prevent filename PHI leakage.

### B. ImageLocation Contract (Display Pixel Provenance)

When facts are extracted from image documents, `evidence_records[].location` contains an `ImageLocation`:

```python
class ImageLocation(BaseModel):
    kind: Literal["image"] = "image"
    bbox: list[float] = Field(description="[x0, y0, x1, y1] bounding box in displayed pixel space.")
    original_width: int
    original_height: int
    coordinate_system: str = "pixel_top_left"  # or "oriented_pixel_top_left"
    exif_orientation: int | None = None  # 1-8
    oriented_width: int | None = None
    oriented_height: int | None = None
    extraction_method: str | None = "ocr:tesseract"
    ocr_words: list[dict[str, Any]] | None = None
    layout_group_id: str | None = None
```

- **Displayed Pixel Space**: Coordinates $[px_0, py_0, px_1, py_1]$ represent displayed image pixels after EXIF orientation transposition.
- **Invertible Upscaling Provenance**: When smaller images are upscaled during OCR preprocessing, coordinates are automatically divided by the scaling factor $S$ so bounding boxes align exactly with the input image.

### C. Milestone 3 Error and Warning Codes

- `ErrorCode.IMAGE_PIXEL_LIMIT_EXCEEDED = "image_pixel_limit_exceeded"`: Image uncompressed dimensions exceed configured limit (`MAX_IMAGE_PIXELS=30,000,000`).
- `WarningCode.OCR_LOW_CONFIDENCE = "ocr_low_confidence"`: One or more OCR word tokens fell below `OCR_LOW_CONFIDENCE=60.0`.



