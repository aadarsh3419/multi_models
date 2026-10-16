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

