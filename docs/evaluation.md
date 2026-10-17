# Clinical Text & Multi-Modal Extraction Evaluation Methodology

**Document Version**: 4.0.0  
**Status**: Authoritative Evaluation Specification  
**Applies to**: Full Platform (Milestones 1–4: Text, PDF, Image, Voice)

---

## 1. Ethical & Scientific Disclaimer

> [!IMPORTANT]
> **Synthetic Regression Benchmark Only**: The evaluation datasets (`tests/evaluation/dev_cases.json` and `tests/evaluation/heldout_cases.json`) consist entirely of synthetic, fictitious clinical narratives authored for automated software regression testing. They do NOT represent real patient health records (PHI), and performance on these benchmarks does NOT establish medical diagnostic accuracy, clinical safety, or regulatory compliance. Independent human clinical review is mandatory prior to any healthcare deployment.

---

## 2. Evaluation Split & Manifest Integrity

The platform benchmark comprises 38 carefully authored synthetic cases spanning 14 clinical categories (vital signs, laboratory tests, medications, allergies, conditions, family history, negation, temporality, ambiguous dates, prompt injection, and multilingual Hindi/Hinglish):

- **Development Split (`tests/evaluation/dev_cases.json`)**: 24 cases used during model development, prompt engineering, and parsing rule refinement.
- **Held-out Split (`tests/evaluation/heldout_cases.json`)**: 14 cases reserved for unbiased evaluation.
- **Cryptographic Seal (`tests/evaluation/heldout_manifest.json`)**:
  - SHA-256 Digest: `535fa32830de62d0e0ea67064a99505778ee9c8a48d31a7dc6cfa0f4b04484f9`
  - Case count: 14 held-out cases
  - Any unauthorized modification or drift in held-out fixtures fails `tests/unit/test_evaluation.py` immediately.

---

## 3. Measured vs. Simulated Components

To maintain scientific honesty and clarity during academic audits and viva examinations, the benchmark distinguishes between measured downstream components and simulated provider outputs:

| Component | In Offline Benchmark Mode (`scripts/evaluate_text_extraction.py`) | In Live Opt-in Mode (`--live-groq`) |
|---|---|---|
| **Upstream LLM Generation** | **Simulated**: Deterministic test double (`SyntheticBenchmarkFakeExtractor`) feeds expected candidates | **Measured**: Live Groq Cloud API calls (`llama-3.3-70b-versatile` / `openai/gpt-oss-20b`) |
| **Evidence Validation** | **Measured**: Independent quote searching & coordinate mapping | **Measured**: Independent quote searching & coordinate mapping |
| **Value & Date Parsing** | **Measured**: Pure syntactic Decimal & date parsing | **Measured**: Pure syntactic Decimal & date parsing |
| **Candidate Deduplication** | **Measured**: Exact duplicate merging & conflict tracking | **Measured**: Exact duplicate merging & conflict tracking |
| **Bipartite Scorer** | **Measured**: Greedy key matching & attribute evaluation | **Measured**: Greedy key matching & attribute evaluation |

> [!NOTE]
> Offline benchmark metrics of Precision = 1.0, Recall = 1.0, and F1 = 1.0 prove that the **downstream extraction engine is 100% lossless and deterministic**—it never corrupts coordinates, drops attributes, or alters values. It does not assert that live Groq models achieve 100% accuracy on open-ended clinical narratives.

---

## 4. Clinical Fact Labeling Guide

The purpose of the platform is **verbatim clinical information extraction**, not diagnosis, translation, or inference. When annotating gold standard facts, annotators and reviewers must strictly adhere to these rules:

### A. General Principles
1. **Explicit Mentions Only**: Extract only facts directly stated in the text. Never infer conditions, unmentioned medications, or unstated values.
2. **Untouched Verbatim Wording**: All raw fields (`label_raw`, `value_raw`, `unit_raw`, `reference_range_raw`, `abnormal_flag_raw`, `date_raw`, `time_raw`, `qualifier_raw`, `experiencer_relationship_raw`) must be extracted verbatim as printed in the text (English, Devanagari Hindi, or Hinglish). Do not translate, transliterate, or standardize spellings.
3. **Empty Fields Represented as Empty Strings**: In the provider-facing schema, unmentioned attributes are represented as empty string `""` (which internal validation maps to `None`).
4. **Verbatim Evidence Quotes**: Every fact must cite one or more verbatim substring quotes (`evidence_refs: [{"segment_id": ..., "quote": "..."}]`). The quote must physically exist in the source segment.
5. **Raw Field Containment Rule**: Every non-empty raw field must appear as an exact substring within the cited evidence quotes.

### B. Semantic Enums & Rules

#### 1. Fact Type (`fact_type`)
- `measurement`: Quantified physiological, laboratory, or diagnostic observations (e.g. Blood Pressure, Glucose, Hemoglobin, Heart Rate, SpO2).
- `symptom`: Patient-reported complaints or subjective sensations (e.g. fever, headache, nausea, cough, chest pain).
- `medication`: Prescribed or administered pharmacological substances (e.g. Paracetamol, Metformin, Lisinopril, Amoxicillin).
- `allergy`: Hypersensitivity or adverse drug/food reactions (e.g. Penicillin allergy, peanut allergy).
- `condition`: Diagnosed illnesses, diseases, or medical syndromes (e.g. Type 2 Diabetes, Hypertension, Asthma).
- `history`: Family medical history or explicit non-recent past medical events.
- `other`: Relevant clinical facts not fitting the categories above.

#### 2. Assertion Status (`assertion_status`)
- `affirmed`: The fact is explicitly reported as present or occurring (e.g. "patient has cough", "BP 130/80").
- `negated`: The fact is explicitly denied or stated as absent (e.g. "denies chest pain", "no fever", "negative for diabetes").
- `uncertain`: The presence is stated as possible, suspect, or rule-out (e.g. "suspected pneumonia", "possible migraine").

#### 3. Temporality (`temporality`)
- `current`: Actively present at the time of documentation.
- `past`: Occurred in the past and resolved.
- `future`: Planned or scheduled for the future (e.g. "scheduled for surgery next week").
- `recurrent`: Periodic or episodic episodes (e.g. "recurrent migraines").
- `historical_unspecified`: Historical context without specific chronological boundaries.
- `uncertain`: Chronology cannot be determined.
- `not_stated`: Chronological timing is not mentioned.
- `not_applicable`: Temporality logically does not apply (specifically: **all reported measurements** take `not_applicable` because a measurement is an instantaneous numerical reading; its date belongs in `date_raw`).

#### 4. Experiencer (`experiencer_type` and `experiencer_relationship_raw`)
- `patient`: The patient being examined/treated. First-person statements ("I have headache", "my cough") default to `patient`.
- `family_member`: Medical facts pertaining to family members (e.g. "Father died of myocardial infarction"). The relationship must be stored verbatim in `experiencer_relationship_raw` (e.g. "father").
- `other_person`: Facts regarding an unrelated individual (e.g. donor, roommate).
- `unknown`: Experiencer cannot be determined.
- `not_applicable`: When experiencer does not logically apply.

#### 5. Reporting Source Role (`reporting_source_role`)
- `clinician`: Stated by a doctor, nurse, or physician (e.g. "Dr. Sharma noted...", "Physical exam revealed...").
- `patient`: Self-reported by the patient (e.g. "Patient states...", "I feel...").
- `caregiver`: Reported by parent, spouse, or aide (e.g. "Mother reports child was irritable...").
- `laboratory`: Printed on laboratory slips or pathology reports.
- `device`: Automatic telemetry or monitor recordings.
- `document_author`: Author of the medical summary.
- `third_party`: EMS, police, employer.
- `unknown`: No explicit source stated.

---

## 5. Exact Matching & Scoring Rules

The benchmark evaluation module (`healthcare_agents.evaluation`) assesses predictions against gold standard facts using deterministic bipartite matching:

### A. Candidate Fact Key Identity
Two candidate facts match if and only if their normalized key matches:
1. **Label**: `NFC` normalized, case-folded (`casefold()`), whitespace-collapsed (`" ".join(text.split())`).
2. **Value**: `NFC` normalized, whitespace-collapsed, **case-sensitive** (exact string).
3. **Unit**: `NFC` normalized, whitespace-collapsed, **case-sensitive** (exact string).

### B. Standard Aggregate Metrics
- $\text{Micro Precision} = \frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FP}}$
- $\text{Micro Recall} = \frac{\sum \text{TP}}{\sum \text{TP} + \sum \text{FN}}$
- $\text{Micro F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 6. Cross-Agent Invariants Verification Suite

Milestone 4 introduces the cross-agent invariants test suite (`tests/cross_agent/test_cross_agent_invariants.py`). This suite runs 20 parametrized architectural invariant tests across all four agents (`text`, `pdf`, `image`, `voice`):

1. **Common `AgentResult` Structure**: Invariant across all modalities (contains `request_id`, `status`, `candidates`, `audit`, `errors`, `warnings`).
2. **Zero PHI in String Representation**: Verifies `str(request)` and `repr(request)` redact patient content across all four request types.
3. **Location Subtype Invariance**: Verifies that every extracted fact candidate possesses evidence records whose location matches the expected modality class (`TextLocation`, `PdfLocation`, `ImageLocation`, or `AudioLocation`).
4. **Offline Partial Status**: In the default offline configuration, every agent successfully ingests documents and terminates with `status="partial"` without network egress.
5. **Universal Error Code Adherence**: All errors emitted across all four agents use valid enum members of `ErrorCode`.
6. **Strict Dependency Graph**: AST static inspection verifies that no agent or extraction module imports prohibited deep learning frameworks or leaks provider logic.
