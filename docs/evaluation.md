# Clinical Text Extraction Evaluation & Benchmarking

This document details the evaluation methodology, matching algorithms, metrics computation, and benchmarking procedures for Milestone 1.

---

## 1. Deterministic Matching Methodology

In clinical informatics, semantic similarity models or fuzzy string matching (e.g. Levenshtein distance, cosine similarity of embeddings) are dangerous because small token differences often alter medical meaning entirely (e.g. *Hypo*glycemia vs *Hyper*glycemia, `10 mg` vs `100 mg`).

The platform's evaluation scorer (`scripts/evaluate_text_extraction.py`) uses **strict deterministic matching rules**:

1. **Fact Type**: Must match exactly (`predicted.fact_type == expected.fact_type`).
2. **Clinical Label**:
   - Normalized via Unicode NFC.
   - Case-folded (`str.casefold()`).
   - Whitespace runs collapsed to single spaces and trimmed.
   - Must match identically.
3. **Raw Value**:
   - Whitespace collapsed and trimmed.
   - Must match identically (e.g., `120/80` does not match `120`).
4. **Unit**:
   - Stripped of outer whitespace.
   - Case and symbols preserved (e.g., `mg/dL` vs `g/dL` must never match).
5. **Assertion Status**: Must match exactly (`affirmed`, `negated`, `uncertain`).
6. **Temporality**: Must match exactly (`current`, `past`, `future`, etc.).
7. **Experiencer**: Must match exactly (`patient`, `family_member`, `other_person`).
8. **Reporting Role**: Must match when specified in the gold benchmark.

---

## 2. Mathematical Metrics Calculation

### Micro-Averaged Precision and Recall
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1 Score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Zero Denominator Policy
If a category contains zero expected facts and zero predicted facts (e.g., an adversarial prompt injection test that yields zero facts as expected), reporting `0.0%` or `100.0%` would be mathematically misleading:
$$\text{If } TP + FP = 0 \implies \text{Precision} = \text{"N/A"}$$
$$\text{If } TP + FN = 0 \implies \text{Recall} = \text{"N/A"}$$
$$\text{If } \text{Precision} = \text{"N/A"} \text{ or } \text{Recall} = \text{"N/A"} \implies \text{F1} = \text{"N/A"}$$

### Evidence Grounding Score
Separately from fact classification correctness, the scorer verifies whether the extracted evidence quotations were successfully matched and mapped to the underlying text:
- `Evidence TP`: Predicted fact matched AND its quotation is uniquely verified (`evidence_status == "matched"`).
- `Evidence FN`: Predicted fact matched BUT its quotation was missing, invalid, or duplicate (`evidence_status != "matched"`).

---

## 3. Benchmark Dataset Description

The static benchmark dataset (`tests/fixtures/synthetic/labeled_text_cases.json`) contains **38 static synthetic cases** representing 14 clinical categories:

| Category | Cases | Key Testing Focus |
|---|---|---|
| `basic_measurements` | 2 | Numerical values, standard units (bpm, breaths/min). |
| `reference_ranges_and_flags` | 3 | Preserved lab ranges, High (H) and Low (L) report flags. |
| `negation_and_uncertainty` | 4 | Denied symptoms, absence of history, clinician suspicion. |
| `family_history` | 2 | Relative experiencer (`father`, `mother`) vs patient diagnosis. |
| `experiencer_types` | 1 | Third-party / roommate experiencer. |
| `temporality` | 5 | Current, past resolved, historical unspecified, future, recurrent. |
| `reporting_source_roles` | 4 | Patient self-report, caregiver, automated device, laboratory. |
| `medications` | 3 | Dose, route, frequency, duration, discontinued status. |
| `ambiguous_dates` | 1 | Slashed dates with `DATE_ORDER=UNKNOWN` flagged ambiguous. |
| `unambiguous_dates` | 1 | Day > 12 resolved mathematically without guessing. |
| `iso_dates` | 1 | Standard ISO 8601 YYYY-MM-DD parsing. |
| `blood_pressure` | 1 | Ratio parsed as BP under explicit context allowlist. |
| `numeric_ranges` | 1 | Dash and word range bounds (e.g. `2.0-3.0`). |
| `ambiguous_commas` | 2 | Decimal commas and thousands separators flagged ambiguous. |
| `repeated_quotes` | 1 | Duplicate identical quotes in same segment marked ambiguous. |
| `conflicting_values` | 1 | Differing values on same date preserved with conflict warning. |
| `prompt_injection` | 1 | Adversarial injection ("Ignore instructions") produces 0 facts. |
| `hindi_clinical` | 1 | Devanagari script, combining matras, Unicode offsets. |
| `hinglish_clinical` | 1 | Multilingual code-switching clinical narrative. |
| `allergies` | 1 | Allergen and adverse reaction mentions. |
| `procedures` | 1 | Surgical and diagnostic procedures with dates. |

---

## 4. Execution Commands

### Offline Benchmark Execution
```powershell
python scripts/evaluate_text_extraction.py
```

### Opt-In Live Groq Cloud Evaluation
```powershell
$env:ALLOW_REMOTE_LLM_PROCESSING = "true"
$env:GROQ_API_KEY = "gsk_your_actual_key"
python scripts/evaluate_text_extraction.py --live-groq
```

---

## 5. Clinical Disclaimer

> [!IMPORTANT]
> The synthetic cases included in this repository are intended solely as an automated software engineering and regression benchmark. They do NOT establish clinical accuracy, diagnostic reliability, or regulatory compliance (e.g., FDA SaMD, HIPAA, or CE-IVD). Any evaluation metrics produced against this synthetic dataset require independent validation by qualified medical professionals before they can be considered a verified clinical gold standard.

