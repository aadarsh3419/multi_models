"""Interactive end-to-end extraction demonstration.

Runs the full pipeline on a multi-fact clinical note using an injected
extraction double to demonstrate every stage:
- Segmentation & Offset Tracking
- Candidate Extraction
- Evidence Quotation Verification & Original Coordinate Mapping
- Deterministic Value & Date Parsing
- Candidate Deduplication & Auditing
"""

import sys
from pathlib import Path
from typing import Any

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from healthcare_agents.contracts.facts import (
    AssertionStatus,
    ExperiencerType,
    FactType,
    ProviderCandidate,
    ProviderEvidenceRef,
    ProviderExtractionResponse,
    ReportingSourceRole,
    Temporality,
)
from healthcare_agents.contracts.requests import TextAgentOptions, TextAgentRequest
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.factory import build_text_agent


class DemoClinicalExtractor(ClinicalFieldExtractor):
    """Demonstration extractor that extracts clinically grounded candidates from sample note."""

    def extract(
        self,
        unit: Any,
        language_hint: str | None = None,
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        candidates: list[ProviderCandidate] = []
        text = getattr(unit, "text", "") or ""
        unit_id = getattr(unit, "unit_id", "s0001")

        if "Blood Pressure: 128/82 mmHg" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Blood Pressure",
                    value_raw="128/82",
                    unit_raw="mmHg",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Blood Pressure: 128/82 mmHg",
                        )
                    ],
                )
            )
        if "Heart Rate: 72 bpm" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Heart Rate",
                    value_raw="72",
                    unit_raw="bpm",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Heart Rate: 72 bpm",
                        )
                    ],
                )
            )
        if "Fasting Blood Glucose: 118 mg/dL" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Fasting Blood Glucose",
                    value_raw="118",
                    unit_raw="mg/dL",
                    reference_range_raw="70-99",
                    abnormal_flag_raw="High",
                    date_raw="2025-04-03",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.LABORATORY,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Fasting Blood Glucose: 118 mg/dL (Reference Range: 70-99 mg/dL) [Flag: High]",
                        ),
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="2025-04-03",
                        ),
                    ],
                )
            )
        if "Hemoglobin A1c: 6.8 %" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Hemoglobin A1c",
                    value_raw="6.8",
                    unit_raw="%",
                    reference_range_raw="4.0-5.6",
                    abnormal_flag_raw="H",
                    date_raw="2025-04-03",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.LABORATORY,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Hemoglobin A1c: 6.8 % (Reference Range: 4.0-5.6 %) [Flag: H]",
                        ),
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="2025-04-03",
                        ),
                    ],
                )
            )
        if "Father had coronary artery disease at age 52" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.HISTORY,
                    label_raw="coronary artery disease",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.PAST,
                    experiencer_type=ExperiencerType.FAMILY_MEMBER,
                    experiencer_relationship_raw="father",
                    reporting_source_role=ReportingSourceRole.PATIENT,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Father had coronary artery disease at age 52",
                        )
                    ],
                )
            )
        if "Metformin 500 mg oral twice daily for 6 months" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEDICATION,
                    label_raw="Metformin",
                    value_raw="500 mg",
                    qualifier_raw="twice daily oral",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Metformin 500 mg oral twice daily for 6 months",
                        )
                    ],
                )
            )
        if "denies chest pain or shortness of breath" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.SYMPTOM,
                    label_raw="chest pain",
                    assertion_status=AssertionStatus.NEGATED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.PATIENT,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="denies chest pain or shortness of breath",
                        )
                    ],
                )
            )

        resp = ProviderExtractionResponse(candidates=candidates)
        meta = {
            "provider": "demo_extractor",
            "model_id": "clinical-demo-v1",
            "response_id": "demo_resp_001",
        }
        return resp, meta


def run_demo() -> None:
    note_path = Path("sample_clinical_note.txt")
    with open(note_path, encoding="utf-8") as f:
        note_text = f.read()

    print("=" * 70)
    print("HEALTHCARE TEXT EXTRACTION FOUNDATION — LIVE DEMONSTRATION")
    print("=" * 70)
    print(f"Input Document: {note_path} ({len(note_text)} code points)")
    print("-" * 70)

    # Initialize agent with DemoClinicalExtractor
    extractor = DemoClinicalExtractor()
    agent = build_text_agent(extractor=extractor)

    # Process request
    request = TextAgentRequest(
        text=note_text,
        options=TextAgentOptions(date_order="UNKNOWN"),
    )
    result = agent.process(request)

    print(f"Execution Status: {str(result.status).upper()}")
    print(f"Request ID:       {result.request_id}")
    print(f"Content SHA-256:  {result.content_sha256}")
    print(f"Candidates Found: {len(result.candidates)}")
    print("-" * 70)
    print("EXTRACTED & VALIDATED CLINICAL FACT CANDIDATES:")
    print("-" * 70)

    for idx, c in enumerate(result.candidates, 1):
        print(f"\n[Candidate #{idx}] ID: {c.candidate_id}")
        print(f"  Category:       {c.fact_type.value.upper()}")
        print(f"  Label:          {c.label_raw}")
        if c.value_raw:
            print(f"  Raw Value:      {c.value_raw} {c.unit_raw or ''}")
        if c.value_parsed and c.value_parsed.parse_status.value == "parsed":
            if c.value_parsed.value_kind == "scalar":
                print(
                    f"  Parsed Scalar:  {c.value_parsed.parsed_scalar} ({c.value_parsed.rule_identifier})"
                )
            elif c.value_parsed.value_kind == "ratio":
                print(
                    f"  Parsed Ratio:   {c.value_parsed.ratio_numerator} / {c.value_parsed.ratio_denominator} ({c.value_parsed.rule_identifier})"
                )
        if c.date_raw:
            print(
                f"  Observation Date: {c.date_raw} -> Parsed ISO: {c.date_parsed.parsed_iso_date if c.date_parsed else 'N/A'}"
            )
        if c.reference_range_raw:
            print(f"  Ref Range:      {c.reference_range_raw} | Flag: {c.abnormal_flag_raw}")
        print(f"  Assertion:      {c.assertion_status.value.upper()}")
        print(f"  Temporality:    {c.temporality.value}")
        print(
            f"  Experiencer:    {c.experiencer_type.value}"
            + (f" ({c.experiencer_relationship_raw})" if c.experiencer_relationship_raw else "")
        )
        print(f"  Reporting Role: {c.reporting_source_role.value}")
        print(f"  Review Status:  {c.review_status.value}")

        for ev in c.evidence_records:
            loc = ev.location
            span_slice = "N/A"
            start_c = getattr(loc, "start", None)
            end_c = getattr(loc, "end", None)
            if start_c is not None and end_c is not None:
                span_slice = note_text[start_c:end_c]
            print(
                f"  Evidence:       [{ev.status.value.upper()}] span=[{start_c}:{end_c}] in {ev.segment_id}"
            )
            print(f'                  Quote: "{ev.quote}"')
            print(f'                  Source Slice: "{span_slice}"')

    print("\n" + "=" * 70)
    print("PIPELINE AUDIT METADATA & PROVENANCE:")
    print("=" * 70)
    audit = result.audit
    print(f"  Schema Version:     {audit.schema_version}")
    print(f"  Provider Name:      {audit.provider} | Model: {audit.model_id}")
    print(f"  Execution Time:     {result.duration_ms:.2f} ms")
    print(f"  Content Digest:     {result.content_sha256}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
