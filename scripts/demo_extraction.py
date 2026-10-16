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

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from healthcare_agents.contracts import (
    EvidenceReference,
    ExtractionCandidateRaw,
    ExtractionResponse,
    TextAgentRequest,
)
from healthcare_agents.extractor_interface import (
    ClinicalFieldExtractor,
    ExtractorCallMetadata,
)
from healthcare_agents.prompt_templates import PROMPT_VERSION, get_prompt_template_hash
from healthcare_agents.source_text import SourceSegment
from healthcare_agents.text_agent import TextAgent
from healthcare_agents.workflow import ExtractionWorkflow


class DemoClinicalExtractor(ClinicalFieldExtractor):
    """Demonstration extractor that extracts clinically grounded candidates from sample note."""

    def extract(
        self,
        segments: list[SourceSegment],
        language_hint: str | None = None,
    ) -> tuple[ExtractionResponse, ExtractorCallMetadata]:
        candidates: list[ExtractionCandidateRaw] = []

        # Find relevant segments for quotation grounding
        seg_map = {s.segment_id: s.exact_text for s in segments}

        for seg_id, text in seg_map.items():
            if "Blood Pressure: 128/82 mmHg" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="measurement",
                        label_raw="Blood Pressure",
                        value_raw="128/82",
                        unit_raw="mmHg",
                        assertion_status="affirmed",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="clinician",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Blood Pressure: 128/82 mmHg",
                            )
                        ],
                    )
                )
            if "Heart Rate: 72 bpm" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="measurement",
                        label_raw="Heart Rate",
                        value_raw="72",
                        unit_raw="bpm",
                        assertion_status="affirmed",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="clinician",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Heart Rate: 72 bpm",
                            )
                        ],
                    )
                )
            if "Fasting Blood Glucose: 118 mg/dL" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="measurement",
                        label_raw="Fasting Blood Glucose",
                        value_raw="118",
                        unit_raw="mg/dL",
                        reference_range_raw="70-99",
                        abnormal_flag_raw="High",
                        date_raw="2025-04-03",
                        assertion_status="affirmed",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="laboratory",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Fasting Blood Glucose: 118 mg/dL (Reference Range: 70-99 mg/dL) [Flag: High]",
                            )
                        ],
                    )
                )
            if "Hemoglobin A1c: 6.8 %" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="measurement",
                        label_raw="Hemoglobin A1c",
                        value_raw="6.8",
                        unit_raw="%",
                        reference_range_raw="4.0-5.6",
                        abnormal_flag_raw="H",
                        date_raw="2025-04-03",
                        assertion_status="affirmed",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="laboratory",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Hemoglobin A1c: 6.8 % (Reference Range: 4.0-5.6 %) [Flag: H]",
                            )
                        ],
                    )
                )
            if "Father had coronary artery disease at age 52" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="family_history",
                        label_raw="coronary artery disease",
                        assertion_status="affirmed",
                        temporality="past",
                        experiencer_type="family_member",
                        experiencer_relationship_raw="father",
                        reporting_source_role="patient",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Father had coronary artery disease at age 52",
                            )
                        ],
                    )
                )
            if "Metformin 500 mg oral twice daily for 6 months" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="medication",
                        label_raw="Metformin",
                        dose_raw="500 mg",
                        route_raw="oral",
                        frequency_raw="twice daily",
                        duration_raw="6 months",
                        assertion_status="affirmed",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="clinician",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="Metformin 500 mg oral twice daily for 6 months",
                            )
                        ],
                    )
                )
            if "denies chest pain or shortness of breath" in text:
                candidates.append(
                    ExtractionCandidateRaw(
                        fact_type="symptom",
                        label_raw="chest pain",
                        assertion_status="negated",
                        temporality="current",
                        experiencer_type="patient",
                        reporting_source_role="patient",
                        evidence_refs=[
                            EvidenceReference(
                                segment_id=seg_id,
                                quote="denies chest pain or shortness of breath",
                            )
                        ],
                    )
                )

        resp = ExtractionResponse(candidates=candidates)
        meta = ExtractorCallMetadata(
            provider_name="demo_extractor",
            model_id="clinical-demo-v1",
            response_id="demo_resp_001",
            prompt_version=PROMPT_VERSION,
            prompt_hash=get_prompt_template_hash(),
            input_tokens=420,
            output_tokens=310,
            attempts_made=1,
            duration_ms=5.4,
        )
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

    # Initialize workflow with DemoClinicalExtractor
    extractor = DemoClinicalExtractor()
    workflow = ExtractionWorkflow(
        extractor=extractor,
        date_order="UNKNOWN",
        max_llm_calls_per_request=8,
    )
    agent = TextAgent(workflow=workflow)

    # Process request
    request = TextAgentRequest(text=note_text)
    result = agent.process(request)

    print(f"Execution Status: {result.status.value.upper()}")
    print(f"Request ID:       {result.request_id}")
    print(f"Content SHA-256:  {result.audit_metadata.source_content_hash}")
    print(f"Duration:         {result.duration_ms:.2f} ms")
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
        if c.dose_raw or c.frequency_raw:
            print(
                f"  Medication:     Dose: {c.dose_raw}, Route: {c.route_raw}, Freq: {c.frequency_raw}, Duration: {c.duration_raw}"
            )
        print(f"  Assertion:      {c.assertion_status.value.upper()}")
        print(f"  Temporality:    {c.temporality.value}")
        print(
            f"  Experiencer:    {c.experiencer_type.value}"
            + (f" ({c.experiencer_relationship_raw})" if c.experiencer_relationship_raw else "")
        )
        print(f"  Reporting Role: {c.reporting_source_role.value}")
        print(f"  Review Status:  {c.review_status.value}")

        for ev in c.evidence_refs:
            span_slice = (
                note_text[ev.start_offset : ev.end_offset] if ev.start_offset is not None else "N/A"
            )
            print(
                f"  Evidence:       [{ev.status.value.upper()}] span=[{ev.start_offset}:{ev.end_offset}] in {ev.segment_id}"
            )
            print(f'                  Quote: "{ev.quote}"')
            print(f'                  Source Slice: "{span_slice}"')

    print("\n" + "=" * 70)
    print("PIPELINE AUDIT METADATA & PROVENANCE:")
    print("=" * 70)
    audit = result.audit_metadata
    print(f"  Schema Version:     {result.schema_version}")
    print(
        f"  Prompt Version:     {audit.prompt_version} (SHA-256: {audit.prompt_template_hash[:16]}...)"
    )
    print(f"  Provider Name:      {audit.provider_name} | Model: {audit.model_id}")
    print(f"  Token Usage:        {audit.token_usage}")
    print(f"  Provider Attempts:  {audit.actual_provider_attempt_count}")
    print("  Stages Completed:")
    for stage_name, outcome in result.stage_outcomes.items():
        print(
            f"    - {stage_name:<20}: {outcome.status.upper()} ({outcome.duration_ms:.2f} ms) {outcome.message or ''}"
        )
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
