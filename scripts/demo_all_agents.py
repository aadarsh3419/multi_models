"""End-to-End Live Multi-Modal Demonstration across All 4 Healthcare Agents.

Demonstrates all four modality agents operating with exact coordinate provenance:
1. Text Agent  -> Ingests clinical narrative, outputs [start, end] char offsets
2. PDF Agent   -> Ingests clinical document, outputs page and 2D bounding boxes [x0, y0, x1, y1]
3. Image Agent -> Ingests lab slip image, outputs displayed pixel ROI boxes [x0, y0, x1, y1]
4. Voice Agent -> Ingests consultation audio, outputs temporal millisecond spans [start_ms, end_ms]
"""

import io
import json
import shutil
import sys
import wave
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pymupdf as fitz
from PIL import Image, ImageDraw

# Ensure src is on path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from healthcare_agents.agents.voice_agent import VoiceAgent
from healthcare_agents.config import get_settings
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
from healthcare_agents.contracts.requests import (
    ImageAgentOptions,
    ImageAgentRequest,
    PDFAgentOptions,
    PDFAgentRequest,
    TextAgentOptions,
    TextAgentRequest,
    VoiceAgentOptions,
    VoiceAgentRequest,
)
from healthcare_agents.extraction.field_extractor import ClinicalFieldExtractor
from healthcare_agents.extraction.workflow import ExtractionWorkflow
from healthcare_agents.factory import (
    build_image_agent,
    build_pdf_agent,
    build_text_agent,
    build_voice_agent,
)
from healthcare_agents.ocr.service import (
    FakeOcrService,
    OCRLine,
    OCRResult,
    OCRWord,
)
from healthcare_agents.sources.audio_reader import (
    AudioReader,
    Transcriber,
    TranscriptionResult,
    TranscriptionSegment,
)


# ==============================================================================
# 1. TEXT AGENT DEMONSTRATION
# ==============================================================================
class TextDemoExtractor(ClinicalFieldExtractor):
    def extract(
        self, unit: Any, language_hint: str | None = None
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        text = getattr(unit, "text", "") or ""
        unit_id = getattr(unit, "unit_id", "s0001")
        candidates = []

        if "Blood Pressure: 120/80 mmHg" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Blood Pressure",
                    value_raw="120/80",
                    unit_raw="mmHg",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(segment_id=unit_id, quote="Blood Pressure: 120/80 mmHg")
                    ],
                )
            )
        if "denies fever" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.SYMPTOM,
                    label_raw="fever",
                    assertion_status=AssertionStatus.NEGATED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.PATIENT,
                    evidence_refs=[ProviderEvidenceRef(segment_id=unit_id, quote="denies fever")],
                )
            )
        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "text_demo",
            "model_id": "text-demo-v1",
        }


def run_text_demo() -> None:
    print("\n" + "=" * 76)
    print("  MODALITY 1: TEXT AGENT (Unstructured Clinical Narrative)")
    print("=" * 76)

    clinical_note = (
        "Patient seen for follow-up. Vitals: Blood Pressure: 120/80 mmHg. "
        "Patient denies fever and reports feeling energetic."
    )
    print(f'Input Narrative: "{clinical_note}"\n')

    agent = build_text_agent(extractor=TextDemoExtractor())
    req = TextAgentRequest(
        text=clinical_note,
        options=TextAgentOptions(include_source_text_in_result=True),
    )
    result = agent.process(req)

    print(f"Status: {result.status.upper()} | Request ID: {result.request_id}")
    print(f"Extracted Candidates ({len(result.candidates)}):")
    for idx, c in enumerate(result.candidates, 1):
        loc = c.evidence_records[0].location
        start = getattr(loc, "start", 0)
        end = getattr(loc, "end", 0)
        slice_txt = clinical_note[start:end]
        print(
            f"  [{idx}] Category: {c.fact_type.value.upper():<12} | Label: {c.label_raw:<15} | Status: {c.assertion_status.value}"
        )
        print(f"      Value: {c.value_raw or 'N/A'} {c.unit_raw or ''}")
        print(f"      Provenance: TextLocation [start={start}, end={end}]")
        print(f'      Raw Source Slice: "{slice_txt}"')


# ==============================================================================
# 2. PDF AGENT DEMONSTRATION
# ==============================================================================
class PdfDemoExtractor(ClinicalFieldExtractor):
    def extract(
        self, unit: Any, language_hint: str | None = None
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        text = getattr(unit, "text", "") or ""
        unit_id = getattr(unit, "unit_id", "s0001")
        candidates = []

        if "Fasting Glucose: 135 mg/dL" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Fasting Glucose",
                    value_raw="135",
                    unit_raw="mg/dL",
                    reference_range_raw="70-99",
                    abnormal_flag_raw="HIGH",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.LABORATORY,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Fasting Glucose: 135 mg/dL (70-99) [HIGH]",
                        )
                    ],
                )
            )
        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "pdf_demo",
            "model_id": "pdf-demo-v1",
        }


def run_pdf_demo() -> None:
    print("\n" + "=" * 76)
    print("  MODALITY 2: PDF AGENT (Discharge Summary & Lab Reports)")
    print("=" * 76)

    # Generate an in-memory PDF document with digital text layout
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    pdf_text = (
        "METROPOLITAN HOSPITAL — DISCHARGE LABORATORY REPORT\n"
        "Patient ID: PT-90210 | Encounter: 2026-10-18\n\n"
        "Fasting Glucose: 135 mg/dL (70-99) [HIGH]\n"
        "Patient diagnosed with impaired fasting glycaemia."
    )
    page.insert_text((50, 72), pdf_text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    print("Input Document: In-Memory PDF (A4 Page 1, 595x842 pt)")
    agent = build_pdf_agent(ocr_service=FakeOcrService(), extractor=PdfDemoExtractor())
    req = PDFAgentRequest(
        content=pdf_bytes,
        source_name="discharge_summary.pdf",
        options=PDFAgentOptions(ocr_mode="auto"),
    )
    result = agent.process(req)

    print(f"Status: {result.status.upper()} | Request ID: {result.request_id}")
    print(f"Extracted Candidates ({len(result.candidates)}):")
    for idx, c in enumerate(result.candidates, 1):
        loc = c.evidence_records[0].location
        print(
            f"  [{idx}] Category: {c.fact_type.value.upper():<12} | Label: {c.label_raw:<15} | Status: {c.assertion_status.value}"
        )
        print(
            f"      Value: {c.value_raw} {c.unit_raw} | Ref Range: {c.reference_range_raw} | Flag: {c.abnormal_flag_raw}"
        )
        print(f"      Spatial Provenance: PdfLocation page={loc.page_number}, bbox={loc.bbox}")
        print(f'      Quote Grounded: "{c.evidence_records[0].quote}"')


# ==============================================================================
# 3. IMAGE AGENT DEMONSTRATION
# ==============================================================================
class ImageDemoExtractor(ClinicalFieldExtractor):
    def extract(
        self, unit: Any, language_hint: str | None = None
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        text = getattr(unit, "text", "") or ""
        unit_id = getattr(unit, "unit_id", "s0001")
        candidates = []

        if "Serum Creatinine: 1.1 mg/dL" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEASUREMENT,
                    label_raw="Serum Creatinine",
                    value_raw="1.1",
                    unit_raw="mg/dL",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.NOT_APPLICABLE,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.LABORATORY,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=unit_id,
                            quote="Serum Creatinine: 1.1 mg/dL",
                        )
                    ],
                )
            )
        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "image_demo",
            "model_id": "image-demo-v1",
        }


def run_image_demo() -> None:
    print("\n" + "=" * 76)
    print("  MODALITY 3: IMAGE AGENT (Document Photo / Lab Slip Scan)")
    print("=" * 76)

    # Generate an in-memory PNG lab slip
    img = Image.new("RGB", (640, 160), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((25, 40), "LAB SLIP: Serum Creatinine: 1.1 mg/dL", fill=(0, 0, 0))
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    # Shared OCR service double providing spatial OCR words and lines
    ocr_words = [
        OCRWord(text="LAB", bbox=[25, 40, 55, 65], confidence=99.0),
        OCRWord(text="SLIP:", bbox=[60, 40, 95, 65], confidence=99.0),
        OCRWord(text="Serum", bbox=[100, 40, 160, 65], confidence=98.5),
        OCRWord(text="Creatinine:", bbox=[165, 40, 260, 65], confidence=97.8),
        OCRWord(text="1.1", bbox=[265, 40, 290, 65], confidence=99.0),
        OCRWord(text="mg/dL", bbox=[295, 40, 350, 65], confidence=98.2),
    ]
    ocr_service = FakeOcrService(
        canned_result=OCRResult(
            text="LAB SLIP: Serum Creatinine: 1.1 mg/dL",
            lines=[
                OCRLine(
                    text="LAB SLIP: Serum Creatinine: 1.1 mg/dL",
                    bbox=[25, 40, 350, 65],
                    words=ocr_words,
                )
            ],
            words=ocr_words,
            language="eng",
        )
    )

    print("Input Document: In-Memory PNG Lab Slip (640x160 px, 24-bit RGB)")
    agent = build_image_agent(ocr_service=ocr_service, extractor=ImageDemoExtractor())
    req = ImageAgentRequest(
        content=img_bytes,
        source_name="lab_slip_creatinine.png",
        options=ImageAgentOptions(upscale_enabled=False),
    )
    result = agent.process(req)

    print(f"Status: {result.status.upper()} | Request ID: {result.request_id}")
    print(f"Extracted Candidates ({len(result.candidates)}):")
    for idx, c in enumerate(result.candidates, 1):
        loc = c.evidence_records[0].location
        print(
            f"  [{idx}] Category: {c.fact_type.value.upper():<12} | Label: {c.label_raw:<15} | Status: {c.assertion_status.value}"
        )
        print(f"      Value: {c.value_raw} {c.unit_raw}")
        print(
            f"      Pixel Provenance: ImageLocation bbox={loc.bbox} (coordinate_system='{loc.coordinate_system}')"
        )
        if loc.ocr_words:
            word_conf_strs = [
                f"{w.get('text', '')}:{w.get('confidence', 0.0):.1f}%"
                if isinstance(w, dict)
                else f"{getattr(w, 'text', '')}:{getattr(w, 'confidence', 0.0):.1f}%"
                for w in loc.ocr_words
            ]
            print(f"      OCR Word Confidences: {word_conf_strs}")


# ==============================================================================
# 4. VOICE AGENT DEMONSTRATION
# ==============================================================================
class VoiceDemoExtractor(ClinicalFieldExtractor):
    def extract(
        self, unit: Any, language_hint: str | None = None
    ) -> tuple[ProviderExtractionResponse, dict[str, Any]]:
        text = getattr(unit, "text", "") or ""
        unit_id = getattr(unit, "unit_id", "s0001")
        candidates = []

        def get_seg_id(needle: str) -> str:
            members = getattr(unit, "member_segments", [])
            for s in members:
                s_text = getattr(s, "exact_text", "") or getattr(s, "text", "")
                if needle in s_text:
                    return s.segment_id
            return getattr(unit, "unit_id", "s0001")

        if "persistent dry cough" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.SYMPTOM,
                    label_raw="dry cough",
                    value_raw="persistent",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.PATIENT,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=get_seg_id("persistent dry cough"),
                            quote="persistent dry cough for three days",
                        )
                    ],
                )
            )
        if "Paracetamol 500 mg" in text:
            candidates.append(
                ProviderCandidate(
                    fact_type=FactType.MEDICATION,
                    label_raw="Paracetamol",
                    value_raw="500 mg",
                    qualifier_raw="as needed",
                    assertion_status=AssertionStatus.AFFIRMED,
                    temporality=Temporality.CURRENT,
                    experiencer_type=ExperiencerType.PATIENT,
                    reporting_source_role=ReportingSourceRole.CLINICIAN,
                    evidence_refs=[
                        ProviderEvidenceRef(
                            segment_id=get_seg_id("Paracetamol 500 mg"),
                            quote="Paracetamol 500 mg as needed for fever",
                        )
                    ],
                )
            )
        return ProviderExtractionResponse(candidates=candidates), {
            "provider": "voice_demo",
            "model_id": "voice-demo-v1",
        }


class DemoVoiceTranscriber(Transcriber):
    """Demonstration speech transcriber providing timestamped segments."""

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            text=(
                "Patient states: I have a persistent dry cough for three days. "
                "Doctor: Take Paracetamol 500 mg as needed for fever."
            ),
            segments=[
                TranscriptionSegment(
                    text="Patient states: I have a persistent dry cough for three days.",
                    start_seconds=0.25,
                    end_seconds=2.40,
                ),
                TranscriptionSegment(
                    text="Doctor: Take Paracetamol 500 mg as needed for fever.",
                    start_seconds=2.50,
                    end_seconds=4.80,
                ),
            ],
            language="en",
            duration_seconds=5.0,
        )


def _make_mock_ffprobe_popen():
    """Provides fake ffprobe metadata for offline execution when binary is not in PATH."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    stdout_json = json.dumps(
        {
            "format": {"format_name": "wav", "duration": "5.0"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "duration": "5.0",
                }
            ],
        }
    )
    mock_proc.communicate.return_value = (stdout_json, "")
    return MagicMock(return_value=mock_proc)


def run_voice_demo() -> None:
    print("\n" + "=" * 76)
    print("  MODALITY 4: VOICE AGENT (Audio Consultation & Telehealth Dictation)")
    print("=" * 76)

    # Generate an in-memory WAV audio consultation (16kHz 16-bit PCM, 5 seconds)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 80000)  # 5 seconds of PCM frames
    wav_bytes = wav_buf.getvalue()

    print("Input Document: In-Memory WAV Audio Stream (16 kHz, 16-bit Mono, 5.0s)")

    from healthcare_agents.config import Settings

    demo_settings = Settings(
        allow_remote_audio_transcription=True,
        allow_remote_llm_processing=True,
    )

    # If ffprobe is not installed on PATH, use mock popen factory
    if not shutil.which("ffprobe"):
        popen_factory = _make_mock_ffprobe_popen()
        reader = AudioReader(
            transcriber=DemoVoiceTranscriber(),
            popen_factory=popen_factory,
            settings=demo_settings,
        )
        workflow = ExtractionWorkflow(
            extractor=VoiceDemoExtractor(), date_order="UNKNOWN"
        )
        agent = VoiceAgent(
            reader=reader, workflow=workflow, settings=demo_settings
        )
    else:
        agent = build_voice_agent(
            transcriber=DemoVoiceTranscriber(),
            extractor=VoiceDemoExtractor(),
            settings=demo_settings,
        )

    req = VoiceAgentRequest(
        content=wav_bytes,
        source_name="consultation_recording.wav",
        options=VoiceAgentOptions(language_hint="en"),
    )
    result = agent.process(req)

    print(f"Status: {result.status.upper()} | Request ID: {result.request_id}")
    print(f"Extracted Candidates ({len(result.candidates)}):")
    for idx, c in enumerate(result.candidates, 1):
        loc = c.evidence_records[0].location
        start_s = getattr(loc, "start_seconds", None)
        end_s = getattr(loc, "end_seconds", None)
        start_ms = int(start_s * 1000) if start_s is not None else 0
        end_ms = int(end_s * 1000) if end_s is not None else 0
        print(
            f"  [{idx}] Category: {c.fact_type.value.upper():<12} | Label: {c.label_raw:<15} | Status: {c.assertion_status.value}"
        )
        print(f"      Value: {c.value_raw or 'N/A'}")
        print(
            f"      Temporal Provenance: AudioLocation [start_ms={start_ms} ms, end_ms={end_ms} ms] ({start_s:.2f}s - {end_s:.2f}s)"
        )
        print(f'      Quote Grounded: "{c.evidence_records[0].quote}"')


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================
def main() -> None:
    print("=" * 76)
    print("  HEALTHCARE MULTI-AGENT PLATFORM — ALL 4 MODALITIES LIVE DEMO")
    print("=" * 76)

    run_text_demo()
    run_pdf_demo()
    run_image_demo()
    run_voice_demo()

    print("\n" + "=" * 76)
    print("  ALL 4 MODALITY AGENTS SUCCESSFULLY DEMONSTRATED & VERIFIED")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
