"""
Unit tests for PdfReader: in-memory ingestion, layout awareness, tables, selective OCR, and deduplication.
Offline only: all test documents generated dynamically in-memory via PyMuPDF.
"""

import io
from uuid import uuid4

import pymupdf
from PIL import Image

from healthcare_agents.config import Settings
from healthcare_agents.contracts.errors import ErrorCode, WarningCode
from healthcare_agents.contracts.requests import PDFAgentOptions
from healthcare_agents.ocr.service import (
    FakeOcrService,
    OCRLine,
    OCRResult,
    OcrTimeoutError,
    OcrUnavailableError,
    OCRWord,
)
from healthcare_agents.sources.pdf_reader import PdfReader, compute_bbox_iou, plan_page_ocr


def make_native_pdf(pages_text: list[list[str]]) -> bytes:
    """Creates an in-memory native text PDF."""
    doc = pymupdf.open()
    for page_lines in pages_text:
        page = doc.new_page(width=612, height=792)  # 8.5 x 11 inches
        y = 72
        for line in page_lines:
            page.insert_text((72, y), line, fontsize=12)
            y += 24
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_scanned_pdf(img_width: int = 500, img_height: int = 700) -> bytes:
    """Creates an in-memory PDF with an embedded raster image filling most of the page."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # Generate in-memory PNG
    img = Image.new("RGB", (img_width, img_height), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Insert image taking up 80% of page area
    page.insert_image(pymupdf.Rect(50, 50, 562, 742), stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_encrypted_pdf() -> bytes:
    """Creates a password-protected PDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Confidential clinical data", fontsize=12)
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="password123",
        owner_pw="owner123",
    )
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Tests: Pure Planning Function & IoU
# ---------------------------------------------------------------------------


def test_compute_bbox_iou():
    # Identical boxes
    box = [10.0, 10.0, 50.0, 50.0]
    assert compute_bbox_iou(box, box) == 1.0

    # Non-overlapping
    box2 = [60.0, 60.0, 100.0, 100.0]
    assert compute_bbox_iou(box, box2) == 0.0

    # 50% overlap
    b1 = [0.0, 0.0, 10.0, 10.0]  # area 100
    b2 = [5.0, 0.0, 15.0, 10.0]  # area 100, intersection 5x10=50, union=150
    assert abs(compute_bbox_iou(b1, b2) - (50.0 / 150.0)) < 1e-4

    # None or invalid boxes
    assert compute_bbox_iou(None, box) == 0.0
    assert compute_bbox_iou(box, [10.0]) == 0.0


def test_plan_page_ocr_modes():
    settings = Settings(
        pdf_min_native_alnum=40,
        pdf_image_region_min_fraction=0.08,
        pdf_fullpage_raster_fraction=0.65,
    )

    # 1. never mode
    plan = plan_page_ocr(1, 0, [], 0.0, "never", settings)
    assert plan["decision"] == "none"
    assert "never" in plan["reasons"][0]

    # 2. always mode
    plan = plan_page_ocr(1, 100, [], 0.0, "always", settings)
    assert plan["decision"] == "full_page"
    assert "always" in plan["reasons"][0]

    # 3. auto mode - text rich, no images -> none
    plan = plan_page_ocr(1, 150, [], 0.0, "auto", settings)
    assert plan["decision"] == "none"

    # 4. auto mode - text deficient, full-page raster image -> full_page
    large_img = {"rect": [50, 50, 550, 750], "fraction": 0.85}
    plan = plan_page_ocr(1, 10, [large_img], 0.85, "auto", settings)
    assert plan["decision"] == "full_page"

    # 5. auto mode - text rich, but contains substantial image region -> selective_regions
    small_img = {"rect": [50, 50, 200, 200], "fraction": 0.15}
    plan = plan_page_ocr(1, 150, [small_img], 0.15, "auto", settings)
    assert plan["decision"] == "selective_regions"
    assert len(plan["targets"]) == 1


# ---------------------------------------------------------------------------
# Tests: Native Digital PDF Extraction
# ---------------------------------------------------------------------------


def test_read_bundle_native_digital_pdf():
    pdf_bytes = make_native_pdf(
        [
            ["Patient Name: John Doe", "Diagnosis: Type 2 Diabetes"],
            ["Prescription: Metformin 500mg BID", "Follow-up: 3 months"],
        ]
    )

    fake_ocr = FakeOcrService()
    reader = PdfReader(ocr_service=fake_ocr)
    req_id = uuid4()

    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, req_id)

    assert bundle is not None
    assert status == "success"
    assert len(errors) == 0
    assert len(warnings) == 0
    assert bundle.request_id == req_id
    assert bundle.source_type == "pdf"
    assert bundle.safe_metadata["page_count"] == 2
    assert len(bundle.ordered_segments) >= 4

    # Spatial provenance checks on segments
    for seg in bundle.ordered_segments:
        loc = seg.source_location
        assert loc.page_number in (1, 2)
        assert loc.coordinate_system == "pdf_points_top_left"
        assert "native" in loc.extraction_method
        assert loc.bbox is not None
        assert len(loc.bbox) == 4
        assert loc.bbox[2] > loc.bbox[0]
        assert loc.bbox[3] > loc.bbox[1]

    # No OCR calls made for text-rich native PDF
    assert fake_ocr.call_count == 0


def test_read_bundle_page_rotations():
    doc = pymupdf.open()
    for rot in [0, 90, 180, 270]:
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (100, 100),
            f"Rotation {rot} clinical examination notes: Patient has stable vitals.",
            fontsize=12,
        )
        page.set_rotation(rot)
    pdf_bytes = doc.tobytes()
    doc.close()

    fake_ocr = FakeOcrService()
    reader = PdfReader(ocr_service=fake_ocr)
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    assert bundle is not None
    assert status == "success"
    assert len(bundle.ordered_segments) == 4
    for idx, seg in enumerate(bundle.ordered_segments):
        assert f"Rotation {[0, 90, 180, 270][idx]}" in seg.text
        assert seg.source_location.page_number == idx + 1
        assert seg.source_location.bbox is not None


# ---------------------------------------------------------------------------
# Tests: Scanned PDF & OCR Execution
# ---------------------------------------------------------------------------


def test_read_bundle_scanned_pdf_with_fake_ocr():
    pdf_bytes = make_scanned_pdf()

    canned_ocr = OCRResult(
        text="Blood Pressure: 130/85 mmHg\nPulse: 72 bpm",
        lines=[
            OCRLine(
                text="Blood Pressure: 130/85 mmHg",
                bbox=[50, 50, 300, 80],
                words=[
                    OCRWord(text="Blood", bbox=[50, 50, 100, 80], confidence=98.0),
                    OCRWord(text="Pressure:", bbox=[105, 50, 180, 80], confidence=95.0),
                    OCRWord(text="130/85", bbox=[185, 50, 240, 80], confidence=99.0),
                    OCRWord(text="mmHg", bbox=[245, 50, 300, 80], confidence=92.0),
                ],
            ),
            OCRLine(
                text="Pulse: 72 bpm",
                bbox=[50, 90, 200, 120],
                words=[
                    OCRWord(text="Pulse:", bbox=[50, 90, 110, 120], confidence=96.0),
                    OCRWord(text="72", bbox=[115, 90, 140, 120], confidence=98.0),
                    OCRWord(text="bpm", bbox=[145, 90, 200, 120], confidence=94.0),
                ],
            ),
        ],
    )

    fake_ocr = FakeOcrService(canned_result=canned_ocr)
    reader = PdfReader(ocr_service=fake_ocr)
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    assert bundle is not None
    assert status == "success"
    assert fake_ocr.call_count == 1
    assert len(bundle.ordered_segments) == 2

    seg1 = bundle.ordered_segments[0]
    assert "Blood Pressure: 130/85 mmHg" in seg1.text
    assert "ocr" in seg1.source_location.extraction_method
    assert seg1.source_location.page_number == 1
    assert seg1.source_location.ocr_words is not None
    assert len(seg1.source_location.ocr_words) == 4
    assert seg1.source_location.ocr_words[0]["text"] == "Blood"
    assert seg1.source_location.ocr_words[0]["confidence"] == 98.0


# ---------------------------------------------------------------------------
# Tests: OCR Resilience, Timeouts, and Warnings
# ---------------------------------------------------------------------------


def test_ocr_timeout_emits_warning():
    pdf_bytes = make_scanned_pdf()

    fake_ocr = FakeOcrService(raise_on_ocr=OcrTimeoutError("OCR execution timed out after 60s"))
    reader = PdfReader(ocr_service=fake_ocr)

    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    # When OCR fails on a scanned page and no text is extracted anywhere, returns failed
    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.NO_USABLE_SOURCE_CONTENT.value for e in errors)
    assert any(w.code == ErrorCode.OCR_TIMEOUT.value for w in warnings)


def test_ocr_unavailable_emits_warning():
    pdf_bytes = make_scanned_pdf()

    fake_ocr = FakeOcrService(raise_on_ocr=OcrUnavailableError("Tesseract not installed"))
    reader = PdfReader(ocr_service=fake_ocr)

    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    assert bundle is None
    assert status == "failed"
    assert any(w.code == ErrorCode.OCR_UNAVAILABLE.value for w in warnings)


# ---------------------------------------------------------------------------
# Tests: Deduplication Native vs OCR (IoU)
# ---------------------------------------------------------------------------


def test_native_ocr_deduplication():
    # Create PDF with native text
    pdf_bytes = make_native_pdf([["Hemoglobin A1c: 6.8%"]])

    # OCR returns identical text at exact overlapping coordinates in 200 DPI pixel space
    # (72pt * 200/72 = 200px, 59pt * 200/72 = 164px, 194pt * 200/72 = 539px, 76pt * 200/72 = 210px)
    canned_ocr = OCRResult(
        text="Hemoglobin A1c: 6.8%",
        lines=[
            OCRLine(
                text="Hemoglobin A1c: 6.8%",
                bbox=[200, 164, 539, 210],
                words=[
                    OCRWord(text="Hemoglobin", bbox=[200, 164, 380, 210]),
                    OCRWord(text="A1c:", bbox=[390, 164, 460, 210]),
                    OCRWord(text="6.8%", bbox=[470, 164, 539, 210]),
                ],
            )
        ],
    )

    fake_ocr = FakeOcrService(canned_result=canned_ocr)
    reader = PdfReader(ocr_service=fake_ocr)

    # Force always OCR mode so OCR runs even with native text
    options = PDFAgentOptions(ocr_mode="always")
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(
        pdf_bytes, uuid4(), options=options
    )

    assert bundle is not None
    # Verify exact duplicate was suppressed: only 1 segment exists (native)
    assert len(bundle.ordered_segments) == 1
    assert "native" in bundle.ordered_segments[0].source_location.extraction_method


def test_native_ocr_different_text_emits_warning():
    pdf_bytes = make_native_pdf([["Fasting Glucose: 110 mg/dL"]])

    # OCR reads differing text at overlapping pixel coordinates
    # (72pt * 200/72 = 200px, 59pt * 200/72 = 164px, 220pt * 200/72 = 611px, 76pt * 200/72 = 210px)
    canned_ocr = OCRResult(
        text="Fasting Glucose: 170 mg/dL",
        lines=[
            OCRLine(
                text="Fasting Glucose: 170 mg/dL",
                bbox=[200, 164, 611, 210],
                words=[],
            )
        ],
    )

    fake_ocr = FakeOcrService(canned_result=canned_ocr)
    reader = PdfReader(ocr_service=fake_ocr)

    options = PDFAgentOptions(ocr_mode="always")
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(
        pdf_bytes, uuid4(), options=options
    )

    assert bundle is not None
    # Both segments preserved when text differs
    assert len(bundle.ordered_segments) == 2
    # Warning emitted
    assert any(w.code == WarningCode.POSSIBLE_NATIVE_OCR_DUPLICATE.value for w in warnings)


# ---------------------------------------------------------------------------
# Tests: Error Boundaries
# ---------------------------------------------------------------------------


def test_file_too_large_error():
    fake_ocr = FakeOcrService()
    settings = Settings(max_pdf_bytes=100)  # low threshold
    reader = PdfReader(ocr_service=fake_ocr, settings=settings)

    pdf_bytes = make_native_pdf([["Test line"]])
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.FILE_TOO_LARGE.value for e in errors)


def test_unsupported_file_type_error():
    fake_ocr = FakeOcrService()
    reader = PdfReader(ocr_service=fake_ocr)

    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(
        b"NOT A PDF FILE DATA", uuid4()
    )

    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.UNSUPPORTED_FILE_TYPE.value for e in errors)


def test_corrupt_pdf_unreadable_error():
    fake_ocr = FakeOcrService()
    reader = PdfReader(ocr_service=fake_ocr)

    corrupt_bytes = b"%PDF-1.4\nCorrupt garbage data that cannot be parsed"
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(corrupt_bytes, uuid4())

    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.PDF_UNREADABLE.value for e in errors)


def test_encrypted_pdf_error():
    fake_ocr = FakeOcrService()
    reader = PdfReader(ocr_service=fake_ocr)

    enc_bytes = make_encrypted_pdf()
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(enc_bytes, uuid4())

    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.PDF_ENCRYPTED.value for e in errors)


def test_page_limit_exceeded_error():
    fake_ocr = FakeOcrService()
    settings = Settings(max_pdf_pages=2)  # Limit 2 pages
    reader = PdfReader(ocr_service=fake_ocr, settings=settings)

    # Make 3 pages
    pdf_bytes = make_native_pdf([["Page 1"], ["Page 2"], ["Page 3"]])
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    assert bundle is None
    assert status == "failed"
    assert any(e.code == ErrorCode.PAGE_LIMIT_EXCEEDED.value for e in errors)


def test_render_pixel_budget_exhaustion():
    fake_ocr = FakeOcrService()
    # Set max render pixels very low so even min DPI (100) exceeds budget
    settings = Settings(max_render_pixels=1000)
    reader = PdfReader(ocr_service=fake_ocr, settings=settings)

    pdf_bytes = make_scanned_pdf(img_width=400, img_height=600)
    bundle, status, errors, warnings, stage_res, plans = reader.read_bundle(pdf_bytes, uuid4())

    # OCR skipped due to budget
    assert any(w.code == WarningCode.OCR_SKIPPED_RENDER_BUDGET.value for w in warnings)
