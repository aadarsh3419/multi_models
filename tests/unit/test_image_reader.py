"""
Unit tests for ImageReader, Pillow safety boundaries, and OCR coordinate mapping.
"""

import io
from uuid import uuid4

from PIL import Image

from healthcare_agents.config import Settings
from healthcare_agents.contracts.errors import ErrorCode, WarningCode
from healthcare_agents.contracts.locations import ImageLocation
from healthcare_agents.contracts.requests import ImageAgentOptions
from healthcare_agents.ocr.service import (
    FakeOcrService,
    OCRLine,
    OCRResult,
    OcrTimeoutError,
    OcrUnavailableError,
    OCRWord,
)
from healthcare_agents.sources.image_reader import ImageReader


def _make_image_bytes(
    width: int = 1200,
    height: int = 800,
    fmt: str = "PNG",
    orientation: int = 1,
) -> bytes:
    """Helper creating memory-only image bytes with optional EXIF orientation."""
    img = Image.new("RGB", (width, height), color="white")
    buf = io.BytesIO()
    if orientation != 1 and fmt.upper() in ("JPEG", "WEBP"):
        exif = img.getexif()
        exif[274] = orientation
        img.save(buf, format=fmt, exif=exif)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def test_image_reader_success_scale_1():
    """Verify standard image reading without upscaling (scale = 1.0)."""
    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text="Fasting Blood Glucose 110 mg/dL",
            lines=[
                OCRLine(
                    text="Fasting Blood Glucose 110 mg/dL",
                    bbox=[50, 100, 450, 140],
                    words=[
                        OCRWord(text="Fasting", bbox=[50, 100, 120, 140], confidence=98.0),
                        OCRWord(text="Blood", bbox=[125, 100, 180, 140], confidence=97.0),
                        OCRWord(text="Glucose", bbox=[185, 100, 260, 140], confidence=99.0),
                        OCRWord(text="110", bbox=[270, 100, 310, 140], confidence=96.0),
                        OCRWord(text="mg/dL", bbox=[315, 100, 380, 140], confidence=95.0),
                    ],
                )
            ],
            words=[],
        )
    )

    settings = Settings(image_upscale_min_side=500)
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    content = _make_image_bytes(width=1200, height=800, fmt="PNG")
    req_id = uuid4()

    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=req_id,
        options=ImageAgentOptions(),
    )

    assert status == "success"
    assert bundle is not None
    assert len(errors) == 0
    assert len(warnings_list) == 0
    assert len(bundle.ordered_segments) == 1

    seg = bundle.ordered_segments[0]
    assert seg.text == "Fasting Blood Glucose 110 mg/dL"
    assert isinstance(seg.source_location, ImageLocation)
    assert seg.source_location.bbox == [50.0, 100.0, 450.0, 140.0]
    assert seg.source_location.coordinate_system == "pixel_top_left"
    assert seg.source_location.original_width == 1200
    assert seg.source_location.original_height == 800
    assert details["upscale_factor"] == 1.0


def test_image_reader_coordinate_round_trip_with_upscaling():
    """
    Verify coordinate inversion when upscaling is triggered.
    Image 400x300, min_side=300 < upscale_min_side=600 -> scale = 2.0.
    OCR outputs bbox [100, 200, 400, 300] in scaled space -> mapped to [50.0, 100.0, 200.0, 150.0].
    """
    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text="Hemoglobin 13.5 g/dL",
            lines=[
                OCRLine(
                    text="Hemoglobin 13.5 g/dL",
                    bbox=[100, 200, 400, 300],
                    words=[
                        OCRWord(text="Hemoglobin", bbox=[100, 200, 250, 300], confidence=95.0),
                        OCRWord(text="13.5", bbox=[260, 200, 320, 300], confidence=96.0),
                        OCRWord(text="g/dL", bbox=[330, 200, 400, 300], confidence=94.0),
                    ],
                )
            ],
            words=[],
        )
    )

    settings = Settings(image_upscale_min_side=600)
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    content = _make_image_bytes(width=400, height=300, fmt="PNG")
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(upscale_enabled=True),
    )

    assert status == "success"
    assert bundle is not None
    assert details["upscale_factor"] == 2.0  # 600 / 300 = 2.0

    seg = bundle.ordered_segments[0]
    loc = seg.source_location
    assert isinstance(loc, ImageLocation)
    # Scaled coordinates [100, 200, 400, 300] divided by 2.0:
    assert loc.bbox == [50.0, 100.0, 200.0, 150.0]
    assert loc.original_width == 400
    assert loc.original_height == 300


def test_image_reader_exif_orientation_handling():
    """Verify EXIF orientation != 1 results in oriented_pixel_top_left coordinate system."""
    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text="Pulse Rate 72 bpm",
            lines=[
                OCRLine(
                    text="Pulse Rate 72 bpm",
                    bbox=[20, 30, 180, 60],
                    words=[
                        OCRWord(text="Pulse", bbox=[20, 30, 60, 60], confidence=90.0),
                        OCRWord(text="Rate", bbox=[65, 30, 100, 60], confidence=90.0),
                        OCRWord(text="72", bbox=[105, 30, 130, 60], confidence=90.0),
                        OCRWord(text="bpm", bbox=[135, 30, 180, 60], confidence=90.0),
                    ],
                )
            ],
            words=[],
        )
    )

    settings = Settings(image_upscale_min_side=0)  # No upscaling
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    # Orientation 6 = 90 deg rotation
    content = _make_image_bytes(width=300, height=500, fmt="JPEG", orientation=6)
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert status == "success"
    assert bundle is not None
    seg = bundle.ordered_segments[0]
    loc = seg.source_location
    assert isinstance(loc, ImageLocation)
    assert loc.coordinate_system == "oriented_pixel_top_left"
    assert loc.exif_orientation == 6
    # After 90 deg transposition, width and height swap:
    assert loc.oriented_width == 500
    assert loc.oriented_height == 300


def test_image_reader_file_size_precheck():
    """Verify file size exceeding max_image_bytes is rejected with FILE_TOO_LARGE."""
    fake_ocr = FakeOcrService()
    settings = Settings(max_image_bytes=100)  # Very small threshold
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    content = _make_image_bytes(width=200, height=200, fmt="PNG")  # > 100 bytes
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.FILE_TOO_LARGE.value


def test_image_reader_empty_content_rejection():
    """Verify empty byte content is rejected with INVALID_INPUT."""
    fake_ocr = FakeOcrService()
    reader = ImageReader(ocr_service=fake_ocr)

    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=b"",
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.INVALID_INPUT.value


def test_image_reader_pixel_limit_check_before_load():
    """Verify pixel limit check triggers before load() when dimensions exceed max_image_pixels."""
    fake_ocr = FakeOcrService()
    settings = Settings(max_image_pixels=10_000)  # 10,000 pixels maximum
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    # 200 x 100 = 20,000 pixels > 10,000
    content = _make_image_bytes(width=200, height=100, fmt="PNG")
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.IMAGE_PIXEL_LIMIT_EXCEEDED.value


def test_image_reader_unsupported_format_gif():
    """Verify GIF format is rejected with UNSUPPORTED_FILE_TYPE."""
    fake_ocr = FakeOcrService()
    reader = ImageReader(ocr_service=fake_ocr)

    # Create GIF bytes
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    gif_bytes = buf.getvalue()

    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=gif_bytes,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.UNSUPPORTED_FILE_TYPE.value


def test_image_reader_corrupt_bytes_rejection():
    """Verify random garbage bytes are rejected with IMAGE_UNREADABLE."""
    fake_ocr = FakeOcrService()
    reader = ImageReader(ocr_service=fake_ocr)

    corrupt_bytes = b"NOT_A_VALID_IMAGE_HEADER_XYZ_1234567890"
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=corrupt_bytes,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.IMAGE_UNREADABLE.value


def test_image_reader_low_confidence_warning():
    """Verify words with confidence below ocr_low_confidence emit OCR_LOW_CONFIDENCE warning."""
    fake_ocr = FakeOcrService(
        canned_result=OCRResult(
            text="Uncertain text here",
            lines=[
                OCRLine(
                    text="Uncertain text here",
                    bbox=[10, 10, 200, 30],
                    words=[
                        OCRWord(text="Uncertain", bbox=[10, 10, 80, 30], confidence=42.0),  # < 60
                        OCRWord(text="text", bbox=[85, 10, 130, 30], confidence=85.0),
                        OCRWord(text="here", bbox=[135, 10, 180, 30], confidence=90.0),
                    ],
                )
            ],
            words=[],
        )
    )

    settings = Settings(ocr_low_confidence=60.0)
    reader = ImageReader(ocr_service=fake_ocr, settings=settings)

    content = _make_image_bytes(width=400, height=200, fmt="PNG")
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert status == "success"
    assert bundle is not None
    assert any(w.code == WarningCode.OCR_LOW_CONFIDENCE.value for w in warnings_list)

    seg = bundle.ordered_segments[0]
    assert WarningCode.OCR_LOW_CONFIDENCE.value in seg.warnings


def test_image_reader_ocr_unavailable_error():
    """Verify OcrUnavailableError from OCR service is translated into structured error."""
    fake_ocr = FakeOcrService(raise_on_ocr=OcrUnavailableError("Tesseract executable not found."))
    reader = ImageReader(ocr_service=fake_ocr)

    content = _make_image_bytes(width=400, height=200, fmt="PNG")
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.OCR_UNAVAILABLE.value


def test_image_reader_ocr_timeout_error():
    """Verify OcrTimeoutError from OCR service is translated into structured error."""
    fake_ocr = FakeOcrService(raise_on_ocr=OcrTimeoutError("OCR timed out."))
    reader = ImageReader(ocr_service=fake_ocr)

    content = _make_image_bytes(width=400, height=200, fmt="PNG")
    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=content,
        request_id=uuid4(),
        options=ImageAgentOptions(),
    )

    assert bundle is None
    assert status == "failed"
    assert len(errors) == 1
    assert errors[0].code == ErrorCode.OCR_TIMEOUT.value
