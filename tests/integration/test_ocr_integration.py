"""
Integration tests for Tesseract OCR service against live Tesseract binary.
Marked with @pytest.mark.ocr_integration.
Cleanly skips if Tesseract executable is not installed on the host system.
"""

import io
import shutil

import pytest
from PIL import Image, ImageDraw

from healthcare_agents.ocr.service import OcrUnavailableError, TesseractOcrService


def is_tesseract_available() -> bool:
    """Checks whether the Tesseract executable is installed and responding."""
    if shutil.which("tesseract") is None:
        return False
    try:
        service = TesseractOcrService()
        version = service.get_version()
        return bool(version)
    except (OcrUnavailableError, Exception):
        return False


@pytest.mark.ocr_integration
def test_live_tesseract_ocr_execution():
    """Executes OCR on a rendered text image using the live Tesseract binary if installed."""
    if not is_tesseract_available():
        pytest.skip(
            "Tesseract binary not installed or accessible on system; skipping live OCR test."
        )

    # Render a high-contrast clinical test image
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "Blood Pressure: 120/80 mmHg", fill="black")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    service = TesseractOcrService()
    result = service.ocr(image_bytes, language="eng")

    assert result is not None
    assert "Blood" in result.text or "Pressure" in result.text
    assert len(result.lines) >= 1
    assert len(result.words) >= 1

    first_word = result.words[0]
    assert len(first_word.bbox) == 4
    assert first_word.confidence is not None
    assert first_word.confidence > 0.0


@pytest.mark.ocr_integration
def test_live_image_reader_ocr_execution():
    """Executes live OCR extraction through ImageReader on a synthetic rendered clinical slip."""
    if not is_tesseract_available():
        pytest.skip(
            "Tesseract binary not installed or accessible on system; skipping live OCR test."
        )

    from uuid import uuid4

    from healthcare_agents.contracts.locations import ImageLocation
    from healthcare_agents.contracts.requests import ImageAgentOptions
    from healthcare_agents.sources.image_reader import ImageReader

    img = Image.new("RGB", (600, 150), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 50), "Hemoglobin 10.2 g/dL", fill="black")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    service = TesseractOcrService()
    reader = ImageReader(ocr_service=service)

    bundle, status, errors, warnings_list, stage_res, details = reader.read_bundle(
        content=image_bytes,
        request_id=uuid4(),
        options=ImageAgentOptions(ocr_languages="eng"),
    )

    assert status == "success"
    assert bundle is not None
    assert len(bundle.ordered_segments) >= 1
    seg = bundle.ordered_segments[0]
    assert "Hemoglobin" in seg.text or "10.2" in seg.text
    assert isinstance(seg.source_location, ImageLocation)
    assert seg.source_location.original_width == 600
    assert seg.source_location.original_height == 150
