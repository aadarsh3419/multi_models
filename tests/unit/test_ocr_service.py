"""
Unit tests for OcrService, TesseractOcrService, and FakeOcrService.
Offline only: never requires an external Tesseract binary.
"""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from healthcare_agents.contracts.errors import ErrorCode
from healthcare_agents.ocr.service import (
    FakeOcrService,
    ImageUnreadableError,
    OCRLine,
    OCRResult,
    OcrTimeoutError,
    OcrUnavailableError,
    OCRWord,
    TesseractOcrService,
)


def create_test_image_bytes(width: int = 100, height: int = 40, color: str = "white") -> bytes:
    """Create in-memory PNG image bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_decompression_bomb_rejection():
    """Verify that images exceeding max_pixels are rejected with ImageUnreadableError."""
    service = FakeOcrService()
    # Image is 50x50 = 2500 pixels. Setting max_pixels to 2000 triggers rejection.
    img_bytes = create_test_image_bytes(50, 50)

    with pytest.raises(ImageUnreadableError) as exc_info:
        service.ocr(img_bytes, max_pixels=2000)

    assert exc_info.value.code == ErrorCode.IMAGE_UNREADABLE.value
    assert "exceeds max_pixels" in exc_info.value.message


def test_ocr_corrupt_image_bytes_rejection():
    """Verify that malformed/non-image bytes raise ImageUnreadableError."""
    service = FakeOcrService()
    corrupt_bytes = b"this is not an image file header"

    with pytest.raises(ImageUnreadableError) as exc_info:
        service.ocr(corrupt_bytes)

    assert exc_info.value.code == ErrorCode.IMAGE_UNREADABLE.value


def test_ocr_missing_language_rejection():
    """Verify that requesting an uninstalled language pack raises OcrUnavailableError naming missing codes."""
    service = FakeOcrService(available_languages=["eng", "osd"])
    img_bytes = create_test_image_bytes(20, 20)

    with pytest.raises(OcrUnavailableError) as exc_info:
        service.ocr(img_bytes, language="eng+hin")

    assert exc_info.value.code == ErrorCode.OCR_UNAVAILABLE.value
    assert "hin" in exc_info.value.message


def test_ocr_timeout_propagation():
    """Verify that simulated timeout raises OcrTimeoutError."""
    service = FakeOcrService(raise_on_ocr=OcrTimeoutError("Timeout exceeded"))
    img_bytes = create_test_image_bytes(20, 20)

    with pytest.raises(OcrTimeoutError) as exc_info:
        service.ocr(img_bytes)

    assert exc_info.value.code == ErrorCode.OCR_TIMEOUT.value


def test_fake_ocr_service_canned_result():
    """Verify FakeOcrService returns configured canned result."""
    custom_result = OCRResult(
        text="Blood Pressure: 120/80 mmHg",
        lines=[
            OCRLine(
                text="Blood Pressure: 120/80 mmHg",
                bbox=[5, 5, 250, 25],
                words=[
                    OCRWord(text="Blood", bbox=[5, 5, 50, 25], confidence=98.0),
                    OCRWord(text="Pressure:", bbox=[55, 5, 120, 25], confidence=96.0),
                    OCRWord(text="120/80", bbox=[125, 5, 180, 25], confidence=99.0),
                    OCRWord(text="mmHg", bbox=[185, 5, 250, 25], confidence=95.0),
                ],
                block_id=1,
                par_id=1,
                line_id=1,
            )
        ],
        words=[
            OCRWord(text="Blood", bbox=[5, 5, 50, 25], confidence=98.0),
            OCRWord(text="Pressure:", bbox=[55, 5, 120, 25], confidence=96.0),
            OCRWord(text="120/80", bbox=[125, 5, 180, 25], confidence=99.0),
            OCRWord(text="mmHg", bbox=[185, 5, 250, 25], confidence=95.0),
        ],
        language="eng",
        engine_version="test-1.0",
    )
    service = FakeOcrService(canned_result=custom_result)
    img_bytes = create_test_image_bytes(300, 50)

    res = service.ocr(img_bytes, language="eng")
    assert res.text == "Blood Pressure: 120/80 mmHg"
    assert len(res.lines) == 1
    assert len(res.words) == 4
    assert service.call_count == 1
    assert service.last_call_params["language"] == "eng"


@patch("pytesseract.get_tesseract_version")
def test_tesseract_service_missing_binary_raises_unavailable(mock_version):
    """Verify that if pytesseract fails to find the tesseract binary, OcrUnavailableError is raised."""
    mock_version.side_effect = FileNotFoundError("tesseract not found")
    service = TesseractOcrService()

    with pytest.raises(OcrUnavailableError) as exc_info:
        service.get_version()

    assert exc_info.value.code == ErrorCode.OCR_UNAVAILABLE.value


@patch("pytesseract.get_languages")
@patch("pytesseract.get_tesseract_version")
@patch("pytesseract.image_to_data")
def test_tesseract_service_token_grouping_and_negative_confidence(
    mock_image_to_data, mock_version, mock_get_languages
):
    """Verify image_to_data token parsing groups by line and converts -1 confidence to None."""
    mock_version.return_value = "5.3.3"
    mock_get_languages.return_value = ["eng"]

    # Mock Tesseract DICT output with 2 lines across 1 block
    mock_image_to_data.return_value = {
        "text": ["", "Hemoglobin", "14.2", "g/dL", "Normal", "range"],
        "left": [0, 10, 80, 120, 10, 60],
        "top": [0, 10, 10, 10, 30, 30],
        "width": [0, 65, 35, 30, 45, 40],
        "height": [0, 15, 15, 15, 15, 15],
        "conf": ["-1", "95.5", "98.0", "92.0", "-1", "88.0"],
        "block_num": [0, 1, 1, 1, 1, 1],
        "par_num": [0, 1, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 1, 2, 2],
    }

    service = TesseractOcrService()
    img_bytes = create_test_image_bytes(200, 100)
    result = service.ocr(img_bytes, language="eng")

    assert len(result.lines) == 2
    line1 = result.lines[0]
    assert line1.text == "Hemoglobin 14.2 g/dL"
    assert line1.bbox == [10, 10, 150, 25]  # [10, 10, 120+30, 10+15]
    assert len(line1.words) == 3
    assert line1.words[0].confidence == 95.5

    line2 = result.lines[1]
    assert line2.text == "Normal range"
    # '-1' raw confidence maps to None
    assert line2.words[0].confidence is None
    assert line2.words[1].confidence == 88.0
