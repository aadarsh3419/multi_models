"""
OCR Package for Healthcare Agents.
Provides shared image-agnostic OCR service contracts and Tesseract implementation.
"""

from healthcare_agents.ocr.service import (
    FakeOcrService,
    ImageUnreadableError,
    OcrError,
    OCRLine,
    OCRResult,
    OcrService,
    OcrTimeoutError,
    OcrUnavailableError,
    OCRWord,
    TesseractOcrService,
)

__all__ = [
    "OcrService",
    "TesseractOcrService",
    "FakeOcrService",
    "OCRWord",
    "OCRLine",
    "OCRResult",
    "OcrError",
    "OcrUnavailableError",
    "OcrTimeoutError",
    "ImageUnreadableError",
]
