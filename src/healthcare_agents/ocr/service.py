"""
Shared OCR service protocol and Tesseract implementation.

Layering Rule: Only this module imports pytesseract.
All other modules consume the OcrService protocol.
"""

import io
import logging
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

import pytesseract
from PIL import Image
from pydantic import BaseModel, Field

from healthcare_agents.contracts.errors import ErrorCode

logger = logging.getLogger("healthcare_agents.ocr")


class OcrError(Exception):
    """Base exception for all OCR operations."""

    def __init__(self, message: str, code: str = ErrorCode.OCR_UNAVAILABLE.value):
        super().__init__(message)
        self.code = code
        self.message = message


class OcrUnavailableError(OcrError):
    """Raised when the Tesseract binary or requested language pack is missing."""

    def __init__(self, message: str):
        super().__init__(message, code=ErrorCode.OCR_UNAVAILABLE.value)


class OcrTimeoutError(OcrError):
    """Raised when an OCR call exceeds its execution timeout."""

    def __init__(self, message: str):
        super().__init__(message, code=ErrorCode.OCR_TIMEOUT.value)


class ImageUnreadableError(OcrError):
    """Raised when image bytes cannot be decoded or exceed decompression limits."""

    def __init__(self, message: str):
        super().__init__(message, code=ErrorCode.IMAGE_UNREADABLE.value)


class OCRWord(BaseModel):
    """Single recognized word token with pixel bounding box and raw confidence."""

    text: str = Field(description="Recognized word text.")
    bbox: list[int] = Field(description="Bounding box in pixel coordinates [x0, y0, x1, y1].")
    confidence: float | None = Field(
        default=None,
        description="Raw Tesseract confidence score (0.0-100.0), or None if unstated/-1.",
    )


class OCRLine(BaseModel):
    """Line of recognized words grouped by block, paragraph, and line IDs."""

    text: str = Field(description="Concatenated line text.")
    bbox: list[int] = Field(description="Bounding box enclosing all words [x0, y0, x1, y1].")
    words: list[OCRWord] = Field(default_factory=list)
    block_id: int = Field(default=1, description="Layout block ID.")
    par_id: int = Field(default=1, description="Paragraph ID within block.")
    line_id: int = Field(default=1, description="Line ID within paragraph.")


class OCRResult(BaseModel):
    """Complete structured output from an OCR execution."""

    text: str = Field(description="Full text concatenated across all lines.")
    lines: list[OCRLine] = Field(default_factory=list)
    words: list[OCRWord] = Field(default_factory=list)
    language: str = Field(default="eng", description="Primary OCR language.")
    engine_version: str = Field(default="tesseract", description="OCR engine identifier.")


@runtime_checkable
class OcrService(Protocol):
    """Abstract protocol for image OCR engines."""

    def ocr(
        self,
        image_bytes: bytes,
        language: str = "eng",
        timeout_seconds: int = 60,
        max_pixels: int = 20_000_000,
    ) -> OCRResult:
        """Execute OCR on raw image bytes."""
        ...

    def get_available_languages(self) -> list[str]:
        """Return list of installed language pack codes."""
        ...

    def get_version(self) -> str:
        """Return the underlying OCR engine version."""
        ...


class TesseractOcrService(OcrService):
    """Production OCR service wrapping the local Tesseract OCR engine via pytesseract."""

    def __init__(self, tesseract_cmd: str | None = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @property
    def pytesseract_version(self) -> str:
        return getattr(pytesseract, "__version__", "unknown")

    def get_version(self) -> str:
        """Detect and return Tesseract engine version, or raise OcrUnavailableError."""
        try:
            return str(pytesseract.get_tesseract_version())
        except Exception as err:
            raise OcrUnavailableError(
                f"Tesseract executable is not installed or not accessible: {type(err).__name__}"
            ) from err

    def get_available_languages(self) -> list[str]:
        """Query installed Tesseract language packs."""
        try:
            return pytesseract.get_languages()
        except Exception as err:
            raise OcrUnavailableError(
                f"Failed to query Tesseract languages: {type(err).__name__}"
            ) from err

    def ocr(
        self,
        image_bytes: bytes,
        language: str = "eng",
        timeout_seconds: int = 60,
        max_pixels: int = 20_000_000,
    ) -> OCRResult:
        """
        Executes Tesseract OCR on image bytes with strict decompression-bomb protection,
        language verification, per-call timeout, and structured token/line grouping.
        """
        # 1. Decode image with local decompression-bomb protection
        try:
            image_stream = io.BytesIO(image_bytes)
            with Image.open(image_stream) as img:
                # Check total pixel volume against budget
                pixel_count = img.width * img.height
                if pixel_count > max_pixels:
                    raise ImageUnreadableError(
                        f"Image pixel count ({pixel_count:,}) exceeds max_pixels limit ({max_pixels:,})."
                    )
                # Load image data safely into memory
                img.load()
                # Work with an independent copy in RGB mode
                image = img.convert("RGB")
        except ImageUnreadableError:
            raise
        except Exception as err:
            raise ImageUnreadableError(
                f"Failed to decode image bytes with Pillow: {type(err).__name__}"
            ) from err

        # 2. Check engine availability
        engine_version = self.get_version()

        # 3. Check requested language packs (support joined languages e.g. 'eng+hin')
        requested_langs = [l_part.strip() for l_part in language.split("+") if l_part.strip()]
        available_langs = self.get_available_languages()
        missing_langs = [l_req for l_req in requested_langs if l_req not in available_langs]
        if missing_langs:
            raise OcrUnavailableError(
                f"Requested OCR language data missing in Tesseract: {missing_langs}. "
                f"Installed languages: {available_langs}."
            )

        # 4. Execute image_to_data with timeout
        try:
            data_dict = pytesseract.image_to_data(
                image,
                lang=language,
                timeout=timeout_seconds,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.pytesseract.TesseractTimeoutError as err:
            raise OcrTimeoutError(
                f"Tesseract OCR timed out after {timeout_seconds} seconds."
            ) from err
        except Exception as err:
            err_name = type(err).__name__
            if "Timeout" in err_name:
                raise OcrTimeoutError(
                    f"Tesseract OCR timed out after {timeout_seconds} seconds."
                ) from err
            raise OcrError(f"Tesseract extraction failed: {err_name}: {err}") from err

        # 5. Group tokens into ordered lines by (block_num, par_num, line_num)
        n_boxes = len(data_dict.get("text", []))
        line_groups: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
        all_words: list[OCRWord] = []

        for i in range(n_boxes):
            word_text = str(data_dict["text"][i]).strip()
            if not word_text:
                continue

            left = int(data_dict["left"][i])
            top = int(data_dict["top"][i])
            width = int(data_dict["width"][i])
            height = int(data_dict["height"][i])
            bbox = [left, top, left + width, top + height]

            raw_conf = data_dict["conf"][i]
            conf_val: float | None = None
            try:
                parsed_conf = float(raw_conf)
                if parsed_conf >= 0.0:
                    conf_val = parsed_conf
            except (ValueError, TypeError):
                conf_val = None

            word_obj = OCRWord(text=word_text, bbox=bbox, confidence=conf_val)
            all_words.append(word_obj)

            block_id = int(data_dict.get("block_num", [0] * n_boxes)[i])
            par_id = int(data_dict.get("par_num", [0] * n_boxes)[i])
            line_id = int(data_dict.get("line_num", [0] * n_boxes)[i])
            line_groups[(block_id, par_id, line_id)].append(word_obj)

        # Assemble lines
        ocr_lines: list[OCRLine] = []
        for (b_id, p_id, l_id), words_in_line in line_groups.items():
            if not words_in_line:
                continue
            line_text = " ".join(w.text for w in words_in_line)
            line_x0 = min(w.bbox[0] for w in words_in_line)
            line_y0 = min(w.bbox[1] for w in words_in_line)
            line_x1 = max(w.bbox[2] for w in words_in_line)
            line_y1 = max(w.bbox[3] for w in words_in_line)
            ocr_lines.append(
                OCRLine(
                    text=line_text,
                    bbox=[line_x0, line_y0, line_x1, line_y1],
                    words=words_in_line,
                    block_id=b_id,
                    par_id=p_id,
                    line_id=l_id,
                )
            )

        full_text = "\n".join(line_item.text for line_item in ocr_lines)
        return OCRResult(
            text=full_text,
            lines=ocr_lines,
            words=all_words,
            language=language,
            engine_version=engine_version,
        )


class FakeOcrService(OcrService):
    """Configurable fake OCR service for unit and integration testing without Tesseract."""

    def __init__(
        self,
        canned_result: OCRResult | None = None,
        raise_on_ocr: Exception | None = None,
        available_languages: list[str] | None = None,
        engine_version: str = "fake-tesseract-5.3.0",
    ):
        self.canned_result = canned_result
        self.raise_on_ocr = raise_on_ocr
        self.available_languages = (
            available_languages if available_languages is not None else ["eng", "hin", "osd"]
        )
        self.engine_version = engine_version
        self.call_count = 0
        self.last_call_params: dict[str, Any] = {}

    @property
    def pytesseract_version(self) -> str:
        return "0.3.13"

    def get_version(self) -> str:
        return self.engine_version

    def get_available_languages(self) -> list[str]:
        return list(self.available_languages)

    def ocr(
        self,
        image_bytes: bytes,
        language: str = "eng",
        timeout_seconds: int = 60,
        max_pixels: int = 20_000_000,
    ) -> OCRResult:
        self.call_count += 1
        self.last_call_params = {
            "image_bytes_len": len(image_bytes),
            "language": language,
            "timeout_seconds": timeout_seconds,
            "max_pixels": max_pixels,
        }

        # Check decompression-bomb with Pillow
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                pixel_count = img.width * img.height
                if pixel_count > max_pixels:
                    raise ImageUnreadableError(
                        f"Image pixel count ({pixel_count}) exceeds max_pixels ({max_pixels})."
                    )
                img.load()
        except ImageUnreadableError:
            raise
        except Exception as err:
            raise ImageUnreadableError(f"Corrupt image bytes: {type(err).__name__}") from err

        if self.raise_on_ocr:
            raise self.raise_on_ocr

        # Check languages
        requested_langs = [l_part.strip() for l_part in language.split("+") if l_part.strip()]
        missing_langs = [
            l_req for l_req in requested_langs if l_req not in self.available_languages
        ]
        if missing_langs:
            raise OcrUnavailableError(
                f"Requested OCR language data missing in Tesseract: {missing_langs}."
            )

        if self.canned_result is not None:
            return self.canned_result

        # Default fallback synthetic result
        return OCRResult(
            text="Fasting Blood Glucose 110 mg/dL",
            lines=[
                OCRLine(
                    text="Fasting Blood Glucose 110 mg/dL",
                    bbox=[10, 10, 300, 30],
                    words=[
                        OCRWord(text="Fasting", bbox=[10, 10, 70, 30], confidence=95.0),
                        OCRWord(text="Blood", bbox=[75, 10, 130, 30], confidence=95.0),
                        OCRWord(text="Glucose", bbox=[135, 10, 200, 30], confidence=95.0),
                        OCRWord(text="110", bbox=[205, 10, 240, 30], confidence=98.0),
                        OCRWord(text="mg/dL", bbox=[245, 10, 300, 30], confidence=92.0),
                    ],
                    block_id=1,
                    par_id=1,
                    line_id=1,
                )
            ],
            words=[
                OCRWord(text="Fasting", bbox=[10, 10, 70, 30], confidence=95.0),
                OCRWord(text="Blood", bbox=[75, 10, 130, 30], confidence=95.0),
                OCRWord(text="Glucose", bbox=[135, 10, 200, 30], confidence=95.0),
                OCRWord(text="110", bbox=[205, 10, 240, 30], confidence=98.0),
                OCRWord(text="mg/dL", bbox=[245, 10, 300, 30], confidence=92.0),
            ],
            language=language,
            engine_version=self.engine_version,
        )
