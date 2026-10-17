"""
Image Document Reader and Provenance Engine.

Layering Rule: This module does NOT import pytesseract or pymupdf.
Reads image files from memory bytes only; never writes image data to disk.
Enforces Pillow safety boundaries (format allowlist, decompression bomb limit, dimension check before load),
handles EXIF transposition, executes OCR via injected OcrService, and maps coordinates back to displayed pixels.
"""

import hashlib
import io
import logging
import warnings
from typing import Any, Literal
from uuid import UUID

from PIL import Image, ImageOps

from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts.errors import ErrorCode, StructuredError, WarningCode
from healthcare_agents.contracts.locations import ImageLocation
from healthcare_agents.contracts.requests import ImageAgentOptions
from healthcare_agents.contracts.results import StageResult
from healthcare_agents.contracts.source_text import SourceSegment, SourceTextBundle
from healthcare_agents.ocr.service import (
    OcrService,
    OcrTimeoutError,
    OcrUnavailableError,
)

logger = logging.getLogger("healthcare_agents.sources.image_reader")

ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF"})


class ImageReader:
    """
    Ingests raw image bytes, applies safety checks, handles EXIF rotation,
    applies invertible upscaling transform, invokes OcrService, and constructs
    spatially-provenanced SourceSegments.
    """

    def __init__(
        self,
        ocr_service: OcrService,
        settings: Settings | None = None,
    ):
        self.ocr_service = ocr_service
        self.settings = settings or get_settings()

    def read_bundle(
        self,
        content: bytes,
        request_id: UUID,
        options: ImageAgentOptions | None = None,
    ) -> tuple[
        SourceTextBundle | None,
        Literal["success", "partial", "failed"],
        list[StructuredError],
        list[StructuredError],
        StageResult,
        dict[str, Any],
    ]:
        """
        Parses image bytes into SourceTextBundle with pixel bounding boxes and OCR word confidence.
        Returns: (bundle, status, errors, warnings, stage_result, image_details)
        """
        options = options or ImageAgentOptions()
        errors: list[StructuredError] = []
        warnings_list: list[StructuredError] = []
        image_details: dict[str, Any] = {}

        content_len = len(content) if content is not None else 0
        content_sha256 = hashlib.sha256(content).hexdigest() if content else ""

        # 1. Content size validation
        if not content or content_len == 0:
            err = StructuredError(
                code=ErrorCode.INVALID_INPUT.value,
                stage="source_reading",
                message="Image content bytes must not be empty.",
                details={"byte_count": 0},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details

        if content_len > self.settings.max_image_bytes:
            err = StructuredError(
                code=ErrorCode.FILE_TOO_LARGE.value,
                stage="source_reading",
                message=(
                    f"Image file size ({content_len:,} bytes) exceeds "
                    f"max_image_bytes ({self.settings.max_image_bytes:,})."
                ),
                details={
                    "file_bytes": content_len,
                    "max_bytes": self.settings.max_image_bytes,
                },
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details

        # 2. Decode image with Pillow under strict safety boundaries
        raw_width = 0
        raw_height = 0
        oriented_w = 0
        oriented_h = 0
        orientation_val = 1
        coord_sys = "pixel_top_left"
        fmt_name = ""

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                stream = io.BytesIO(content)
                with Image.open(stream) as img:
                    fmt_name = (getattr(img, "format", "") or "").upper()
                    if fmt_name not in ALLOWED_IMAGE_FORMATS:
                        err = StructuredError(
                            code=ErrorCode.UNSUPPORTED_FILE_TYPE.value,
                            stage="source_reading",
                            message=(
                                f"Unsupported image format '{fmt_name}'. "
                                f"Allowed formats: {sorted(ALLOWED_IMAGE_FORMATS)}."
                            ),
                            details={"format": fmt_name},
                        )
                        errors.append(err)
                        stage_res = StageResult(stage_name="source_reading", status="failed")
                        return None, "failed", errors, warnings_list, stage_res, image_details

                    n_frames = getattr(img, "n_frames", 1)
                    is_animated = getattr(img, "is_animated", False)
                    if n_frames > 1 or is_animated:
                        err = StructuredError(
                            code=ErrorCode.UNSUPPORTED_FILE_TYPE.value,
                            stage="source_reading",
                            message="Animated or multi-frame image files are not supported.",
                            details={"n_frames": n_frames, "is_animated": is_animated},
                        )
                        errors.append(err)
                        stage_res = StageResult(stage_name="source_reading", status="failed")
                        return None, "failed", errors, warnings_list, stage_res, image_details

                    raw_width, raw_height = img.size
                    pixel_count = raw_width * raw_height
                    if pixel_count > self.settings.max_image_pixels:
                        err = StructuredError(
                            code=ErrorCode.IMAGE_PIXEL_LIMIT_EXCEEDED.value,
                            stage="source_reading",
                            message=(
                                f"Image pixel count ({pixel_count:,}) exceeds "
                                f"max_image_pixels ({self.settings.max_image_pixels:,})."
                            ),
                            details={
                                "pixel_count": pixel_count,
                                "max_pixels": self.settings.max_image_pixels,
                                "width": raw_width,
                                "height": raw_height,
                            },
                        )
                        errors.append(err)
                        stage_res = StageResult(stage_name="source_reading", status="failed")
                        return None, "failed", errors, warnings_list, stage_res, image_details

                    # Safe load into memory
                    img.load()

                    # Check EXIF orientation tag
                    try:
                        exif = img.getexif()
                        if exif:
                            tag_val = exif.get(274, 1)
                            if isinstance(tag_val, int) and tag_val in range(1, 9):
                                orientation_val = tag_val
                    except Exception:
                        orientation_val = 1

                    transposed = ImageOps.exif_transpose(img)
                    oriented_w, oriented_h = transposed.size
                    coord_sys = (
                        "oriented_pixel_top_left" if orientation_val != 1 else "pixel_top_left"
                    )

                    # Grayscale conversion for OCR preprocessing
                    gray = transposed.convert("L")

            except (Image.DecompressionBombError, Image.DecompressionBombWarning) as bomb_err:
                err = StructuredError(
                    code=ErrorCode.IMAGE_PIXEL_LIMIT_EXCEEDED.value,
                    stage="source_reading",
                    message=f"Image exceeds decompression safety limits: {type(bomb_err).__name__}",
                    details={"exception": type(bomb_err).__name__},
                )
                errors.append(err)
                stage_res = StageResult(stage_name="source_reading", status="failed")
                return None, "failed", errors, warnings_list, stage_res, image_details

            except Exception as err_pillow:
                err = StructuredError(
                    code=ErrorCode.IMAGE_UNREADABLE.value,
                    stage="source_reading",
                    message=f"Corrupt or unreadable image file: {type(err_pillow).__name__}",
                    details={"exception": type(err_pillow).__name__},
                )
                errors.append(err)
                stage_res = StageResult(stage_name="source_reading", status="failed")
                return None, "failed", errors, warnings_list, stage_res, image_details

        # 3. Optional upscaling transform (at most 2x when shorter side < image_upscale_min_side)
        effective_upscale = options.upscale_enabled
        min_side = min(oriented_w, oriented_h)
        scale = 1.0

        if (
            effective_upscale
            and self.settings.image_upscale_min_side > 0
            and min_side < self.settings.image_upscale_min_side
            and min_side > 0
        ):
            scale = min(2.0, self.settings.image_upscale_min_side / min_side)
            new_w = max(1, int(round(oriented_w * scale)))
            new_h = max(1, int(round(oriented_h * scale)))
            processed_img = gray.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        else:
            scale = 1.0
            processed_img = gray

        image_details = {
            "format": fmt_name,
            "raw_width": raw_width,
            "raw_height": raw_height,
            "oriented_width": oriented_w,
            "oriented_height": oriented_h,
            "exif_orientation": orientation_val,
            "coordinate_system": coord_sys,
            "upscale_factor": scale,
        }

        # 4. Convert preprocessed image to PNG bytes for OCR execution
        buf = io.BytesIO()
        processed_img.save(buf, format="PNG")
        ocr_input_bytes = buf.getvalue()

        ocr_langs = options.ocr_languages or self.settings.ocr_languages
        try:
            ocr_result = self.ocr_service.ocr(
                image_bytes=ocr_input_bytes,
                language=ocr_langs,
                timeout_seconds=self.settings.ocr_timeout_seconds,
                max_pixels=self.settings.max_render_pixels,
            )
        except OcrUnavailableError as ocr_unavail:
            err = StructuredError(
                code=ErrorCode.OCR_UNAVAILABLE.value,
                stage="source_reading",
                message=str(ocr_unavail),
                details={"ocr_languages": ocr_langs},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details
        except OcrTimeoutError as ocr_time:
            err = StructuredError(
                code=ErrorCode.OCR_TIMEOUT.value,
                stage="source_reading",
                message=str(ocr_time),
                details={"timeout_seconds": self.settings.ocr_timeout_seconds},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details
        except Exception as ocr_gen_err:
            err = StructuredError(
                code=ErrorCode.IMAGE_UNREADABLE.value,
                stage="source_reading",
                message=f"OCR execution failed: {type(ocr_gen_err).__name__}",
                details={"exception": type(ocr_gen_err).__name__},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details

        # 5. Build SourceSegments and invert scale coordinates back to displayed pixel space
        segments: list[SourceSegment] = []
        has_low_confidence = False
        seg_idx = 1

        for line in ocr_result.lines:
            line_text = line.text.strip()
            if not line_text:
                continue

            # Invert line bounding box
            if scale != 1.0:
                inv_box = [round(c / scale, 1) for c in line.bbox]
            else:
                inv_box = [float(c) for c in line.bbox]

            # Clamp coordinates to displayed oriented boundaries [0, 0, oriented_w, oriented_h]
            clamped_line_box = [
                max(0.0, min(float(oriented_w), inv_box[0])),
                max(0.0, min(float(oriented_h), inv_box[1])),
                max(0.0, min(float(oriented_w), inv_box[2])),
                max(0.0, min(float(oriented_h), inv_box[3])),
            ]

            # Invert word tokens and track confidence
            word_dicts: list[dict[str, Any]] = []
            word_confs: list[float] = []

            for w in line.words:
                if scale != 1.0:
                    inv_w_box = [round(c / scale, 1) for c in w.bbox]
                else:
                    inv_w_box = [float(c) for c in w.bbox]

                word_dicts.append(
                    {
                        "text": w.text,
                        "bbox": inv_w_box,
                        "confidence": w.confidence,
                    }
                )
                if w.confidence is not None:
                    word_confs.append(w.confidence)
                    if w.confidence < self.settings.ocr_low_confidence:
                        has_low_confidence = True

            loc = ImageLocation(
                kind="image",
                bbox=clamped_line_box,
                original_width=raw_width,
                original_height=raw_height,
                coordinate_system=coord_sys,
                exif_orientation=orientation_val,
                oriented_width=oriented_w,
                oriented_height=oriented_h,
                extraction_method="ocr:tesseract",
                ocr_words=word_dicts,
                layout_group_id=f"block_{line.block_id}_par_{line.par_id}",
            )

            seg_warnings: list[str] = []
            if any(c < self.settings.ocr_low_confidence for c in word_confs):
                seg_warnings.append(WarningCode.OCR_LOW_CONFIDENCE.value)

            seg = SourceSegment(
                segment_id=f"img_seg_{seg_idx:04d}",
                text=line_text,
                extraction_method="ocr:tesseract",
                source_location=loc,
                layout_group_id=f"block_{line.block_id}_par_{line.par_id}",
                ocr_word_confidences=word_confs,
                warnings=seg_warnings,
            )
            segments.append(seg)
            seg_idx += 1

        # Check for overall low confidence warning
        if has_low_confidence:
            warnings_list.append(
                StructuredError(
                    code=WarningCode.OCR_LOW_CONFIDENCE.value,
                    stage="source_reading",
                    message=(
                        f"One or more OCR words have confidence below {self.settings.ocr_low_confidence:.1f}."
                    ),
                    details={"threshold": self.settings.ocr_low_confidence},
                )
            )

        # Check if any usable text was extracted
        if not segments:
            err = StructuredError(
                code=ErrorCode.NO_USABLE_SOURCE_CONTENT.value,
                stage="source_reading",
                message="No usable text could be extracted from the image document via OCR.",
                details={"ocr_word_count": len(ocr_result.words)},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings_list, stage_res, image_details

        # Construct final SourceTextBundle
        bundle = SourceTextBundle(
            request_id=request_id,
            source_type="image",
            extraction_method="ocr:tesseract",
            ordered_segments=segments,
            safe_metadata={
                "source_id": f"src_img_{content_sha256[:12]}",
                "media_type": f"image/{fmt_name.lower()}",
                "format": fmt_name,
                "byte_count": content_len,
                "content_sha256": content_sha256,
                "width": raw_width,
                "height": raw_height,
                "oriented_width": oriented_w,
                "oriented_height": oriented_h,
                "exif_orientation": orientation_val,
                "upscale_factor": scale,
                "segment_count": len(segments),
                "extraction_method": "ocr:tesseract",
            },
        )

        stage_res = StageResult(
            stage_name="source_reading",
            status="success",
            counts={"segments": len(segments), "words": len(ocr_result.words)},
            details={
                "upscale_factor": scale,
                "coordinate_system": coord_sys,
                "has_low_confidence": has_low_confidence,
            },
        )

        return bundle, "success", errors, warnings_list, stage_res, image_details
