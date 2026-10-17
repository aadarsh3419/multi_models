"""
PDF Document Reader and Provenance Engine.

Layering Rule: Only this module imports pymupdf.
Reads PDF from memory bytes only; never writes PDF to disk.
Extracts native text lines, tables (via find_tables), image regions,
executes selective OCR via injected OcrService, and suppresses native/OCR duplicates.
"""

import hashlib
import logging
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Literal
from uuid import UUID

os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")
import pymupdf

from healthcare_agents.config import Settings, get_settings
from healthcare_agents.contracts.errors import ErrorCode, StructuredError, WarningCode
from healthcare_agents.contracts.locations import PdfLocation
from healthcare_agents.contracts.requests import PDFAgentOptions
from healthcare_agents.contracts.results import StageResult
from healthcare_agents.contracts.source_text import SourceSegment, SourceTextBundle
from healthcare_agents.ocr.service import OcrService, OcrUnavailableError

logger = logging.getLogger("healthcare_agents.sources.pdf_reader")

ALNUM_REGEX = re.compile(r"[a-zA-Z0-9\u0900-\u097F]")


def compute_bbox_iou(box1: list[float] | None, box2: list[float] | None) -> float:
    """Computes Intersection-over-Union (IoU) of two 2D bounding boxes [x0, y0, x1, y1]."""
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.0

    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    ix0 = max(x0_1, x0_2)
    iy0 = max(y0_1, y0_2)
    ix1 = min(x1_1, x1_2)
    iy1 = min(y1_1, y1_2)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0

    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = max(0.0, x1_1 - x0_1) * max(0.0, y1_1 - y0_1)
    area2 = max(0.0, x1_2 - x0_2) * max(0.0, y1_2 - y0_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0.0 else 0.0


def plan_page_ocr(
    page_num: int,
    native_alnum_count: int,
    image_regions: list[dict[str, Any]],
    raster_area_fraction: float,
    ocr_mode: Literal["auto", "never", "always"],
    settings: Settings,
) -> dict[str, Any]:
    """
    Pure decision function that plans OCR for a single PDF page.
    Returns plan dictionary with decision ('none', 'full_page', 'selective_regions'), reasons, and targets.
    """
    reasons: list[str] = []
    decision: Literal["none", "full_page", "selective_regions"] = "none"
    targets: list[dict[str, Any]] = []

    if ocr_mode == "never":
        if native_alnum_count < settings.pdf_min_native_alnum:
            reasons.append(
                f"OCR mode is 'never', but page has deficient native text ({native_alnum_count} < {settings.pdf_min_native_alnum})."
            )
        else:
            reasons.append("OCR mode is 'never'; native text sufficient.")
        decision = "none"

    elif ocr_mode == "always":
        decision = "full_page"
        reasons.append("OCR mode is 'always'; full page OCR requested.")

    else:
        # 'auto' mode
        if raster_area_fraction >= settings.pdf_fullpage_raster_fraction:
            decision = "full_page"
            reasons.append(
                f"Raster coverage fraction ({raster_area_fraction:.2f}) >= fullpage threshold ({settings.pdf_fullpage_raster_fraction})."
            )
        elif native_alnum_count < settings.pdf_min_native_alnum:
            decision = "full_page"
            reasons.append(
                f"Native alphanumeric count ({native_alnum_count}) < minimum threshold ({settings.pdf_min_native_alnum})."
            )
        else:
            qualified_images = [
                img
                for img in image_regions
                if img["fraction"] >= settings.pdf_image_region_min_fraction
                and img.get("native_alnum_count", 0) < settings.pdf_min_native_alnum
            ]
            if qualified_images:
                decision = "selective_regions"
                targets = qualified_images
                reasons.append(
                    f"Selected {len(qualified_images)} image region(s) exceeding {settings.pdf_image_region_min_fraction * 100:.0f}% page area with deficient native text."
                )
            else:
                decision = "none"
                reasons.append("Native text sufficient; no qualified image regions requiring OCR.")

    return {
        "page": page_num,
        "decision": decision,
        "reasons": reasons,
        "targets": targets,
    }


class PdfReader:
    """
    Production PDF ingestion engine for healthcare documents.
    Injected with OcrService and Settings; opens documents in-memory only.
    """

    def __init__(self, ocr_service: OcrService, settings: Settings | None = None):
        self.ocr_service = ocr_service
        self.settings = settings or get_settings()

    @property
    def pymupdf_version(self) -> str:
        return getattr(pymupdf, "__version__", "unknown")

    @property
    def table_api_available(self) -> bool:
        return hasattr(pymupdf.Page, "find_tables")

    def read_bundle(
        self,
        content: bytes,
        request_id: UUID,
        options: PDFAgentOptions | None = None,
    ) -> tuple[
        SourceTextBundle | None,
        Literal["success", "partial", "failed"],
        list[StructuredError],
        list[StructuredError],
        StageResult,
        list[dict[str, Any]],
    ]:
        """
        Parses PDF bytes into SourceTextBundle with full layout and spatial provenance.
        Returns: (bundle, status, errors, warnings, stage_result, page_plans)
        """
        options = options or PDFAgentOptions()
        errors: list[StructuredError] = []
        warnings: list[StructuredError] = []
        page_plans: list[dict[str, Any]] = []

        content_len = len(content)
        content_sha256 = hashlib.sha256(content).hexdigest()

        # 1. Size limit validation
        if content_len > self.settings.max_pdf_bytes:
            err = StructuredError(
                code=ErrorCode.FILE_TOO_LARGE.value,
                stage="source_reading",
                message=f"PDF file size ({content_len:,} bytes) exceeds MAX_PDF_BYTES ({self.settings.max_pdf_bytes:,}).",
                details={"file_bytes": content_len, "max_bytes": self.settings.max_pdf_bytes},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        # 2. Magic header check (%PDF-) within first 1024 bytes
        header_sample = content[:1024]
        if b"%PDF-" not in header_sample:
            err = StructuredError(
                code=ErrorCode.UNSUPPORTED_FILE_TYPE.value,
                stage="source_reading",
                message="File is missing valid '%PDF-' header within the first 1024 bytes.",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        # 3. Open in-memory PyMuPDF document
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as open_err:
            err = StructuredError(
                code=ErrorCode.PDF_UNREADABLE.value,
                stage="source_reading",
                message=f"PyMuPDF failed to parse document stream: {type(open_err).__name__}",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        # 4. Check encryption & zero pages
        if doc.page_count == 0:
            doc.close()
            err = StructuredError(
                code=ErrorCode.PDF_UNREADABLE.value,
                stage="source_reading",
                message="PDF document contains zero pages.",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        if doc.is_encrypted or doc.needs_pass:
            doc.close()
            err = StructuredError(
                code=ErrorCode.PDF_ENCRYPTED.value,
                stage="source_reading",
                message="PDF document is password-protected or encrypted; decryption is not supported.",
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        # 5. Page count limit check (Fail before touching any page)
        if doc.page_count > self.settings.max_pdf_pages:
            p_count = doc.page_count
            doc.close()
            err = StructuredError(
                code=ErrorCode.PAGE_LIMIT_EXCEEDED.value,
                stage="source_reading",
                message=f"PDF document page count ({p_count}) exceeds MAX_PDF_PAGES ({self.settings.max_pdf_pages}).",
                details={"page_count": p_count, "max_pages": self.settings.max_pdf_pages},
            )
            errors.append(err)
            stage_res = StageResult(stage_name="source_reading", status="failed")
            return None, "failed", errors, warnings, stage_res, page_plans

        # 6. Process pages sequentially
        all_segments: list[SourceSegment] = []
        seg_counter = 1
        total_ocr_regions_executed = 0
        ocr_budget_exhausted = False
        table_api_available = hasattr(pymupdf.Page, "find_tables")
        ocr_encountered_issues = False

        for page_idx in range(doc.page_count):
            page_num = page_idx + 1
            try:
                page = doc[page_idx]
                page_rect = page.rect
                page_width = float(page_rect.width)
                page_height = float(page_rect.height)
                page_area = max(1.0, page_width * page_height)
                rotation = int(page.rotation)

                page_native_segs: list[SourceSegment] = []

                # A. Extract Tables if find_tables is available
                extracted_table_bboxes: list[pymupdf.Rect] = []
                if table_api_available:
                    try:
                        table_tabs = page.find_tables()
                        if table_tabs and table_tabs.tables:
                            for t_idx, t in enumerate(table_tabs.tables):
                                table_id = f"p{page_num}t{t_idx + 1}"
                                t_rows = t.extract()
                                if not t_rows or all(not any(c for c in r) for r in t_rows):
                                    warnings.append(
                                        StructuredError(
                                            code=WarningCode.TABLE_STRUCTURE_UNCERTAIN.value,
                                            stage="source_reading",
                                            message=f"Table structure uncertain on page {page_num}; falling back to native lines.",
                                            details={"page": page_num, "table_id": table_id},
                                        )
                                    )
                                    continue

                                t_bbox = pymupdf.Rect(t.bbox)
                                extracted_table_bboxes.append(t_bbox)

                                for row_idx, row_cells in enumerate(t_rows):
                                    row_id = f"{table_id}r{row_idx + 1}"
                                    row_bbox = (
                                        list(t.rows[row_idx].bbox)
                                        if row_idx < len(t.rows)
                                        else list(t.bbox)
                                    )
                                    is_header = row_idx == 0

                                    # Join cells with " | "
                                    cleaned_cells = [str(c or "").strip() for c in row_cells]
                                    row_text = " | ".join(cleaned_cells)
                                    if not row_text.strip():
                                        continue

                                    # Compute per-cell char spans
                                    cell_spans: list[tuple[int, int]] = []
                                    cursor = 0
                                    for c in cleaned_cells:
                                        c_len = len(c)
                                        cell_spans.append((cursor, cursor + c_len))
                                        cursor += c_len + 3  # len(" | ") == 3

                                    seg_id = f"p{page_num}_s{seg_counter:04d}"
                                    seg_counter += 1

                                    loc = PdfLocation(
                                        page_number=page_num,
                                        bbox=row_bbox,
                                        coordinate_system="pdf_points_top_left",
                                        page_width=page_width,
                                        page_height=page_height,
                                        rotation=rotation,
                                        table_id=table_id,
                                        row_id=row_id,
                                        cell_char_spans=cell_spans,
                                        extraction_method="pymupdf:table",
                                    )
                                    page_native_segs.append(
                                        SourceSegment(
                                            segment_id=seg_id,
                                            text=row_text,
                                            extraction_method="pymupdf:table",
                                            source_location=loc,
                                            page_id=f"page_{page_num}",
                                            table_id=table_id,
                                            row_id=row_id,
                                            is_table_header=is_header,
                                            table_row_cells=cleaned_cells,
                                            table_cell_spans=cell_spans,
                                        )
                                    )
                    except Exception as tab_err:
                        logger.warning(
                            "Table extraction failed on page %d: %s",
                            page_num,
                            type(tab_err).__name__,
                        )
                        warnings.append(
                            StructuredError(
                                code=WarningCode.TABLE_STRUCTURE_UNCERTAIN.value,
                                stage="source_reading",
                                message=f"Table extraction failed on page {page_num}: {type(tab_err).__name__}",
                                details={"page": page_num},
                            )
                        )

                # B. Extract Native Lines (skip lines that lie inside extracted tables)
                native_words = page.get_text("words") or []
                lines_by_block_line: dict[tuple[int, int], list[Any]] = defaultdict(list)
                for w in native_words:
                    # w: (x0, y0, x1, y1, word, block_no, line_no, word_no)
                    w_rect = pymupdf.Rect(w[0], w[1], w[2], w[3])
                    # Check if word falls inside an already extracted table
                    in_table = any(w_rect.intersects(tbox) for tbox in extracted_table_bboxes)
                    if not in_table:
                        lines_by_block_line[(int(w[5]), int(w[6]))].append(w)

                for (b_no, l_no), words in lines_by_block_line.items():
                    sorted_words = sorted(words, key=lambda x: x[0])
                    line_text = " ".join(str(w[4]) for w in sorted_words).strip()
                    if not line_text:
                        continue

                    lx0 = float(min(w[0] for w in sorted_words))
                    ly0 = float(min(w[1] for w in sorted_words))
                    lx1 = float(max(w[2] for w in sorted_words))
                    ly1 = float(max(w[3] for w in sorted_words))

                    block_id = f"p{page_num}b{b_no}"
                    line_id = f"p{page_num}b{b_no}l{l_no}"
                    seg_id = f"p{page_num}_s{seg_counter:04d}"
                    seg_counter += 1

                    loc = PdfLocation(
                        page_number=page_num,
                        bbox=[lx0, ly0, lx1, ly1],
                        coordinate_system="pdf_points_top_left",
                        page_width=page_width,
                        page_height=page_height,
                        rotation=rotation,
                        block_id=block_id,
                        line_id=line_id,
                        extraction_method="pymupdf:native",
                    )
                    page_native_segs.append(
                        SourceSegment(
                            segment_id=seg_id,
                            text=line_text,
                            extraction_method="pymupdf:native",
                            source_location=loc,
                            page_id=f"page_{page_num}",
                            layout_group_id=block_id,
                        )
                    )

                # Count native alphanumeric characters on the page
                total_page_native_text = "".join(s.text for s in page_native_segs)
                native_alnum_count = len(ALNUM_REGEX.findall(total_page_native_text))

                # C. Detect Embedded Image Regions
                image_info_list = page.get_image_info() or []
                detected_image_regions: list[dict[str, Any]] = []
                total_raster_area = 0.0

                for img_idx, img_info in enumerate(image_info_list):
                    bbox_raw = img_info.get("bbox")
                    if bbox_raw:
                        ix0, iy0, ix1, iy1 = [float(coord) for coord in bbox_raw]
                        i_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                        fraction = i_area / page_area
                        total_raster_area += i_area

                        # Count native alphanumeric characters lying inside this image bbox
                        img_rect = pymupdf.Rect(ix0, iy0, ix1, iy1)
                        img_native_words = [
                            w for w in native_words if img_rect.intersects(pymupdf.Rect(w[:4]))
                        ]
                        img_native_alnum = len(
                            ALNUM_REGEX.findall("".join(w[4] for w in img_native_words))
                        )

                        detected_image_regions.append(
                            {
                                "index": img_idx,
                                "bbox": [ix0, iy0, ix1, iy1],
                                "area": i_area,
                                "fraction": fraction,
                                "native_alnum_count": img_native_alnum,
                            }
                        )

                raster_fraction = min(1.0, total_raster_area / page_area)

                # D. Plan Page OCR
                plan = plan_page_ocr(
                    page_num=page_num,
                    native_alnum_count=native_alnum_count,
                    image_regions=detected_image_regions,
                    raster_area_fraction=raster_fraction,
                    ocr_mode=options.ocr_mode,
                    settings=self.settings,
                )
                page_plans.append(plan)

                if plan["decision"] == "none" and options.ocr_mode == "never":
                    if native_alnum_count < self.settings.pdf_min_native_alnum:
                        warnings.append(
                            StructuredError(
                                code=WarningCode.PAGE_TEXT_DEFICIENT_OCR_DISABLED.value,
                                stage="source_reading",
                                message=f"Page {page_num} has deficient text ({native_alnum_count} alnum chars), but OCR mode is 'never'.",
                                details={"page": page_num, "alnum_count": native_alnum_count},
                            )
                        )

                # E. Execute Planned OCR
                page_ocr_segs: list[SourceSegment] = []

                if plan["decision"] != "none":
                    ocr_targets: list[dict[str, Any]] = []
                    if plan["decision"] == "full_page":
                        ocr_targets = [
                            {
                                "bbox": [0.0, 0.0, page_width, page_height],
                                "is_full_page": True,
                            }
                        ]
                    else:
                        ocr_targets = plan["targets"]

                    for target in ocr_targets:
                        # Check region call budget
                        if total_ocr_regions_executed >= self.settings.max_ocr_regions_per_request:
                            ocr_budget_exhausted = True
                            ocr_encountered_issues = True
                            warnings.append(
                                StructuredError(
                                    code=WarningCode.OCR_SKIPPED_RENDER_BUDGET.value,
                                    stage="source_reading",
                                    message=f"OCR region call cap ({self.settings.max_ocr_regions_per_request}) reached on page {page_num}; skipping remaining OCR regions.",
                                    details={"page": page_num},
                                )
                            )
                            break

                        target_bbox = target["bbox"]
                        clip_rect = pymupdf.Rect(target_bbox)

                        # Stepwise render pixel budgeting
                        w_pts = clip_rect.width
                        h_pts = clip_rect.height
                        chosen_dpi: int | None = None

                        for test_dpi in [self.settings.ocr_dpi, 150, self.settings.ocr_min_dpi]:
                            scale = test_dpi / 72.0
                            rendered_pixels = int(w_pts * scale) * int(h_pts * scale)
                            if rendered_pixels <= self.settings.max_render_pixels:
                                chosen_dpi = test_dpi
                                break

                        if chosen_dpi is None:
                            ocr_encountered_issues = True
                            warnings.append(
                                StructuredError(
                                    code=WarningCode.OCR_SKIPPED_RENDER_BUDGET.value,
                                    stage="source_reading",
                                    message=f"Page {page_num} region exceeds MAX_RENDER_PIXELS ({self.settings.max_render_pixels:,}) even at minimum DPI ({self.settings.ocr_min_dpi}); skipping OCR.",
                                    details={"page": page_num, "bbox": target_bbox},
                                )
                            )
                            continue

                        # Render pixmap safely within budget
                        try:
                            pix = page.get_pixmap(dpi=chosen_dpi, clip=clip_rect)
                            png_bytes = pix.tobytes("png")
                            total_ocr_regions_executed += 1

                            # Invoke shared OcrService
                            ocr_result = self.ocr_service.ocr(
                                image_bytes=png_bytes,
                                language=options.ocr_languages,
                                timeout_seconds=self.settings.ocr_timeout_seconds,
                                max_pixels=self.settings.max_render_pixels,
                            )

                            # Map OCR lines to PDF coordinate space
                            scale = chosen_dpi / 72.0
                            for line in ocr_result.lines:
                                if not line.text.strip():
                                    continue

                                px0, py0, px1, py1 = line.bbox
                                # Coordinate transform from pixel to PDF points
                                pdf_x0 = clip_rect.x0 + (px0 / scale)
                                pdf_y0 = clip_rect.y0 + (py0 / scale)
                                pdf_x1 = clip_rect.x0 + (px1 / scale)
                                pdf_y1 = clip_rect.y0 + (py1 / scale)

                                # Word-level confidences
                                word_confs = [
                                    w.confidence for w in line.words if w.confidence is not None
                                ]

                                seg_id = f"p{page_num}_s{seg_counter:04d}"
                                seg_counter += 1

                                loc = PdfLocation(
                                    page_number=page_num,
                                    bbox=[pdf_x0, pdf_y0, pdf_x1, pdf_y1],
                                    coordinate_system="pdf_points_top_left",
                                    page_width=page_width,
                                    page_height=page_height,
                                    rotation=rotation,
                                    block_id=f"p{page_num}ocr_b{line.block_id}",
                                    line_id=f"p{page_num}ocr_b{line.block_id}l{line.line_id}",
                                    render_dpi=chosen_dpi,
                                    render_width=pix.width,
                                    render_height=pix.height,
                                    extraction_method="ocr:tesseract",
                                    ocr_words=[w.model_dump() for w in line.words],
                                    layout_group_id=f"ocr_b{line.block_id}",
                                )

                                page_ocr_segs.append(
                                    SourceSegment(
                                        segment_id=seg_id,
                                        text=line.text,
                                        extraction_method="ocr:tesseract",
                                        source_location=loc,
                                        page_id=f"page_{page_num}",
                                        layout_group_id=f"ocr_b{line.block_id}",
                                        ocr_word_confidences=word_confs if word_confs else None,
                                    )
                                )

                        except OcrUnavailableError as ocr_unavail:
                            ocr_encountered_issues = True
                            warnings.append(
                                StructuredError(
                                    code=ErrorCode.OCR_UNAVAILABLE.value,
                                    stage="source_reading",
                                    message=f"OCR unavailable on page {page_num}: {ocr_unavail.message}",
                                    details={"page": page_num},
                                )
                            )
                            break
                        except Exception as ocr_err:
                            ocr_encountered_issues = True
                            logger.warning(
                                "OCR execution error on page %d: %s",
                                page_num,
                                type(ocr_err).__name__,
                            )
                            warnings.append(
                                StructuredError(
                                    code=ErrorCode.OCR_TIMEOUT.value
                                    if "timeout" in type(ocr_err).__name__.lower()
                                    else ErrorCode.IMAGE_UNREADABLE.value,
                                    stage="source_reading",
                                    message=f"OCR execution failed on page {page_num}: {type(ocr_err).__name__}",
                                    details={"page": page_num},
                                )
                            )

                # F. Deduplicate Native vs OCR Segments (Same page only)
                deduped_ocr_segs: list[SourceSegment] = []
                for ocr_seg in page_ocr_segs:
                    norm_ocr = " ".join(
                        unicodedata.normalize("NFC", ocr_seg.text).casefold().split()
                    )
                    drop_ocr = False

                    for nat_seg in page_native_segs:
                        norm_nat = " ".join(
                            unicodedata.normalize("NFC", nat_seg.text).casefold().split()
                        )
                        iou = compute_bbox_iou(
                            ocr_seg.source_location.bbox
                            if hasattr(ocr_seg.source_location, "bbox")
                            else None,
                            nat_seg.source_location.bbox
                            if hasattr(nat_seg.source_location, "bbox")
                            else None,
                        )

                        if iou >= self.settings.dedup_iou_threshold:
                            if norm_ocr == norm_nat:
                                # Exact duplicate with high IoU -> suppress OCR segment
                                drop_ocr = True
                                break
                            else:
                                # Overlapping bounding box with differing text
                                warnings.append(
                                    StructuredError(
                                        code=WarningCode.POSSIBLE_NATIVE_OCR_DUPLICATE.value,
                                        stage="source_reading",
                                        message=f"High bbox overlap ({iou:.2f}) between native and OCR segments with differing text on page {page_num}.",
                                        details={
                                            "page": page_num,
                                            "native_id": nat_seg.segment_id,
                                            "ocr_id": ocr_seg.segment_id,
                                            "iou": round(iou, 3),
                                        },
                                    )
                                )

                    if not drop_ocr:
                        deduped_ocr_segs.append(ocr_seg)

                # Combine native and accepted OCR segments for this page
                all_segments.extend(page_native_segs)
                all_segments.extend(deduped_ocr_segs)

            except Exception as page_err:
                logger.error(
                    "Unexpected processing error on page %d: %s", page_num, type(page_err).__name__
                )
                warnings.append(
                    StructuredError(
                        code=WarningCode.PAGE_PROCESSING_FAILED.value,
                        stage="source_reading",
                        message=f"Processing failed unexpectedly on page {page_num}: {type(page_err).__name__}",
                        details={"page": page_num},
                    )
                )

        # Close PyMuPDF document
        total_pages = doc.page_count
        doc.close()

        # 7. Evaluate overall bundle status
        total_chars = sum(len(s.text) for s in all_segments)

        if not all_segments or total_chars == 0:
            err = StructuredError(
                code=ErrorCode.NO_USABLE_SOURCE_CONTENT.value,
                stage="source_reading",
                message="No usable clinical text or OCR content could be extracted from any page of the PDF document.",
                details={"page_count": total_pages},
            )
            errors.append(err)
            stage_res = StageResult(
                stage_name="source_reading",
                status="failed",
                counts={"pages": total_pages, "segments": 0, "chars": 0},
            )
            return None, "failed", errors, warnings, stage_res, page_plans

        # Determine if status is partial (due to skipped/failed OCR or page errors)
        if ocr_encountered_issues or ocr_budget_exhausted:
            overall_status = "partial"
        else:
            overall_status = "success"

        stage_res = StageResult(
            stage_name="source_reading",
            status=overall_status,
            counts={
                "pages": total_pages,
                "segments": len(all_segments),
                "chars": total_chars,
                "ocr_regions_executed": total_ocr_regions_executed,
            },
            details={"page_plans": page_plans},
        )

        safe_metadata = {
            "source_id": f"src_pdf_{content_sha256[:12]}",
            "media_type": "application/pdf",
            "page_count": total_pages,
            "char_count": total_chars,
            "byte_count": content_len,
            "content_sha256": content_sha256,
        }

        has_native = any(
            getattr(s.source_location, "extraction_method", "") == "native" for s in all_segments
        )
        has_ocr = any(
            getattr(s.source_location, "extraction_method", "") == "ocr" for s in all_segments
        )
        bundle_method = "hybrid" if (has_native and has_ocr) else ("ocr" if has_ocr else "native")

        bundle = SourceTextBundle(
            request_id=request_id,
            source_type="pdf",
            safe_metadata=safe_metadata,
            extraction_method=bundle_method,
            ordered_segments=all_segments,
        )

        return bundle, overall_status, errors, warnings, stage_res, page_plans
