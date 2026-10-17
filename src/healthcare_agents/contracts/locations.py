"""
Multi-modal Source Location Contracts.

Provides a discriminated union `SourceLocation` covering text, pdf, image, and audio.
Zero provider or agent dependencies.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


class TextLocation(BaseModel):
    """
    Zero-based Unicode code-point offsets into the original text string.
    start and end correspond to Python str slicing: raw_text[start:end].
    """

    kind: Literal["text"] = "text"
    start: int = Field(ge=0, description="Start index in Unicode code points (inclusive).")
    end: int = Field(gt=0, description="End index in Unicode code points (exclusive).")

    @model_validator(mode="after")
    def validate_span(self) -> "TextLocation":
        if self.end <= self.start:
            raise ValueError(
                f"end index ({self.end}) must be strictly greater than start ({self.start})"
            )
        return self


class PdfLocation(BaseModel):
    """
    Location within a PDF document.
    Coordinates are in PDF points (72 points/inch) or normalized points, specified by coordinate_system.
    """

    kind: Literal["pdf"] = "pdf"
    page_number: int = Field(ge=1, description="1-indexed physical page number.")
    bbox: list[float] | None = Field(
        default=None,
        description="Bounding box [x0, y0, x1, y1] in coordinate_system units.",
    )
    coordinate_system: str | None = Field(
        default=None,
        description="Mandatory when bbox is specified (e.g. 'pdf_points_top_left', 'pdf_points_bottom_left').",
    )
    page_width: float | None = Field(default=None, ge=0.0)
    page_height: float | None = Field(default=None, ge=0.0)
    rotation: int | None = Field(
        default=None, description="Page rotation in degrees (0, 90, 180, 270)."
    )
    block_id: str | None = None
    line_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    cell_id: str | None = None
    render_dpi: int | None = Field(default=None, ge=72)
    render_width: int | None = Field(default=None, ge=1)
    render_height: int | None = Field(default=None, ge=1)
    cell_char_spans: list[tuple[int, int]] | None = None
    extraction_method: str | None = None
    ocr_words: list[dict[str, Any]] | None = None
    layout_group_id: str | None = None

    @model_validator(mode="after")
    def validate_bbox_coordinate_system(self) -> "PdfLocation":
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain exactly 4 floats [x0, y0, x1, y1]")
            if not self.coordinate_system:
                raise ValueError(
                    "coordinate_system is mandatory when bbox is present in PdfLocation"
                )
        return self


class ImageLocation(BaseModel):
    """
    Location within an image document (scanned slip, prescription photo).
    Bounding box in original image pixel coordinates.
    """

    kind: Literal["image"] = "image"
    bbox: list[float] = Field(
        description="Bounding box [x0, y0, x1, y1] in original pixel coordinates."
    )
    original_width: int = Field(ge=1, description="Original image width in pixels.")
    original_height: int = Field(ge=1, description="Original image height in pixels.")
    coordinate_system: str = Field(
        default="pixel_top_left",
        description="Coordinate system specification (e.g. 'pixel_top_left', 'oriented_pixel_top_left').",
    )
    exif_orientation: int | None = Field(
        default=None,
        description="EXIF orientation tag value (1-8) if present in source image.",
    )
    oriented_width: int | None = Field(
        default=None,
        ge=1,
        description="Image width in pixels after applying EXIF transposition.",
    )
    oriented_height: int | None = Field(
        default=None,
        ge=1,
        description="Image height in pixels after applying EXIF transposition.",
    )
    extraction_method: str | None = Field(
        default=None,
        description="Extraction method identifier (e.g. 'ocr:tesseract').",
    )
    ocr_words: list[dict[str, Any]] | None = Field(
        default=None,
        description="Fine-grained word tokens with bboxes and confidence scores.",
    )
    layout_group_id: str | None = Field(
        default=None,
        description="Visual or layout grouping ID (e.g. 'block_1_par_1').",
    )

    @model_validator(mode="after")
    def validate_image_coords(self) -> "ImageLocation":
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly 4 numbers [x0, y0, x1, y1]")
        return self


class AudioLocation(BaseModel):
    """
    Temporal location within an audio stream.
    timestamps_available=False indicates audio where offsets cannot be determined.
    """

    kind: Literal["audio"] = "audio"
    timestamps_available: bool = Field(
        default=True,
        description="Whether temporal offsets are available.",
    )
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_audio_timestamps(self) -> "AudioLocation":
        if not self.timestamps_available:
            if self.start_seconds is not None or self.end_seconds is not None:
                raise ValueError(
                    "start_seconds and end_seconds must both be None when timestamps_available is False"
                )
        else:
            if self.start_seconds is None or self.end_seconds is None:
                raise ValueError(
                    "start_seconds and end_seconds must be provided when timestamps_available is True"
                )
            if self.end_seconds < self.start_seconds:
                raise ValueError(
                    f"end_seconds ({self.end_seconds}) cannot be less than start_seconds ({self.start_seconds})"
                )
        return self


# Discriminated union of all four source location modalities
SourceLocation = Annotated[
    TextLocation | PdfLocation | ImageLocation | AudioLocation,
    Field(discriminator="kind"),
]
