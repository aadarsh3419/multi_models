"""
Value and Date parsing contract models.

Zero provider or agent dependencies.
"""

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ParseStatus(str, Enum):
    """Parsing outcome status for values and dates."""

    PARSED = "parsed"
    parsed = "parsed"
    AMBIGUOUS = "ambiguous"
    ambiguous = "ambiguous"
    INVALID = "invalid"
    invalid = "invalid"
    NOT_ATTEMPTED = "not_attempted"
    not_attempted = "not_attempted"
    NOT_STATED = "not_stated"
    not_stated = "not_stated"
    NOT_APPLICABLE = "not_applicable"
    not_applicable = "not_applicable"


class ValueParseResult(BaseModel):
    """
    Structured outcome of deterministic, rule-based value parsing.
    Maintains verbatim raw value without physiological inference or unit conversion.
    """

    raw_value: str = Field(description="Original, unmodified string from the clinical text.")
    value_kind: Literal["number", "blood_pressure", "numeric_range", "text", "empty"] = Field(
        description="Categorized shape of the raw value.",
    )
    parse_status: ParseStatus = Field(description="Deterministic status of the parse operation.")
    numeric_value: Decimal | None = Field(
        default=None, description="Parsed exact decimal for single scalar numbers."
    )
    systolic: int | None = Field(
        default=None, description="Systolic reading when label is an allowlisted BP label."
    )
    diastolic: int | None = Field(
        default=None, description="Diastolic reading when label is an allowlisted BP label."
    )
    range_low: Decimal | None = Field(default=None, description="Lower bound of parsed range.")
    range_high: Decimal | None = Field(default=None, description="Upper bound of parsed range.")
    range_low_inclusive: bool | None = Field(
        default=None, description="Whether lower bound is inclusive."
    )
    range_high_inclusive: bool | None = Field(
        default=None, description="Whether upper bound is inclusive."
    )
    rule_id: str = Field(
        default="unparsed", description="Identifier of the deterministic parsing rule applied."
    )
    explanation: str = Field(
        default="No parsing attempted", description="Static explanation of parsing decision."
    )

    @property
    def parsed_scalar(self) -> float | None:
        if self.numeric_value is not None:
            return float(self.numeric_value)
        if self.range_high is not None and self.range_low is None:
            return float(self.range_high)
        if self.range_low is not None and self.range_high is None:
            return float(self.range_low)
        return None

    @property
    def ratio_numerator(self) -> float | None:
        return float(self.systolic) if self.systolic is not None else None

    @property
    def ratio_denominator(self) -> float | None:
        return float(self.diastolic) if self.diastolic is not None else None

    @property
    def range_lower(self) -> float | None:
        return float(self.range_low) if self.range_low is not None else None

    @property
    def range_upper(self) -> float | None:
        return float(self.range_high) if self.range_high is not None else None

    @property
    def rule_identifier(self) -> str:
        return self.rule_id

    @property
    def safe_warning(self) -> str:
        return self.explanation

    @property
    def comparator(self) -> str | None:
        if "upper_bound" in self.rule_id:
            return "<=" if self.range_high_inclusive else "<"
        if "lower_bound" in self.rule_id:
            return ">=" if self.range_low_inclusive else ">"
        return None
