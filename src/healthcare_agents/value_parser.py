"""Deterministic value and date parsers with typed result contracts.

Parses clinical measurements, ranges, comparators, and dates using an explicit,
documented grammar without LLM guessing, semantic inference, or silent corrections.
"""

import re
import unicodedata
from datetime import date

from healthcare_agents.contracts import (
    DateParseResult,
    ParseStatus,
    ValueParseResult,
)

# Allowlist of context labels that justify interpreting a slash value (e.g. 120/80)
# as blood pressure (systolic/diastolic).
BLOOD_PRESSURE_CONTEXT_ALLOWLIST = {
    "bp",
    "b.p.",
    "blood pressure",
    "blood_pressure",
    "systolic/diastolic",
    "systolic / diastolic",
    "arterial pressure",
    "nibp",
}


class ValueParser:
    """Deterministic parser for clinical observation values.

    Supported patterns:
    - Scalar numbers: integer, decimal (e.g., '140', '140.5', '+5.2', '-2.1')
    - Comparison operators: '< 0.05', '<= 10', '> 100', '>= 12.5'
    - Ranges: '4.0 - 5.5', '4.0 to 5.5', '4.0–5.5' (en-dash), '4.0—5.5' (em-dash)
    - Blood pressure / ratios: '120/80' (only when context is explicitly in the BP allowlist)
    - Ambiguous numbers: decimal commas without clear format ('1,5', '1,000') flagged as ambiguous
    """

    # Regex patterns
    SCALAR_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)$")
    COMPARATOR_RE = re.compile(r"^([<>]=?|[≤≥])\s*([+-]?\d+(?:\.\d+)?)$")
    RANGE_RE = re.compile(
        r"^([+-]?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*([+-]?\d+(?:\.\d+)?)$",
        re.IGNORECASE,
    )
    SLASH_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$")
    # Ambiguous comma patterns: e.g. "1,5" or "1,000" or "1,234.56"
    COMMA_NUMBER_RE = re.compile(r"^[+-]?\d+,\d+(?:\.\d+)?$")

    @classmethod
    def parse(
        cls,
        raw_value: str | None,
        field_context: str | None = None,
    ) -> ValueParseResult:
        """Parse a raw value string into a typed ValueParseResult."""
        if raw_value is None or not raw_value.strip():
            return ValueParseResult(
                raw_value="" if raw_value is None else raw_value,
                value_kind="unknown",
                parse_status=ParseStatus.not_stated,
                safe_warning="No value provided in source.",
            )

        trimmed = raw_value.strip()

        # Check for ambiguous comma formats (e.g. "1,5" or "1,000")
        if cls.COMMA_NUMBER_RE.match(trimmed):
            return ValueParseResult(
                raw_value=raw_value,
                value_kind="unknown",
                parse_status=ParseStatus.ambiguous,
                rule_identifier="ambiguous_comma_format",
                safe_warning="Number contains comma separator without explicit locale format rule.",
            )

        # Check for Comparison Operator: e.g. "< 0.05", ">= 10"
        comp_match = cls.COMPARATOR_RE.match(trimmed)
        if comp_match:
            op_sym, num_str = comp_match.groups()
            # Normalize unicode comparison symbols
            if op_sym == "≤":
                op_sym = "<="
            elif op_sym == "≥":
                op_sym = ">="
            try:
                scalar_val = float(num_str)
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="scalar",
                    parse_status=ParseStatus.parsed,
                    parsed_scalar=scalar_val,
                    comparator=op_sym,
                    rule_identifier="comparator_scalar",
                )
            except ValueError:
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="scalar",
                    parse_status=ParseStatus.invalid,
                    safe_warning=f"Could not convert '{num_str}' to float.",
                )

        # Check for Scalar Number: e.g. "140", "140.5"
        scalar_match = cls.SCALAR_RE.match(trimmed)
        if scalar_match:
            try:
                scalar_val = float(scalar_match.group(1))
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="scalar",
                    parse_status=ParseStatus.parsed,
                    parsed_scalar=scalar_val,
                    rule_identifier="standard_scalar",
                )
            except ValueError:
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="scalar",
                    parse_status=ParseStatus.invalid,
                    safe_warning=f"Could not convert '{trimmed}' to float.",
                )

        # Check for Range: e.g. "4.0 - 5.5", "4.0 to 5.5", "4.0–5.5"
        range_match = cls.RANGE_RE.match(trimmed)
        if range_match:
            lower_str, upper_str = range_match.groups()
            try:
                lower_val = float(lower_str)
                upper_val = float(upper_str)
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="range",
                    parse_status=ParseStatus.parsed,
                    range_lower=lower_val,
                    range_upper=upper_val,
                    rule_identifier="numeric_range",
                )
            except ValueError:
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="range",
                    parse_status=ParseStatus.invalid,
                    safe_warning="Failed converting range bounds to float.",
                )

        # Check for Slash / Ratio: e.g. "120/80"
        slash_match = cls.SLASH_RE.match(trimmed)
        if slash_match:
            num_str, denom_str = slash_match.groups()
            is_bp_context = False
            if field_context:
                norm_ctx = unicodedata.normalize("NFC", field_context).strip().lower()
                norm_ctx = re.sub(r"\s+", " ", norm_ctx)
                if norm_ctx in BLOOD_PRESSURE_CONTEXT_ALLOWLIST:
                    is_bp_context = True

            if is_bp_context:
                try:
                    num_val = float(num_str)
                    denom_val = float(denom_str)
                    return ValueParseResult(
                        raw_value=raw_value,
                        value_kind="ratio",
                        parse_status=ParseStatus.parsed,
                        ratio_numerator=num_val,
                        ratio_denominator=denom_val,
                        rule_identifier="blood_pressure_ratio",
                    )
                except ValueError:
                    return ValueParseResult(
                        raw_value=raw_value,
                        value_kind="ratio",
                        parse_status=ParseStatus.invalid,
                        safe_warning="Failed converting blood pressure components to float.",
                    )
            else:
                # Do NOT infer blood pressure without allowlisted context!
                # Report as ratio or unknown with explanation
                try:
                    num_val = float(num_str)
                    denom_val = float(denom_str)
                    return ValueParseResult(
                        raw_value=raw_value,
                        value_kind="ratio",
                        parse_status=ParseStatus.ambiguous,
                        ratio_numerator=num_val,
                        ratio_denominator=denom_val,
                        rule_identifier="non_bp_slash_ratio",
                        safe_warning="Slash ratio encountered without blood pressure context allowlist match.",
                    )
                except ValueError:
                    return ValueParseResult(
                        raw_value=raw_value,
                        value_kind="ratio",
                        parse_status=ParseStatus.invalid,
                        safe_warning="Non-numeric slash value.",
                    )

        # Non-numeric or complex text value
        return ValueParseResult(
            raw_value=raw_value,
            value_kind="text",
            parse_status=ParseStatus.not_applicable,
            rule_identifier="raw_text_unparsed",
            safe_warning="Value is non-numeric text; numeric parsing not applicable.",
        )


class DateParser:
    """Deterministic parser for clinical observation dates.

    Rules:
    - ISO YYYY-MM-DD accepted unambiguously.
    - Configurable date order: UNKNOWN, DMY, MDY.
    - If date order is UNKNOWN, ambiguous dates like '03/04/2025' remain ambiguous.
    - If unambiguous mathematically (e.g. day > 12 in 25/04/2025) or configured DMY/MDY,
      parsed into standard ISO format.
    - Real calendar date validation (rejects invalid dates like 2025-02-31).
    - Missing years are never inferred.
    """

    ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
    SLASH_DOT_DASH_DATE_RE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")

    @classmethod
    def parse(
        cls,
        raw_date: str | None,
        date_order: str = "UNKNOWN",
    ) -> DateParseResult:
        """Parse raw date string using strict calendar validation and configured date order."""
        if raw_date is None or not raw_date.strip():
            return DateParseResult(
                raw_date="" if raw_date is None else raw_date,
                parse_status=ParseStatus.not_stated,
                safe_warning="No observation date provided.",
            )

        trimmed = raw_date.strip()
        date_order = date_order.upper()

        # 1. ISO 8601 (YYYY-MM-DD)
        iso_match = cls.ISO_DATE_RE.match(trimmed)
        if iso_match:
            year_i, month_i, day_i = map(int, iso_match.groups())
            try:
                d = date(year_i, month_i, day_i)
                return DateParseResult(
                    raw_date=raw_date,
                    parse_status=ParseStatus.parsed,
                    parsed_iso_date=d.isoformat(),
                    applied_date_format_rule="iso_ymd",
                )
            except ValueError as err:
                return DateParseResult(
                    raw_date=raw_date,
                    parse_status=ParseStatus.invalid,
                    safe_warning=f"Invalid ISO calendar date: {err}",
                )

        # 2. Ambiguous or formatted two-component first (e.g., DD/MM/YYYY or MM/DD/YYYY)
        match_slash = cls.SLASH_DOT_DASH_DATE_RE.match(trimmed)
        if match_slash:
            part1_i, part2_i, year_i = map(int, match_slash.groups())

            # Check if one part is strictly > 12 (must be day)
            if part1_i > 12 and part2_i <= 12:
                # Definitely DMY
                try:
                    d = date(year_i, part2_i, part1_i)
                    return DateParseResult(
                        raw_date=raw_date,
                        parse_status=ParseStatus.parsed,
                        parsed_iso_date=d.isoformat(),
                        applied_date_format_rule="dmy_unambiguous_day_gt_12",
                    )
                except ValueError as err:
                    return DateParseResult(
                        raw_date=raw_date,
                        parse_status=ParseStatus.invalid,
                        safe_warning=f"Invalid calendar date: {err}",
                    )

            if part2_i > 12 and part1_i <= 12:
                # Definitely MDY
                try:
                    d = date(year_i, part1_i, part2_i)
                    return DateParseResult(
                        raw_date=raw_date,
                        parse_status=ParseStatus.parsed,
                        parsed_iso_date=d.isoformat(),
                        applied_date_format_rule="mdy_unambiguous_day_gt_12",
                    )
                except ValueError as err:
                    return DateParseResult(
                        raw_date=raw_date,
                        parse_status=ParseStatus.invalid,
                        safe_warning=f"Invalid calendar date: {err}",
                    )

            # Both parts <= 12: ambiguous between DMY and MDY
            if part1_i <= 12 and part2_i <= 12:
                if date_order == "DMY":
                    try:
                        d = date(year_i, part2_i, part1_i)
                        return DateParseResult(
                            raw_date=raw_date,
                            parse_status=ParseStatus.parsed,
                            parsed_iso_date=d.isoformat(),
                            applied_date_format_rule="configured_dmy",
                        )
                    except ValueError as err:
                        return DateParseResult(
                            raw_date=raw_date,
                            parse_status=ParseStatus.invalid,
                            safe_warning=f"Invalid calendar date under DMY: {err}",
                        )
                elif date_order == "MDY":
                    try:
                        d = date(year_i, part1_i, part2_i)
                        return DateParseResult(
                            raw_date=raw_date,
                            parse_status=ParseStatus.parsed,
                            parsed_iso_date=d.isoformat(),
                            applied_date_format_rule="configured_mdy",
                        )
                    except ValueError as err:
                        return DateParseResult(
                            raw_date=raw_date,
                            parse_status=ParseStatus.invalid,
                            safe_warning=f"Invalid calendar date under MDY: {err}",
                        )
                else:
                    # DATE_ORDER == UNKNOWN
                    return DateParseResult(
                        raw_date=raw_date,
                        parse_status=ParseStatus.ambiguous,
                        safe_warning=f"Date '{trimmed}' is ambiguous between DMY and MDY with DATE_ORDER=UNKNOWN.",
                    )

        # Date format not recognized
        return DateParseResult(
            raw_date=raw_date,
            parse_status=ParseStatus.invalid,
            safe_warning=f"Unrecognized date expression or missing year in '{trimmed}'.",
        )
