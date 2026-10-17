"""
Deterministic ValueParser and DateParser.

Pure rule-based parsing for clinical measurements, blood pressure, reference ranges, and observation dates.
Zero clinical judgment or unit normalization.
Zero provider SDK imports.
"""

import re
from calendar import isleap
from decimal import Decimal, InvalidOperation
from typing import Any

from healthcare_agents.contracts.facts import DateParseStatus
from healthcare_agents.contracts.parsing import ParseStatus, ValueParseResult
from healthcare_agents.sources.normalization import canonicalize_label


class ValueParser:
    """
    Deterministic parser for numerical values, blood pressures, and reference ranges.
    """

    NUMBER_REGEX = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
    BP_REGEX = re.compile(r"^(\d{2,3})\s*/\s*(\d{2,3})$")
    RANGE_TWO_SIDED_REGEX = re.compile(
        r"^([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|—|to)\s*([0-9]+(?:\.[0-9]+)?)$",
        re.IGNORECASE,
    )
    RANGE_ONE_SIDED_REGEX = re.compile(r"^(<=|>=|<|>|≤|≥)\s*([0-9]+(?:\.[0-9]+)?)$")

    def __init__(self, bp_labels: list[str] | None = None):
        raw_labels = bp_labels if bp_labels is not None else ["bp", "blood pressure"]
        self.bp_labels = {canonicalize_label(x) for x in raw_labels}

    @classmethod
    def parse(cls, raw_value: str | None, field_context: str | None = None) -> ValueParseResult:
        parser = cls()
        if raw_value is None:
            return ValueParseResult(
                raw_value="",
                value_kind="empty",
                parse_status=ParseStatus.NOT_STATED,
                rule_id="empty_value",
                explanation="No value provided",
            )
        clean = raw_value.strip()
        if not clean:
            return ValueParseResult(
                raw_value=raw_value,
                value_kind="empty",
                parse_status=ParseStatus.NOT_STATED,
                rule_id="empty_value",
                explanation="No value provided",
            )
        # Check commas
        if "," in clean:
            return ValueParseResult(
                raw_value=raw_value,
                value_kind="text",
                parse_status=ParseStatus.AMBIGUOUS,
                rule_id="ambiguous_comma_format",
                explanation="ambiguous comma format",
            )
        # Range or comparison check: "4.0 to 5.5", "< 0.05", ">= 10.0"
        if cls.RANGE_TWO_SIDED_REGEX.match(clean) or cls.RANGE_ONE_SIDED_REGEX.match(clean):
            res_range = parser.parse_reference_range(clean)
            if res_range.parse_status == ParseStatus.PARSED:
                return res_range
        # BP check
        bp_match = cls.BP_REGEX.match(clean)
        if bp_match:
            norm_ctx = canonicalize_label(field_context or "")
            if norm_ctx in parser.bp_labels or (
                field_context and "blood pressure" in field_context.lower()
            ):
                sys_val = int(bp_match.group(1))
                dia_val = int(bp_match.group(2))
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="blood_pressure",
                    parse_status=ParseStatus.PARSED,
                    systolic=sys_val,
                    diastolic=dia_val,
                    rule_id="blood_pressure_ratio",
                    explanation="blood_pressure_ratio",
                )
            else:
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="blood_pressure",
                    parse_status=ParseStatus.AMBIGUOUS,
                    rule_id="non_bp_slash_ratio",
                    explanation="without blood pressure context allowlist",
                )
        return parser.parse_value(clean, field_context or "")

    def parse_value(self, raw_value: str, raw_label: str) -> ValueParseResult:
        """
        Parses value_raw using deterministic rules.
        Preserves raw_value verbatim on every path.
        """
        raw_clean = raw_value.strip() if raw_value else ""
        if not raw_clean:
            return ValueParseResult(
                raw_value=raw_value,
                value_kind="empty",
                parse_status=ParseStatus.NOT_STATED,
                rule_id="empty_value",
                explanation="No value provided",
            )

        # 1. Comma Ambiguity Check (European decimal comma vs thousands separator vs Indian grouping)
        if "," in raw_clean:
            return ValueParseResult(
                raw_value=raw_value,
                value_kind="text",
                parse_status=ParseStatus.AMBIGUOUS,
                rule_id="comma_ambiguity",
                explanation="Value contains comma; decimal comma vs thousands separator ambiguous",
            )

        # 2. Blood Pressure (e.g. "120/80")
        bp_match = self.BP_REGEX.match(raw_clean)
        if bp_match:
            norm_label = canonicalize_label(raw_label)
            if norm_label in self.bp_labels:
                sys_val = int(bp_match.group(1))
                dia_val = int(bp_match.group(2))
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="blood_pressure",
                    parse_status=ParseStatus.PARSED,
                    systolic=sys_val,
                    diastolic=dia_val,
                    rule_id="bp_strict_match",
                    explanation=f"Parsed systolic {sys_val} and diastolic {dia_val} under allowlisted BP label",
                )
            else:
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="blood_pressure",
                    parse_status=ParseStatus.AMBIGUOUS,
                    rule_id="bp_label_not_allowlisted",
                    explanation=f"Ratio 'A/B' detected but label '{raw_label}' is not in configured BP_LABELS",
                )

        # 3. Pure Scalar Number
        if self.NUMBER_REGEX.match(raw_clean):
            try:
                dec_val = Decimal(raw_clean)
                return ValueParseResult(
                    raw_value=raw_value,
                    value_kind="number",
                    parse_status=ParseStatus.PARSED,
                    numeric_value=dec_val,
                    rule_id="scalar_number_decimal",
                    explanation="Parsed pure numeric scalar",
                )
            except InvalidOperation:
                pass

        # 4. Fallback: Arbitrary Text
        return ValueParseResult(
            raw_value=raw_value,
            value_kind="text",
            parse_status=ParseStatus.NOT_ATTEMPTED,
            rule_id="unstructured_text",
            explanation="Value is unstructured non-numeric text",
        )

    def parse_reference_range(self, raw_range: str) -> ValueParseResult:
        """
        Parses reference_range_raw into numerical bounds.
        """
        clean = raw_range.strip() if raw_range else ""
        if not clean:
            return ValueParseResult(
                raw_value=raw_range,
                value_kind="empty",
                parse_status=ParseStatus.NOT_STATED,
                rule_id="empty_reference_range",
                explanation="No reference range provided",
            )

        if "," in clean:
            return ValueParseResult(
                raw_value=raw_range,
                value_kind="numeric_range",
                parse_status=ParseStatus.AMBIGUOUS,
                rule_id="range_comma_ambiguity",
                explanation="Reference range contains comma; ambiguous separator",
            )

        # Two-sided range: A-B, A–B, A to B
        two_match = self.RANGE_TWO_SIDED_REGEX.match(clean)
        if two_match:
            try:
                low = Decimal(two_match.group(1))
                high = Decimal(two_match.group(2))
                if low > high:
                    return ValueParseResult(
                        raw_value=raw_range,
                        value_kind="numeric_range",
                        parse_status=ParseStatus.INVALID,
                        rule_id="range_low_gt_high",
                        explanation=f"Range lower bound {low} exceeds upper bound {high}",
                    )
                return ValueParseResult(
                    raw_value=raw_range,
                    value_kind="numeric_range",
                    parse_status=ParseStatus.PARSED,
                    range_low=low,
                    range_high=high,
                    range_low_inclusive=True,
                    range_high_inclusive=True,
                    rule_id="two_sided_range_inclusive",
                    explanation="Parsed two-sided inclusive range",
                )
            except InvalidOperation:
                return ValueParseResult(
                    raw_value=raw_range,
                    value_kind="numeric_range",
                    parse_status=ParseStatus.INVALID,
                    rule_id="range_decimal_invalid",
                    explanation="Failed to parse decimal bounds in range",
                )

        # One-sided range: <A, <=A, ≤A, >A, >=A, ≥A
        one_match = self.RANGE_ONE_SIDED_REGEX.match(clean)
        if one_match:
            op = one_match.group(1)
            bound_str = one_match.group(2)
            try:
                bound = Decimal(bound_str)
                if op in ("<", "<=", "≤"):
                    return ValueParseResult(
                        raw_value=raw_range,
                        value_kind="numeric_range",
                        parse_status=ParseStatus.PARSED,
                        range_low=None,
                        range_high=bound,
                        range_low_inclusive=None,
                        range_high_inclusive=(op in ("<=", "≤")),
                        rule_id="one_sided_upper_bound",
                        explanation=f"Parsed upper bound with operator {op}",
                    )
                else:
                    return ValueParseResult(
                        raw_value=raw_range,
                        value_kind="numeric_range",
                        parse_status=ParseStatus.PARSED,
                        range_low=bound,
                        range_high=None,
                        range_low_inclusive=(op in (">=", "≥")),
                        range_high_inclusive=None,
                        rule_id="one_sided_lower_bound",
                        explanation=f"Parsed lower bound with operator {op}",
                    )
            except InvalidOperation:
                pass

        return ValueParseResult(
            raw_value=raw_range,
            value_kind="numeric_range",
            parse_status=ParseStatus.AMBIGUOUS,
            rule_id="range_unrecognized_pattern",
            explanation="Unrecognized reference range pattern",
        )


class DateParser:
    """
    Deterministic parser for observation dates.
    Adheres strictly to configured DATE_ORDER ('UNKNOWN', 'DMY', 'MDY').
    """

    ISO_REGEX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
    NUMERIC_SLASH_REGEX = re.compile(r"^(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{4})$")

    DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    @classmethod
    def parse(cls, raw_date: str | None, date_order: str = "UNKNOWN") -> Any:
        from healthcare_agents.contracts import DateParseResult

        if not raw_date:
            return DateParseResult(
                raw_date="",
                parse_status=ParseStatus.NOT_STATED,
            )
        clean = raw_date.strip()
        if re.match(r"^\d{1,2}[/\.-]\d{1,2}$", clean):
            return DateParseResult(
                raw_date=raw_date,
                parse_status=ParseStatus.INVALID,
                safe_warning="missing year",
            )
        iso_val, status, rule = cls.parse_date(raw_date, date_order)
        warning = None
        if status == DateParseStatus.AMBIGUOUS:
            warning = f"Date {raw_date} is ambiguous between DMY and MDY"
        elif status == DateParseStatus.INVALID:
            warning = "Invalid ISO calendar date"
        applied_rule = "iso_ymd" if rule == "iso_standard" else rule
        return DateParseResult(
            raw_date=raw_date,
            parse_status=ParseStatus(status.value),
            parsed_iso_date=iso_val,
            applied_date_format_rule=applied_rule,
            safe_warning=warning,
        )

    @classmethod
    def parse_date(
        cls,
        raw_date: str,
        date_order: str = "UNKNOWN",
    ) -> tuple[str | None, DateParseStatus, str]:
        """
        Parses date_raw into ISO YYYY-MM-DD or flags ambiguity / invalid calendar status.
        Returns:
            tuple[parsed_iso_date, DateParseStatus, rule_applied]
        """
        clean = raw_date.strip() if raw_date else ""
        if not clean:
            return None, DateParseStatus.NOT_STATED, "empty_date"

        # 1. ISO Format YYYY-MM-DD
        iso_match = cls.ISO_REGEX.match(clean)
        if iso_match:
            year = int(iso_match.group(1))
            month = int(iso_match.group(2))
            day = int(iso_match.group(3))
            if cls._is_valid_calendar_date(year, month, day):
                return f"{year:04d}-{month:02d}-{day:02d}", DateParseStatus.PARSED, "iso_standard"
            else:
                return None, DateParseStatus.INVALID, "iso_invalid_calendar_date"

        # 2. Year-Only (e.g. "2021")
        if re.match(r"^\d{4}$", clean):
            return clean, DateParseStatus.PARSED, "year_only"

        # 3. Numeric Three-Part Date (D/M/Y or M/D/Y)
        num_match = cls.NUMERIC_SLASH_REGEX.match(clean)
        if num_match:
            p1 = int(num_match.group(1))
            p2 = int(num_match.group(2))
            year = int(num_match.group(3))

            order = date_order.upper()
            rule_id = f"numeric_{order.lower()}"
            if order == "UNKNOWN":
                if p1 > 12 and 1 <= p2 <= 12:
                    day, month = p1, p2
                    rule_id = "dmy_unambiguous_day_gt_12"
                elif p2 > 12 and 1 <= p1 <= 12:
                    month, day = p1, p2
                    rule_id = "mdy_unambiguous_day_gt_12"
                else:
                    return None, DateParseStatus.AMBIGUOUS, "date_order_unknown"
            elif order == "DMY":
                day, month = p1, p2
            elif order == "MDY":
                month, day = p1, p2
            else:
                return None, DateParseStatus.AMBIGUOUS, "unrecognized_date_order"

            if cls._is_valid_calendar_date(year, month, day):
                return f"{year:04d}-{month:02d}-{day:02d}", DateParseStatus.PARSED, rule_id
            else:
                return None, DateParseStatus.INVALID, f"invalid_calendar_date_{order.lower()}"

        # 4. Two-digit years, relative expressions, or free text -> Not Attempted
        return None, DateParseStatus.NOT_ATTEMPTED, "unsupported_date_format"

    @classmethod
    def _is_valid_calendar_date(cls, year: int, month: int, day: int) -> bool:
        if month < 1 or month > 12:
            return False
        if day < 1:
            return False

        max_days = cls.DAYS_IN_MONTH[month]
        if month == 2 and isleap(year):
            max_days = 29

        return day <= max_days
