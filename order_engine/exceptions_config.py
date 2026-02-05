"""
Data Quality Exceptions Configuration
=====================================
Manages approved exceptions for data quality checks.
Users can add entries to skip known-good items that would otherwise be flagged.
"""

import yaml
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class Exception:
    """Represents an approved exception."""
    code: str                    # Error code (e.g., REL001, WS001)
    exception_type: str          # Type of exception
    key_fields: Dict[str, Any]   # Fields that identify this exception
    note: str = ""               # User's note explaining why it's approved
    approved_by: str = ""        # Who approved it
    approved_date: str = ""      # When it was approved


# Error code definitions
ERROR_CODES = {
    # Whitespace Issues (WS)
    "WS001": "Leading/trailing whitespace",
    "WS002": "Double spaces in text",

    # Typos (TYP)
    "TYP001": "Potential typo - similar strings",
    "TYP002": "Known misspelling",

    # Relationship Issues (REL)
    "REL001": "Make appears under multiple manufacturers",
    "REL002": "Make appears as multiple vehicle types",
    "REL003": "Model appears under multiple makes",

    # Pricing Issues (PRC)
    "PRC001": "Missing retail price",
    "PRC002": "Missing cost",
    "PRC003": "Negative margin (cost > retail)",
    "PRC004": "Price below minimum for vehicle type",
    "PRC005": "Price above maximum for vehicle type",

    # Date Logic Issues (DT)
    "DT001": "Purchase date before order date",
    "DT002": "Hold date before purchase date",
    "DT003": "Sold date before hold date",
    "DT004": "Future date (beyond reasonable)",
    "DT005": "Date too far in past",
    "DT006": "Missing required date",
    "DT007": "Date sequence illogical",

    # Missing Data (MSS)
    "MSS001": "Missing VIN",
    "MSS002": "Missing Stock#",
    "MSS003": "Missing Manufacturer",
    "MSS004": "Missing Make",
    "MSS005": "Missing Model",
    "MSS006": "Missing Vehicle Type",
    "MSS007": "Missing Location",
    "MSS008": "Missing Status",

    # Duplicates (DUP)
    "DUP001": "Duplicate VIN",
    "DUP002": "Duplicate Stock#",

    # Status Issues (STS)
    "STS001": "Unit in status too long",
    "STS002": "Unexpected status in inventory",

    # Licensing (LIC)
    "LIC001": "Unit at unlicensed location",

    # Outliers - Not errors, but investigate (OUT)
    "OUT001": "Age outlier - unit on lot unusually long",
    "OUT002": "Price outlier - unusually high for type",
    "OUT003": "Price outlier - unusually low for type",
    "OUT004": "Margin outlier - unusually high",
    "OUT005": "Margin outlier - unusually low",
    "OUT006": "Status age outlier",
    "OUT007": "Velocity outlier - slow moving category",
}


class ExceptionsManager:
    """Manages the approved exceptions file."""

    DEFAULT_FILE = "data_quality_exceptions.yaml"

    def __init__(self, config_path: Path = None):
        if config_path is None:
            # Default to Claude Toolkit folder
            config_path = Path(__file__).parent.parent / self.DEFAULT_FILE
        self.config_path = Path(config_path)
        self._exceptions: Dict[str, List[Exception]] = {}
        self._load_exceptions()

    def _load_exceptions(self):
        """Load exceptions from YAML file."""
        if not self.config_path.exists():
            self._create_default_file()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            self._exceptions = {}
            for exc_data in data.get("approved_exceptions", []):
                code = exc_data.get("code", "")
                if code not in self._exceptions:
                    self._exceptions[code] = []

                self._exceptions[code].append(Exception(
                    code=code,
                    exception_type=exc_data.get("type", ""),
                    key_fields=exc_data.get("key_fields", {}),
                    note=exc_data.get("note", ""),
                    approved_by=exc_data.get("approved_by", ""),
                    approved_date=exc_data.get("approved_date", "")
                ))
        except Exception as e:
            print(f"Warning: Could not load exceptions file: {e}")
            self._exceptions = {}

    def _create_default_file(self):
        """Create a default exceptions file with examples."""
        default_content = {
            "# FTRV Data Quality Exceptions": None,
            "# Add entries below to approve items that are flagged but not actual errors": None,
            "# Each entry needs: code, type, key_fields, and optionally note/approved_by/approved_date": None,
            "approved_exceptions": [
                {
                    "code": "REL001",
                    "type": "make_manufacturer",
                    "key_fields": {
                        "make": "EXAMPLE_MAKE",
                        "manufacturers": ["MFG1", "MFG2"]
                    },
                    "note": "This make is legitimately sold by both manufacturers",
                    "approved_by": "",
                    "approved_date": ""
                },
                {
                    "code": "REL003",
                    "type": "model_make",
                    "key_fields": {
                        "model": "EXAMPLE_MODEL",
                        "makes": ["MAKE1", "MAKE2"]
                    },
                    "note": "Same model number used by related brands",
                    "approved_by": "",
                    "approved_date": ""
                },
                {
                    "code": "TYP001",
                    "type": "similar_strings",
                    "key_fields": {
                        "field": "Make",
                        "value1": "SERIES 7",
                        "value2": "SERIES 8"
                    },
                    "note": "These are different product lines, not typos",
                    "approved_by": "",
                    "approved_date": ""
                }
            ]
        }

        # Write with comments preserved
        yaml_content = """# FTRV Data Quality Exceptions
# =============================
# Add entries below to approve items that are flagged but not actual errors.
# Each entry needs: code, type, key_fields, and optionally note/approved_by/approved_date
#
# Error Code Reference:
# ---------------------
# REL001: Make appears under multiple manufacturers
# REL002: Make appears as multiple vehicle types
# REL003: Model appears under multiple makes
# TYP001: Potential typo - similar strings
# OUT001-OUT007: Outliers (investigate, not necessarily errors)
# See exceptions_config.py for full list
#
# Example entries are provided below - modify or delete as needed.

approved_exceptions:
  # Example: Make legitimately sold by multiple manufacturers
  # - code: REL001
  #   type: make_manufacturer
  #   key_fields:
  #     make: PUMA
  #     manufacturers: [PALOMINO, FOREST RIVER]
  #   note: "PUMA brand is sold by both Palomino and Forest River divisions"
  #   approved_by: Ray
  #   approved_date: "2026-02-05"

  # Example: Similar strings that are NOT typos
  # - code: TYP001
  #   type: similar_strings
  #   key_fields:
  #     field: Make
  #     value1: CLIPPER 3K SERIES
  #     value2: CLIPPER 4K SERIES
  #   note: "These are different product series, not typos"
  #   approved_by: Ray
  #   approved_date: "2026-02-05"

"""

        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

    def is_approved(self, code: str, key_fields: Dict[str, Any]) -> bool:
        """
        Check if an issue is in the approved exceptions list.

        Args:
            code: Error code (e.g., REL001)
            key_fields: Dict of fields that identify this specific issue

        Returns:
            True if this issue is approved (should be skipped)
        """
        if code not in self._exceptions:
            return False

        for exc in self._exceptions[code]:
            if self._matches_exception(exc, key_fields):
                return True

        return False

    def _matches_exception(self, exc: Exception, key_fields: Dict[str, Any]) -> bool:
        """Check if key_fields match an exception's key_fields."""
        exc_fields = exc.key_fields

        for key, value in exc_fields.items():
            if key not in key_fields:
                continue

            exc_value = value
            check_value = key_fields[key]

            # Handle list comparisons
            if isinstance(exc_value, list) and isinstance(check_value, list):
                if set(exc_value) != set(check_value):
                    return False
            elif isinstance(exc_value, list):
                if check_value not in exc_value:
                    return False
            elif isinstance(check_value, list):
                if exc_value not in check_value:
                    return False
            else:
                # String comparison (case insensitive)
                if str(exc_value).upper() != str(check_value).upper():
                    return False

        return True

    def add_exception(self, code: str, exception_type: str, key_fields: Dict[str, Any],
                     note: str = "", approved_by: str = "") -> None:
        """Add a new exception to the file."""
        exc = Exception(
            code=code,
            exception_type=exception_type,
            key_fields=key_fields,
            note=note,
            approved_by=approved_by,
            approved_date=datetime.now().strftime("%Y-%m-%d")
        )

        if code not in self._exceptions:
            self._exceptions[code] = []
        self._exceptions[code].append(exc)

        self._save_exceptions()

    def _save_exceptions(self):
        """Save exceptions back to YAML file."""
        all_exceptions = []
        for code, exc_list in self._exceptions.items():
            for exc in exc_list:
                all_exceptions.append({
                    "code": exc.code,
                    "type": exc.exception_type,
                    "key_fields": exc.key_fields,
                    "note": exc.note,
                    "approved_by": exc.approved_by,
                    "approved_date": exc.approved_date
                })

        data = {"approved_exceptions": all_exceptions}

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_exception_template(self, code: str) -> Dict[str, Any]:
        """Get a template for adding an exception of a given type."""
        templates = {
            "REL001": {
                "code": "REL001",
                "type": "make_manufacturer",
                "key_fields": {"make": "", "manufacturers": []},
                "note": "",
                "approved_by": "",
                "approved_date": ""
            },
            "REL002": {
                "code": "REL002",
                "type": "make_vehtype",
                "key_fields": {"make": "", "veh_types": []},
                "note": "",
                "approved_by": "",
                "approved_date": ""
            },
            "REL003": {
                "code": "REL003",
                "type": "model_make",
                "key_fields": {"model": "", "makes": []},
                "note": "",
                "approved_by": "",
                "approved_date": ""
            },
            "TYP001": {
                "code": "TYP001",
                "type": "similar_strings",
                "key_fields": {"field": "", "value1": "", "value2": ""},
                "note": "",
                "approved_by": "",
                "approved_date": ""
            },
        }
        return templates.get(code, {"code": code, "type": "", "key_fields": {}, "note": ""})


def get_error_description(code: str) -> str:
    """Get human-readable description for an error code."""
    return ERROR_CODES.get(code, f"Unknown error code: {code}")


def get_error_category(code: str) -> str:
    """Get the category for an error code."""
    prefix = code[:2] if len(code) >= 2 else code
    categories = {
        "WS": "Whitespace Issues",
        "TY": "Typos",
        "RE": "Relationship Inconsistencies",
        "PR": "Pricing Issues",
        "DT": "Date Logic Issues",
        "MS": "Missing Data",
        "DU": "Duplicates",
        "ST": "Status Anomalies",
        "LI": "Licensing Violations",
        "OU": "Outliers (Investigate)",
    }
    return categories.get(prefix, "Other")
