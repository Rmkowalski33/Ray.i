"""
Data Quality Analyzer Module
============================
Identifies data anomalies, outliers, and inconsistencies in inventory data.

Features:
- Error codes for each issue type (WS001, REL001, DT001, etc.)
- Exceptions file support - mark known-good items to skip
- Date logic validation - chronological order of events
- Outlier detection - items to investigate (not necessarily errors)
- Relationship consistency checks
- Missing data detection
- Pricing anomalies

Checks:
- Licensing violations (units at unlicensed locations)
- Text quality (whitespace, potential typos)
- Relationship consistency (manufacturer-make-model-type)
- Date logic (purchase, hold, sold dates in order)
- Numeric outliers (pricing, age, margin)
- Missing required data
- Duplicates
- Status anomalies
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import re

from .config import Config, default_config
from .data_loader import DataLoader
from .brand_licensing import BrandLicensingAnalyzer
from .exceptions_config import ExceptionsManager, ERROR_CODES, get_error_description, get_error_category


class DataQualityAnalyzer:
    """Analyze inventory data for quality issues and anomalies."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None,
                 exceptions_file: Path = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)
        self.brand_licensing = BrandLicensingAnalyzer(
            data_loader=self.data_loader, config=self.config
        )

        # Initialize exceptions manager
        self.exceptions = ExceptionsManager(exceptions_file)

        # Cache for loaded data
        self._inventory = None
        self._locations = None
        self._licensing = None

    def _load_data(self):
        """Load all required data."""
        if self._inventory is None:
            self._inventory = self.data_loader.load_current_inventory()
        if self._locations is None:
            self._locations = self.data_loader.load_locations()
        if self._licensing is None:
            self._licensing = self.data_loader.load_brand_licensing()

    def _is_approved(self, code: str, key_fields: Dict[str, Any]) -> bool:
        """Check if an issue is in the approved exceptions list."""
        return self.exceptions.is_approved(code, key_fields)

    def run_full_analysis(self, include_outliers: bool = True) -> Dict[str, Any]:
        """
        Run all data quality checks and return comprehensive report.

        Args:
            include_outliers: If True, include outlier detection (investigate, not errors)

        Returns:
            Dict with findings organized by category
        """
        self._load_data()

        print("Running data quality analysis...")
        print("-" * 50)

        results = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_units_analyzed": len(self._inventory) if self._inventory is not None else 0,
            "categories": {},
            "exceptions_file": str(self.exceptions.config_path)
        }

        # Core data quality checks
        checks = [
            ("Licensing Violations", self.check_licensing_violations),
            ("Text Quality Issues", self.check_text_quality),
            ("Whitespace Issues", self.check_whitespace_issues),
            ("Relationship Inconsistencies", self.check_relationship_consistency),
            ("Date Logic Issues", self.check_date_logic),  # NEW
            ("Model Year Issues", self.check_model_year_issues),
            ("Pricing Issues", self.check_pricing_issues),
            ("Missing Required Data", self.check_missing_data),
            ("Duplicate Detection", self.check_potential_duplicates),
            ("Status Anomalies", self.check_status_anomalies),
        ]

        # Outlier checks (investigate, not necessarily errors)
        if include_outliers:
            checks.extend([
                ("Age Outliers", self.check_age_outliers),  # NEW
                ("Margin Outliers", self.check_margin_outliers),  # NEW
            ])

        total_issues = 0
        total_outliers = 0

        for name, check_func in checks:
            print(f"  Checking {name}...")
            try:
                findings = check_func()
                results["categories"][name] = findings
                issue_count = findings.get("issue_count", 0)
                skipped_count = findings.get("skipped_approved", 0)

                if "Outlier" in name:
                    total_outliers += issue_count
                else:
                    total_issues += issue_count

                if issue_count > 0:
                    print(f"    -> Found {issue_count} items")
                if skipped_count > 0:
                    print(f"    -> Skipped {skipped_count} approved exceptions")

            except Exception as e:
                results["categories"][name] = {
                    "status": "error",
                    "message": str(e),
                    "issue_count": 0
                }
                print(f"    -> Error: {e}")

        results["total_issues"] = total_issues
        results["total_outliers"] = total_outliers
        print("-" * 50)
        print(f"Total issues found: {total_issues}")

        return results

    def check_licensing_violations(self) -> Dict[str, Any]:
        """
        Find units assigned to locations where the make is not licensed.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "violations": []}

        if self._licensing is None or self._licensing.empty:
            return {"status": "no_licensing_data", "issue_count": 0, "violations": []}

        violations = []

        # Build lookup of licensed makes by location
        licensed_lookup = defaultdict(set)
        for _, row in self._licensing.iterrows():
            loc = str(row.get("Location", "")).strip().upper()
            make = str(row.get("Inv_Make", row.get("BRAND SERIES SUBSERIES", ""))).strip().upper()
            if loc and make:
                licensed_lookup[loc].add(make)

        # Check each inventory unit
        for _, unit in self._inventory.iterrows():
            loc = str(unit.get("PC", "")).strip().upper()
            make = str(unit.get("Make", "")).strip().upper()

            # Skip units at central locations (PDI, YARD, CORP, etc.)
            if loc in ["PDI", "YARD", "CORP", "PDIT", ""] or not loc:
                continue

            # Skip if location not in licensing (might be new location)
            if loc not in licensed_lookup:
                continue

            # Check if make is licensed at this location
            if make and make not in licensed_lookup[loc]:
                violations.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "location": loc,
                    "make": make,
                    "manufacturer": unit.get("Manufacturer", ""),
                    "model": unit.get("Model", ""),
                    "status": unit.get("Status", ""),
                    "issue": f"Make '{make}' not licensed at {loc}"
                })

        return {
            "status": "completed",
            "issue_count": len(violations),
            "violations": violations,
            "description": "Units at locations where their make is not licensed"
        }

    def check_whitespace_issues(self) -> Dict[str, Any]:
        """
        Find fields with leading/trailing whitespace or double spaces.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        text_fields = ["Manufacturer", "Make", "Model", "Floorplan", "PC", "Status"]
        text_fields = [f for f in text_fields if f in self._inventory.columns]

        for _, unit in self._inventory.iterrows():
            for field in text_fields:
                value = unit.get(field, "")
                if pd.isna(value):
                    continue
                value = str(value)

                problems = []
                if value != value.strip():
                    problems.append("leading/trailing whitespace")
                if "  " in value:
                    problems.append("double spaces")

                if problems:
                    issues.append({
                        "stock_num": unit.get("Stock#", ""),
                        "field": field,
                        "value": repr(value),  # Show exact string with quotes
                        "clean_value": " ".join(value.split()),
                        "problems": ", ".join(problems)
                    })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "issues": issues,
            "description": "Text fields with whitespace problems"
        }

    def check_text_quality(self) -> Dict[str, Any]:
        """
        Find potential typos and inconsistent naming (TYP001).
        Uses fuzzy matching to find similar strings that might be duplicates.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        skipped = 0

        # Check each categorical field for similar values
        fields_to_check = ["Manufacturer", "Make"]

        for field in fields_to_check:
            if field not in self._inventory.columns:
                continue

            unique_values = self._inventory[field].dropna().astype(str).unique()
            unique_values = [v.strip() for v in unique_values if v.strip()]

            # Find similar pairs
            similar_pairs = self._find_similar_strings(unique_values, threshold=0.85)

            for val1, val2, similarity in similar_pairs:
                code = "TYP001"
                key_fields = {"field": field, "value1": val1, "value2": val2}

                # Check if this pair is approved (either direction)
                key_fields_reverse = {"field": field, "value1": val2, "value2": val1}
                if self._is_approved(code, key_fields) or self._is_approved(code, key_fields_reverse):
                    skipped += 1
                    continue

                count1 = len(self._inventory[self._inventory[field].astype(str).str.strip() == val1])
                count2 = len(self._inventory[self._inventory[field].astype(str).str.strip() == val2])

                issues.append({
                    "code": code,
                    "field": field,
                    "value_1": val1,
                    "count_1": count1,
                    "value_2": val2,
                    "count_2": count2,
                    "similarity": f"{similarity:.0%}",
                    "issue": f"Possible typo - '{val1}' vs '{val2}'"
                })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "skipped_approved": skipped,
            "issues": issues,
            "description": "Potential typos or inconsistent naming (similar strings)"
        }

    def _find_similar_strings(self, strings: List[str], threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """Find pairs of strings that are similar but not identical."""
        similar = []
        strings = list(set(strings))  # Dedupe

        for i, s1 in enumerate(strings):
            for s2 in strings[i+1:]:
                if s1.upper() == s2.upper():
                    # Exact match different case
                    similar.append((s1, s2, 1.0))
                else:
                    sim = self._string_similarity(s1.upper(), s2.upper())
                    if sim >= threshold:
                        similar.append((s1, s2, sim))

        return similar

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings using Levenshtein-like approach."""
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        # Simple similarity based on common characters
        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max(len1, len2) * 0.3:
            return 0.0  # Too different in length

        # Count matching character positions
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))

        # Also check for substring relationship
        if s1 in s2 or s2 in s1:
            return min(len1, len2) / max(len1, len2)

        return matches / max(len1, len2)

    def check_relationship_consistency(self) -> Dict[str, Any]:
        """
        Check for inconsistent relationships:
        - Same Make under different Manufacturers (REL001)
        - Same Make with different Vehicle Types (REL002)
        - Same Model under different Makes (REL003)
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        skipped = 0

        # Check Make -> Manufacturer consistency (REL001)
        if "Make" in self._inventory.columns and "Manufacturer" in self._inventory.columns:
            make_mfg = self._inventory.groupby("Make")["Manufacturer"].nunique()
            inconsistent_makes = make_mfg[make_mfg > 1].index.tolist()

            for make in inconsistent_makes:
                mfgs = self._inventory[self._inventory["Make"] == make]["Manufacturer"].unique()
                mfg_list = sorted(list(mfgs))
                counts = {
                    mfg: len(self._inventory[(self._inventory["Make"] == make) &
                                             (self._inventory["Manufacturer"] == mfg)])
                    for mfg in mfgs
                }

                code = "REL001"
                key_fields = {"make": make, "manufacturers": mfg_list}

                if self._is_approved(code, key_fields):
                    skipped += 1
                    continue

                issues.append({
                    "code": code,
                    "type": "Make-Manufacturer",
                    "entity": make,
                    "values": mfg_list,
                    "counts": counts,
                    "issue": f"Make '{make}' appears under {len(mfgs)} manufacturers: {', '.join(mfg_list)}"
                })

        # Check Make -> Veh Type consistency (REL002)
        if "Make" in self._inventory.columns and "Veh Type" in self._inventory.columns:
            make_type = self._inventory.groupby("Make")["Veh Type"].nunique()
            inconsistent_types = make_type[make_type > 1].index.tolist()

            for make in inconsistent_types:
                types = self._inventory[self._inventory["Make"] == make]["Veh Type"].unique()
                type_list = sorted(list(types))
                counts = {
                    t: len(self._inventory[(self._inventory["Make"] == make) &
                                           (self._inventory["Veh Type"] == t)])
                    for t in types
                }

                code = "REL002"
                key_fields = {"make": make, "veh_types": type_list}

                if self._is_approved(code, key_fields):
                    skipped += 1
                    continue

                # Only flag if it looks like an error (one type has very few units)
                total = sum(counts.values())
                if any(c < total * 0.05 for c in counts.values()):  # Less than 5% in one type
                    issues.append({
                        "code": code,
                        "type": "Make-VehType",
                        "entity": make,
                        "values": type_list,
                        "counts": counts,
                        "issue": f"Make '{make}' appears as {len(types)} vehicle types: {', '.join(type_list)}"
                    })

        # Check Model -> Make consistency (REL003)
        if "Model" in self._inventory.columns and "Make" in self._inventory.columns:
            model_make = self._inventory.groupby("Model")["Make"].nunique()
            inconsistent_models = model_make[model_make > 1].index.tolist()

            for model in inconsistent_models:
                makes = self._inventory[self._inventory["Model"] == model]["Make"].unique()
                make_list = sorted(list(makes))
                counts = {
                    m: len(self._inventory[(self._inventory["Model"] == model) &
                                           (self._inventory["Make"] == m)])
                    for m in makes
                }

                code = "REL003"
                key_fields = {"model": model, "makes": make_list}

                if self._is_approved(code, key_fields):
                    skipped += 1
                    continue

                issues.append({
                    "code": code,
                    "type": "Model-Make",
                    "entity": model,
                    "values": make_list,
                    "counts": counts,
                    "issue": f"Model '{model}' appears under {len(makes)} makes: {', '.join(make_list)}"
                })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "skipped_approved": skipped,
            "issues": issues,
            "description": "Inconsistent relationships between Manufacturer/Make/Model/Type"
        }

    def check_model_year_issues(self) -> Dict[str, Any]:
        """
        Find model years that don't make sense:
        - Too old (before reasonable RV age)
        - Future years beyond next year
        - Missing/invalid years
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        current_year = datetime.now().year
        min_valid_year = current_year - 15  # RVs older than 15 years unusual for dealer
        max_valid_year = current_year + 2   # Can have next year + 1 models

        year_col = None
        for col in ["Year", "Model Year", "MY"]:
            if col in self._inventory.columns:
                year_col = col
                break

        if not year_col:
            return {"status": "no_year_column", "issue_count": 0, "issues": []}

        for _, unit in self._inventory.iterrows():
            year = unit.get(year_col)

            if pd.isna(year) or year == "" or year == 0:
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "year": str(year),
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "issue": "Missing or invalid model year"
                })
                continue

            try:
                year_int = int(float(year))
            except (ValueError, TypeError):
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "year": str(year),
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "issue": f"Non-numeric model year: '{year}'"
                })
                continue

            if year_int < min_valid_year:
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "year": year_int,
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "issue": f"Model year {year_int} is unusually old (>{15} years)"
                })
            elif year_int > max_valid_year:
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "year": year_int,
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "issue": f"Model year {year_int} is in the future"
                })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "issues": issues,
            "valid_range": f"{min_valid_year}-{max_valid_year}",
            "description": "Model years outside expected range or missing"
        }

    def check_pricing_issues(self) -> Dict[str, Any]:
        """
        Find pricing anomalies:
        - Missing retail price
        - Missing cost
        - Cost > Retail (negative margin)
        - Extremely high or low prices
        - Zero prices
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []

        retail_col = None
        cost_col = None

        for col in ["Retail Price", "Retail", "MSRP", "Price"]:
            if col in self._inventory.columns:
                retail_col = col
                break

        for col in ["Total Cost", "Cost", "Invoice"]:
            if col in self._inventory.columns:
                cost_col = col
                break

        if not retail_col and not cost_col:
            return {"status": "no_price_columns", "issue_count": 0, "issues": []}

        # Define reasonable price bounds by vehicle type
        price_bounds = {
            "TT": (5000, 150000),
            "FW": (15000, 200000),
            "MH": (50000, 500000),
            "TH": (20000, 250000),
            "default": (5000, 500000)
        }

        for _, unit in self._inventory.iterrows():
            veh_type = str(unit.get("Veh Type", "")).upper()
            min_price, max_price = price_bounds.get(veh_type, price_bounds["default"])

            retail = unit.get(retail_col) if retail_col else None
            cost = unit.get(cost_col) if cost_col else None

            stock_info = {
                "stock_num": unit.get("Stock#", ""),
                "vin": unit.get("VIN", ""),
                "make": unit.get("Make", ""),
                "model": unit.get("Model", ""),
                "veh_type": veh_type
            }

            # Check missing retail
            if retail_col:
                if pd.isna(retail) or retail == 0:
                    issues.append({
                        **stock_info,
                        "retail": retail,
                        "cost": cost,
                        "issue": "Missing or zero retail price"
                    })
                elif retail < min_price:
                    issues.append({
                        **stock_info,
                        "retail": retail,
                        "cost": cost,
                        "issue": f"Retail ${retail:,.0f} below minimum ${min_price:,} for {veh_type}"
                    })
                elif retail > max_price:
                    issues.append({
                        **stock_info,
                        "retail": retail,
                        "cost": cost,
                        "issue": f"Retail ${retail:,.0f} above maximum ${max_price:,} for {veh_type}"
                    })

            # Check missing cost
            if cost_col:
                if pd.isna(cost) or cost == 0:
                    issues.append({
                        **stock_info,
                        "retail": retail,
                        "cost": cost,
                        "issue": "Missing or zero cost"
                    })

            # Check negative margin
            if retail_col and cost_col:
                if not pd.isna(retail) and not pd.isna(cost) and retail > 0 and cost > 0:
                    if cost > retail:
                        margin = (retail - cost) / retail * 100
                        issues.append({
                            **stock_info,
                            "retail": retail,
                            "cost": cost,
                            "issue": f"Negative margin: Cost ${cost:,.0f} > Retail ${retail:,.0f} ({margin:.1f}%)"
                        })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "issues": issues,
            "price_columns": {"retail": retail_col, "cost": cost_col},
            "description": "Pricing anomalies (missing, negative margin, outliers)"
        }

    def check_missing_data(self) -> Dict[str, Any]:
        """
        Check for missing values in required fields.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        required_fields = [
            "Stock#", "VIN", "Manufacturer", "Make", "Model",
            "Veh Type", "PC", "Status"
        ]
        required_fields = [f for f in required_fields if f in self._inventory.columns]

        issues = []
        field_summary = {}

        for field in required_fields:
            missing = self._inventory[
                self._inventory[field].isna() |
                (self._inventory[field].astype(str).str.strip() == "")
            ]

            if len(missing) > 0:
                field_summary[field] = len(missing)

                # Add all records for complete action list
                for _, unit in missing.iterrows():
                    issues.append({
                        "stock_num": unit.get("Stock#", ""),
                        "vin": unit.get("VIN", ""),
                        "field": field,
                        "make": unit.get("Make", ""),
                        "model": unit.get("Model", ""),
                        "location": unit.get("PC", ""),
                        "issue": f"Missing required field: {field}"
                    })

        return {
            "status": "completed",
            "issue_count": sum(field_summary.values()),
            "field_summary": field_summary,
            "issues": issues,
            "description": "Missing values in required fields"
        }

    def check_potential_duplicates(self) -> Dict[str, Any]:
        """
        Find potential duplicate records based on VIN or Stock#.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []

        # Check duplicate VINs
        if "VIN" in self._inventory.columns:
            vin_counts = self._inventory["VIN"].value_counts()
            dup_vins = vin_counts[vin_counts > 1]

            for vin, count in dup_vins.items():
                if pd.isna(vin) or str(vin).strip() == "":
                    continue
                dup_records = self._inventory[self._inventory["VIN"] == vin]
                issues.append({
                    "type": "Duplicate VIN",
                    "value": vin,
                    "count": count,
                    "stock_nums": dup_records["Stock#"].tolist() if "Stock#" in dup_records.columns else [],
                    "locations": dup_records["PC"].tolist() if "PC" in dup_records.columns else [],
                    "issue": f"VIN '{vin}' appears {count} times"
                })

        # Check duplicate Stock#s
        if "Stock#" in self._inventory.columns:
            stock_counts = self._inventory["Stock#"].value_counts()
            dup_stocks = stock_counts[stock_counts > 1]

            for stock, count in dup_stocks.items():
                if pd.isna(stock) or str(stock).strip() == "":
                    continue
                dup_records = self._inventory[self._inventory["Stock#"] == stock]
                issues.append({
                    "type": "Duplicate Stock#",
                    "value": stock,
                    "count": count,
                    "vins": dup_records["VIN"].tolist() if "VIN" in dup_records.columns else [],
                    "issue": f"Stock# '{stock}' appears {count} times"
                })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "issues": issues,
            "description": "Potential duplicate records"
        }

    def check_status_anomalies(self) -> Dict[str, Any]:
        """
        Find status-related anomalies:
        - Units in status for unusually long time
        - Unusual status combinations
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []

        # Check for long status ages
        if "Status Age" in self._inventory.columns and "Status" in self._inventory.columns:
            # Define max reasonable days by status
            status_max_days = {
                "ORDERED": 180,
                "PURCHASED": 90,
                "SHIPPED": 30,
                "DRIVER NEEDED": 14,
                "PRE-PDI": 14,
                "IN SERVICE": 45,
                "QC NEEDED": 7,
            }

            for _, unit in self._inventory.iterrows():
                status = str(unit.get("Status", "")).upper()
                age = unit.get("Status Age")

                if pd.isna(age):
                    continue

                try:
                    age_days = int(age)
                except (ValueError, TypeError):
                    continue

                max_days = status_max_days.get(status)
                if max_days and age_days > max_days:
                    issues.append({
                        "stock_num": unit.get("Stock#", ""),
                        "vin": unit.get("VIN", ""),
                        "status": status,
                        "status_age_days": age_days,
                        "max_expected": max_days,
                        "make": unit.get("Make", ""),
                        "model": unit.get("Model", ""),
                        "location": unit.get("PC", ""),
                        "issue": f"In '{status}' for {age_days} days (expected max {max_days})"
                    })

        # Check for sold/hold units still in active inventory
        if "Status" in self._inventory.columns:
            concerning_statuses = ["SOLD", "DELETED", "WHOLESALE", "CANCELLED"]
            for status in concerning_statuses:
                count = len(self._inventory[
                    self._inventory["Status"].astype(str).str.upper().str.contains(status, na=False)
                ])
                if count > 0:
                    issues.append({
                        "stock_num": "N/A",
                        "vin": "N/A",
                        "status": status,
                        "issue": f"{count} units with '{status}' status in inventory extract"
                    })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "issues": issues,
            "description": "Status-related anomalies (stuck units, unexpected statuses)"
        }

    def check_date_logic(self) -> Dict[str, Any]:
        """
        Check date fields for logical consistency:
        - Purchase date should be reasonable
        - Hold date should be after purchase date
        - Sold date should be after hold date
        - Dates shouldn't be in the future (with some exceptions)
        - Dates shouldn't be too far in the past
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        skipped = 0
        today = datetime.now()
        max_future_days = 7  # Allow a week into the future for data entry timing
        max_past_years = 3   # Flag dates more than 3 years old

        # Date column mappings
        date_cols = {
            "purch_date": "Purch. Date",
            "hold_date": "Hold Date",
            "sold_date": "Sold Date",
            "price_change_date": "Last Price Change Date"
        }

        # Filter to columns that exist
        available_cols = {k: v for k, v in date_cols.items() if v in self._inventory.columns}

        for idx, unit in self._inventory.iterrows():
            stock_num = unit.get("Stock#", "")

            # Get dates
            purch_date = self._parse_date(unit.get(available_cols.get("purch_date", "")))
            hold_date = self._parse_date(unit.get(available_cols.get("hold_date", "")))
            sold_date = self._parse_date(unit.get(available_cols.get("sold_date", "")))

            base_info = {
                "stock_num": stock_num,
                "vin": unit.get("VIN", ""),
                "make": unit.get("Make", ""),
                "model": unit.get("Model", ""),
                "location": unit.get("PC", ""),
            }

            # Check: Hold date before purchase date
            if purch_date and hold_date and hold_date < purch_date:
                code = "DT002"
                key_fields = {"stock_num": stock_num, "purch_date": str(purch_date), "hold_date": str(hold_date)}

                if self._is_approved(code, key_fields):
                    skipped += 1
                else:
                    issues.append({
                        **base_info,
                        "code": code,
                        "purch_date": str(purch_date.date()) if purch_date else "",
                        "hold_date": str(hold_date.date()) if hold_date else "",
                        "issue": f"Hold date ({hold_date.date()}) is before purchase date ({purch_date.date()})"
                    })

            # Check: Sold date before hold date
            if hold_date and sold_date and sold_date < hold_date:
                code = "DT003"
                issues.append({
                    **base_info,
                    "code": code,
                    "hold_date": str(hold_date.date()) if hold_date else "",
                    "sold_date": str(sold_date.date()) if sold_date else "",
                    "issue": f"Sold date ({sold_date.date()}) is before hold date ({hold_date.date()})"
                })

            # Check: Future dates (beyond reasonable)
            for date_name, date_val in [("Purchase", purch_date), ("Hold", hold_date), ("Sold", sold_date)]:
                if date_val and date_val > today + timedelta(days=max_future_days):
                    code = "DT004"
                    issues.append({
                        **base_info,
                        "code": code,
                        "date_field": date_name,
                        "date_value": str(date_val.date()),
                        "issue": f"{date_name} date ({date_val.date()}) is in the future"
                    })

            # Check: Very old dates
            for date_name, date_val in [("Purchase", purch_date), ("Hold", hold_date)]:
                if date_val and date_val < today - timedelta(days=max_past_years * 365):
                    code = "DT005"
                    issues.append({
                        **base_info,
                        "code": code,
                        "date_field": date_name,
                        "date_value": str(date_val.date()),
                        "issue": f"{date_name} date ({date_val.date()}) is more than {max_past_years} years old"
                    })

        return {
            "status": "completed",
            "issue_count": len(issues),
            "skipped_approved": skipped,
            "issues": issues,
            "description": "Date logic issues (illogical sequences, future/past dates)"
        }

    def _parse_date(self, value) -> Optional[datetime]:
        """Parse a date value from various formats."""
        if pd.isna(value) or value == "" or value == 0:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        try:
            return pd.to_datetime(value)
        except:
            return None

    def check_age_outliers(self) -> Dict[str, Any]:
        """
        Find units that are outliers by age - unusually long time on lot.
        These are not errors, but items to investigate.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        skipped = 0

        # Age thresholds by vehicle type (in days)
        age_thresholds = {
            "TT": 180,   # Travel Trailers: 6 months
            "FW": 210,   # Fifth Wheels: 7 months
            "MH": 270,   # Motorhomes: 9 months
            "TH": 210,   # Toy Haulers: 7 months
            "default": 180
        }

        age_col = None
        for col in ["Age", "PC Age", "Lot Age"]:
            if col in self._inventory.columns:
                age_col = col
                break

        if not age_col:
            return {"status": "no_age_column", "issue_count": 0, "issues": []}

        for _, unit in self._inventory.iterrows():
            veh_type = str(unit.get("Veh Type", "")).upper()
            threshold = age_thresholds.get(veh_type, age_thresholds["default"])

            age = unit.get(age_col)
            if pd.isna(age):
                continue

            try:
                age_days = int(age)
            except (ValueError, TypeError):
                continue

            if age_days > threshold:
                code = "OUT001"
                stock_num = unit.get("Stock#", "")

                # Check if approved
                key_fields = {"stock_num": stock_num, "age": age_days}
                if self._is_approved(code, key_fields):
                    skipped += 1
                    continue

                # Calculate how much over threshold
                pct_over = (age_days - threshold) / threshold * 100

                issues.append({
                    "stock_num": stock_num,
                    "vin": unit.get("VIN", ""),
                    "code": code,
                    "veh_type": veh_type,
                    "age_days": age_days,
                    "threshold_days": threshold,
                    "pct_over_threshold": round(pct_over, 1),
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "location": unit.get("PC", ""),
                    "retail_price": unit.get("Retail Price", 0),
                    "issue": f"Age {age_days} days exceeds {threshold} day threshold for {veh_type} (+{pct_over:.0f}%)"
                })

        # Sort by age descending
        issues.sort(key=lambda x: x.get("age_days", 0), reverse=True)

        return {
            "status": "completed",
            "issue_count": len(issues),
            "skipped_approved": skipped,
            "issues": issues,
            "description": "Age outliers - units on lot longer than typical (investigate)",
            "is_outlier_check": True  # Flag that this is investigate, not error
        }

    def check_margin_outliers(self) -> Dict[str, Any]:
        """
        Find units with unusual margins - either very high or very low.
        These are not errors, but items to investigate for pricing review.
        """
        if self._inventory is None or self._inventory.empty:
            return {"status": "no_data", "issue_count": 0, "issues": []}

        issues = []
        skipped = 0

        retail_col = None
        cost_col = None

        for col in ["Retail Price", "Retail", "MSRP", "Price"]:
            if col in self._inventory.columns:
                retail_col = col
                break

        for col in ["Total Cost", "Cost", "Invoice"]:
            if col in self._inventory.columns:
                cost_col = col
                break

        if not retail_col or not cost_col:
            return {"status": "no_price_columns", "issue_count": 0, "issues": []}

        # Calculate margins for all units
        margins = []
        for idx, unit in self._inventory.iterrows():
            retail = unit.get(retail_col)
            cost = unit.get(cost_col)

            if pd.isna(retail) or pd.isna(cost) or retail <= 0 or cost <= 0:
                continue

            margin_pct = (retail - cost) / retail * 100
            margins.append({
                "idx": idx,
                "margin": margin_pct,
                "retail": retail,
                "cost": cost,
                "unit": unit
            })

        if len(margins) < 10:
            return {"status": "insufficient_data", "issue_count": 0, "issues": []}

        # Calculate statistics
        margin_values = [m["margin"] for m in margins]
        mean_margin = np.mean(margin_values)
        std_margin = np.std(margin_values)

        # Flag outliers (beyond 2 standard deviations)
        high_threshold = mean_margin + 2 * std_margin
        low_threshold = mean_margin - 2 * std_margin

        for m in margins:
            margin = m["margin"]
            unit = m["unit"]

            if margin > high_threshold:
                code = "OUT004"
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "code": code,
                    "veh_type": unit.get("Veh Type", ""),
                    "margin_pct": round(margin, 1),
                    "mean_margin": round(mean_margin, 1),
                    "retail_price": m["retail"],
                    "total_cost": m["cost"],
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "location": unit.get("PC", ""),
                    "issue": f"High margin {margin:.1f}% (avg: {mean_margin:.1f}%) - verify pricing"
                })
            elif margin < low_threshold and margin > 0:  # Low but positive
                code = "OUT005"
                issues.append({
                    "stock_num": unit.get("Stock#", ""),
                    "vin": unit.get("VIN", ""),
                    "code": code,
                    "veh_type": unit.get("Veh Type", ""),
                    "margin_pct": round(margin, 1),
                    "mean_margin": round(mean_margin, 1),
                    "retail_price": m["retail"],
                    "total_cost": m["cost"],
                    "make": unit.get("Make", ""),
                    "model": unit.get("Model", ""),
                    "location": unit.get("PC", ""),
                    "issue": f"Low margin {margin:.1f}% (avg: {mean_margin:.1f}%) - verify pricing"
                })

        # Sort by margin
        issues.sort(key=lambda x: abs(x.get("margin_pct", 0) - mean_margin), reverse=True)

        return {
            "status": "completed",
            "issue_count": len(issues),
            "skipped_approved": skipped,
            "issues": issues,
            "statistics": {
                "mean_margin": round(mean_margin, 2),
                "std_margin": round(std_margin, 2),
                "high_threshold": round(high_threshold, 2),
                "low_threshold": round(low_threshold, 2)
            },
            "description": "Margin outliers - unusually high or low margins (investigate)",
            "is_outlier_check": True
        }


class DataCleaner:
    """Auto-fix common data quality issues."""

    # Known typos and their corrections
    KNOWN_TYPOS = {
        "VIKING 5K SEREIS": "VIKING 5K SERIES",
        # Add more as discovered
    }

    # Fields to clean whitespace
    TEXT_FIELDS = ["Manufacturer", "Make", "Model", "Floorplan", "PC", "Status"]

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)
        self.output_path = self.config.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def preview_fixes(self) -> Dict[str, Any]:
        """
        Preview what fixes would be applied without changing anything.

        Returns:
            Dict with proposed fixes organized by type
        """
        inventory = self.data_loader.load_current_inventory()

        if inventory is None or inventory.empty:
            return {"status": "no_data", "fixes": []}

        fixes = {
            "whitespace_fixes": [],
            "typo_fixes": [],
            "total_records_affected": 0,
            "total_fixes": 0
        }

        affected_stocks = set()

        # Check each row for fixable issues
        for idx, row in inventory.iterrows():
            stock_num = row.get("Stock#", "")

            # Check text fields for whitespace
            for field in self.TEXT_FIELDS:
                if field not in inventory.columns:
                    continue

                value = row.get(field)
                if pd.isna(value):
                    continue

                value_str = str(value)
                cleaned = " ".join(value_str.split())  # Normalize whitespace

                if value_str != cleaned:
                    fixes["whitespace_fixes"].append({
                        "stock_num": stock_num,
                        "field": field,
                        "original": value_str,
                        "fixed": cleaned,
                        "fix_type": "whitespace"
                    })
                    affected_stocks.add(stock_num)
                    fixes["total_fixes"] += 1

            # Check for known typos
            for field in ["Manufacturer", "Make", "Model"]:
                if field not in inventory.columns:
                    continue

                value = row.get(field)
                if pd.isna(value):
                    continue

                value_upper = str(value).strip().upper()

                for typo, correction in self.KNOWN_TYPOS.items():
                    if value_upper == typo.upper():
                        fixes["typo_fixes"].append({
                            "stock_num": stock_num,
                            "field": field,
                            "original": str(value),
                            "fixed": correction,
                            "fix_type": "typo"
                        })
                        affected_stocks.add(stock_num)
                        fixes["total_fixes"] += 1

        fixes["total_records_affected"] = len(affected_stocks)

        return fixes

    def apply_fixes(self, export_cleaned: bool = True) -> Dict[str, Any]:
        """
        Apply fixes and optionally export cleaned data.

        Args:
            export_cleaned: If True, export a cleaned Excel file

        Returns:
            Dict with results and path to cleaned file
        """
        inventory = self.data_loader.load_current_inventory()

        if inventory is None or inventory.empty:
            return {"status": "no_data", "fixes_applied": 0}

        fixes_applied = []
        df = inventory.copy()

        # Apply whitespace fixes
        for field in self.TEXT_FIELDS:
            if field not in df.columns:
                continue

            # Create mask for non-null values
            mask = df[field].notna()

            # Get original values for comparison
            original = df.loc[mask, field].astype(str)

            # Clean whitespace
            cleaned = original.apply(lambda x: " ".join(x.split()))

            # Find changed values
            changed_mask = original != cleaned
            changed_indices = original[changed_mask].index

            for idx in changed_indices:
                fixes_applied.append({
                    "stock_num": df.loc[idx, "Stock#"] if "Stock#" in df.columns else idx,
                    "field": field,
                    "original": original[idx],
                    "fixed": cleaned[idx],
                    "fix_type": "whitespace"
                })

            # Apply the fix
            df.loc[mask, field] = cleaned

        # Apply typo fixes
        for field in ["Manufacturer", "Make", "Model"]:
            if field not in df.columns:
                continue

            for typo, correction in self.KNOWN_TYPOS.items():
                mask = df[field].astype(str).str.upper().str.strip() == typo.upper()
                changed_indices = df[mask].index

                for idx in changed_indices:
                    fixes_applied.append({
                        "stock_num": df.loc[idx, "Stock#"] if "Stock#" in df.columns else idx,
                        "field": field,
                        "original": df.loc[idx, field],
                        "fixed": correction,
                        "fix_type": "typo"
                    })

                df.loc[mask, field] = correction

        result = {
            "status": "success",
            "fixes_applied": len(fixes_applied),
            "fixes_detail": fixes_applied,
        }

        # Export cleaned data
        if export_cleaned and len(fixes_applied) > 0:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Inventory_Cleaned_{date_str}.xlsx"
            output_file = self.output_path / filename

            # Export with multiple sheets
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Fixes applied log
                fixes_df = pd.DataFrame(fixes_applied)
                fixes_df = fixes_df.rename(columns={
                    "stock_num": "Stock#",
                    "field": "Field",
                    "original": "Original Value",
                    "fixed": "Corrected Value",
                    "fix_type": "Fix Type"
                })
                fixes_df.to_excel(writer, sheet_name="Fixes Applied", index=False)

                # Cleaned inventory
                df.to_excel(writer, sheet_name="Cleaned Inventory", index=False)

            result["cleaned_file"] = str(output_file)

        return result

    def generate_fix_script(self) -> str:
        """
        Generate a SQL/update script for applying fixes to source system.

        Returns:
            Path to generated script file
        """
        fixes = self.preview_fixes()

        if fixes["total_fixes"] == 0:
            return None

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Data_Fixes_{date_str}.txt"
        output_file = self.output_path / filename

        lines = [
            "=" * 70,
            "FTRV DATA QUALITY FIXES",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Fixes: {fixes['total_fixes']}",
            f"Records Affected: {fixes['total_records_affected']}",
            "=" * 70,
            "",
            "WHITESPACE FIXES (trim/normalize spaces):",
            "-" * 50,
        ]

        for fix in fixes["whitespace_fixes"]:
            lines.append(f"Stock# {fix['stock_num']}: {fix['field']}")
            lines.append(f"  FROM: \"{fix['original']}\"")
            lines.append(f"  TO:   \"{fix['fixed']}\"")
            lines.append("")

        lines.append("")
        lines.append("TYPO FIXES:")
        lines.append("-" * 50)

        for fix in fixes["typo_fixes"]:
            lines.append(f"Stock# {fix['stock_num']}: {fix['field']}")
            lines.append(f"  FROM: \"{fix['original']}\"")
            lines.append(f"  TO:   \"{fix['fixed']}\"")
            lines.append("")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return str(output_file)


class DataQualityReportGenerator:
    """Generate Excel report from data quality analysis - clerk-friendly format."""

    # Priority levels and actions for each category
    CATEGORY_CONFIG = {
        "Licensing Violations": {
            "priority": "HIGH",
            "action": "Transfer unit or update licensing",
            "short_name": "Licensing"
        },
        "Pricing Issues": {
            "priority": "HIGH",
            "action": "Update pricing in system",
            "short_name": "Pricing"
        },
        "Duplicate Detection": {
            "priority": "HIGH",
            "action": "Investigate and remove duplicate",
            "short_name": "Duplicates"
        },
        "Missing Required Data": {
            "priority": "MEDIUM",
            "action": "Fill in missing field",
            "short_name": "Missing Data"
        },
        "Relationship Inconsistencies": {
            "priority": "MEDIUM",
            "action": "Verify and correct relationship",
            "short_name": "Relationships"
        },
        "Text Quality Issues": {
            "priority": "MEDIUM",
            "action": "Review for typo correction",
            "short_name": "Typos"
        },
        "Model Year Issues": {
            "priority": "MEDIUM",
            "action": "Verify and correct model year",
            "short_name": "Model Years"
        },
        "Status Anomalies": {
            "priority": "MEDIUM",
            "action": "Update status or investigate delay",
            "short_name": "Status"
        },
        "Whitespace Issues": {
            "priority": "LOW",
            "action": "Trim whitespace from field",
            "short_name": "Whitespace"
        },
    }

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.output_path = self.config.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_report(self, analysis_results: Dict[str, Any], include_dms_fixes: bool = True) -> str:
        """
        Generate clerk-friendly Excel report from analysis results.

        Returns:
            Path to generated Excel file
        """
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Data_Quality_Report_{date_str}.xlsx"
        output_file = self.output_path / filename

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary first
            self._write_summary_tab(writer, analysis_results)

            # Master action list - ALL issues in one place, sorted by priority
            self._write_action_list_tab(writer, analysis_results)

            # DMS Fixes tab - clear before/after for easy DMS updates
            if include_dms_fixes:
                self._write_dms_fixes_tab(writer, analysis_results)

            # Individual category tabs for deep-dive
            for category, findings in analysis_results.get("categories", {}).items():
                if findings.get("issue_count", 0) > 0:
                    short_name = self.CATEGORY_CONFIG.get(category, {}).get("short_name", category[:20])
                    self._write_category_tab(writer, short_name, category, findings)

        return str(output_file)

    def _write_dms_fixes_tab(self, writer: pd.ExcelWriter, results: Dict):
        """Write DMS-friendly fixes tab with clear before/after values."""
        rows = []

        categories = results.get("categories", {})

        # Whitespace fixes
        whitespace = categories.get("Whitespace Issues", {}).get("issues", [])
        for issue in whitespace:
            rows.append({
                "Fix Type": "WHITESPACE",
                "Stock#": issue.get("stock_num", ""),
                "Field to Update": issue.get("field", ""),
                "Current Value (WRONG)": issue.get("value", "").strip("'\""),
                "Correct Value (UPDATE TO)": issue.get("clean_value", ""),
                "Problem": issue.get("problems", ""),
                "Instructions": "Remove extra spaces",
                "Fixed in DMS?": ""
            })

        # Typo fixes - extract from Text Quality Issues
        text_quality = categories.get("Text Quality Issues", {}).get("issues", [])
        for issue in text_quality:
            # Look for the known typo
            val1 = issue.get("value_1", "")
            val2 = issue.get("value_2", "")

            # Check if either is a known typo
            known_typos = {
                "VIKING 5K SEREIS": "VIKING 5K SERIES",
            }

            if val1.upper() in known_typos:
                rows.append({
                    "Fix Type": "TYPO",
                    "Stock#": f"ALL with {issue.get('field', '')}='{val1}'",
                    "Field to Update": issue.get("field", ""),
                    "Current Value (WRONG)": val1,
                    "Correct Value (UPDATE TO)": known_typos[val1.upper()],
                    "Problem": "Spelling error",
                    "Instructions": f"Update all {issue.get('count_1', 0)} records",
                    "Fixed in DMS?": ""
                })
            elif val2.upper() in known_typos:
                rows.append({
                    "Fix Type": "TYPO",
                    "Stock#": f"ALL with {issue.get('field', '')}='{val2}'",
                    "Field to Update": issue.get("field", ""),
                    "Current Value (WRONG)": val2,
                    "Correct Value (UPDATE TO)": known_typos[val2.upper()],
                    "Problem": "Spelling error",
                    "Instructions": f"Update all {issue.get('count_2', 0)} records",
                    "Fixed in DMS?": ""
                })

        if not rows:
            df = pd.DataFrame([["No auto-fixable issues (whitespace/typos) found"]], columns=["Status"])
            df.to_excel(writer, sheet_name="DMS Fixes", index=False)
            return

        # Header rows for instructions
        header_rows = [
            ["DMS UPDATE INSTRUCTIONS", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["Use this tab to update records in your DMS.", "", "", "", "", "", "", ""],
            ["1. Find the Stock# in your DMS", "", "", "", "", "", "", ""],
            ["2. Go to the Field listed", "", "", "", "", "", "", ""],
            ["3. Change from 'Current Value' to 'Correct Value'", "", "", "", "", "", "", ""],
            ["4. Mark 'Fixed in DMS?' column when done", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
        ]

        header_df = pd.DataFrame(header_rows)
        header_df.to_excel(writer, sheet_name="DMS Fixes", index=False, header=False)

        df = pd.DataFrame(rows)
        col_order = [
            "Fix Type", "Stock#", "Field to Update",
            "Current Value (WRONG)", "Correct Value (UPDATE TO)",
            "Problem", "Instructions", "Fixed in DMS?"
        ]
        df = df[col_order]
        df.to_excel(writer, sheet_name="DMS Fixes", index=False, startrow=9)

    def _write_summary_tab(self, writer: pd.ExcelWriter, results: Dict):
        """Write executive summary tab."""
        rows = []
        rows.append(["FTRV DATA QUALITY REPORT", ""])
        rows.append(["For: Inventory Clerks", ""])
        rows.append(["Generated", results.get("generated_at", "")])
        rows.append(["", ""])
        rows.append(["OVERVIEW", ""])
        rows.append(["Total Units Analyzed", f"{results.get('total_units_analyzed', 0):,}"])
        rows.append(["Total Issues Found", f"{results.get('total_issues', 0):,}"])
        rows.append(["", ""])

        # Group by priority
        high_count = 0
        medium_count = 0
        low_count = 0

        rows.append(["ISSUES BY CATEGORY", "Count", "Priority", "Action Required"])
        for category, findings in results.get("categories", {}).items():
            count = findings.get("issue_count", 0)
            config = self.CATEGORY_CONFIG.get(category, {"priority": "LOW", "action": "Review"})
            if count > 0:
                rows.append([category, count, config["priority"], config["action"]])
                if config["priority"] == "HIGH":
                    high_count += count
                elif config["priority"] == "MEDIUM":
                    medium_count += count
                else:
                    low_count += count

        rows.append(["", "", "", ""])
        rows.append(["PRIORITY TOTALS", "", "", ""])
        rows.append(["HIGH Priority Issues", high_count, "", "Fix immediately"])
        rows.append(["MEDIUM Priority Issues", medium_count, "", "Fix this week"])
        rows.append(["LOW Priority Issues", low_count, "", "Fix when time permits"])
        rows.append(["", "", "", ""])
        rows.append(["INSTRUCTIONS", "", "", ""])
        rows.append(["1. Start with the 'Action List' tab - it has ALL issues sorted by priority", "", "", ""])
        rows.append(["2. Work through HIGH priority items first (red)", "", "", ""])
        rows.append(["3. Mark items complete as you fix them", "", "", ""])
        rows.append(["4. See individual tabs for more detail on each category", "", "", ""])

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Summary", index=False, header=False)

    def _write_action_list_tab(self, writer: pd.ExcelWriter, results: Dict):
        """Write master action list with ALL issues sorted by priority."""
        all_issues = []

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

        for category, findings in results.get("categories", {}).items():
            issues = findings.get("issues", findings.get("violations", []))
            config = self.CATEGORY_CONFIG.get(category, {"priority": "LOW", "action": "Review", "short_name": category})

            for issue in issues:
                # Skip summary rows
                if issue.get("stock_num") == "..." or issue.get("stock_num") == "N/A":
                    continue

                row = {
                    "Priority": config["priority"],
                    "_priority_sort": priority_order.get(config["priority"], 2),
                    "Stock#": issue.get("stock_num", ""),
                    "VIN": issue.get("vin", ""),
                    "Category": config["short_name"],
                    "Issue": issue.get("issue", issue.get("suggestion", "")),
                    "Action Required": config["action"],
                    "Location": issue.get("location", issue.get("PC", "")),
                    "Make": issue.get("make", ""),
                    "Model": issue.get("model", ""),
                    "Field": issue.get("field", ""),
                    "Current Value": issue.get("value", issue.get("clean_value", "")),
                    "Fixed?": "",  # Empty column for clerks to mark
                }
                all_issues.append(row)

        if not all_issues:
            df = pd.DataFrame([["No issues found - data quality looks good!"]], columns=["Status"])
            df.to_excel(writer, sheet_name="Action List", index=False)
            return

        df = pd.DataFrame(all_issues)

        # Sort by priority then stock number
        df = df.sort_values(["_priority_sort", "Stock#"])
        df = df.drop(columns=["_priority_sort"])

        # Reorder columns for clerk workflow
        col_order = [
            "Priority", "Stock#", "VIN", "Category", "Issue",
            "Action Required", "Fixed?", "Location", "Make", "Model", "Field", "Current Value"
        ]
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]

        df.to_excel(writer, sheet_name="Action List", index=False)

    def _write_category_tab(self, writer: pd.ExcelWriter, tab_name: str, category: str, findings: Dict):
        """Write a category findings tab with detailed info."""
        if findings.get("status") == "error":
            df = pd.DataFrame([["Error", findings.get("message", "Unknown error")]],
                            columns=["Status", "Message"])
            df.to_excel(writer, sheet_name=tab_name, index=False)
            return

        issues = findings.get("issues", findings.get("violations", []))

        if not issues:
            df = pd.DataFrame([["No issues found in this category"]], columns=["Status"])
            df.to_excel(writer, sheet_name=tab_name, index=False)
            return

        config = self.CATEGORY_CONFIG.get(category, {"priority": "LOW", "action": "Review"})

        # Header info
        header_rows = [
            [f"Category: {category}", ""],
            [f"Priority: {config['priority']}", ""],
            [f"Action: {config['action']}", ""],
            [f"Total Issues: {len(issues)}", ""],
            ["", ""],
        ]
        header_df = pd.DataFrame(header_rows, columns=["Info", ""])
        header_df.to_excel(writer, sheet_name=tab_name, index=False)

        # Convert issues to DataFrame
        df = pd.DataFrame(issues)

        # Add action column
        df["Action Required"] = config["action"]
        df["Fixed?"] = ""

        # Reorder columns to put key info first
        priority_cols = ["stock_num", "vin", "issue", "Action Required", "Fixed?",
                        "field", "type", "value", "location", "make", "model"]
        cols = [c for c in priority_cols if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        # Rename columns for clarity
        rename_map = {
            "stock_num": "Stock#",
            "vin": "VIN",
            "issue": "Issue Description",
            "field": "Field",
            "value": "Value",
            "location": "Location",
            "make": "Make",
            "model": "Model",
        }
        df = df.rename(columns=rename_map)

        df.to_excel(writer, sheet_name=tab_name, index=False, startrow=6)
