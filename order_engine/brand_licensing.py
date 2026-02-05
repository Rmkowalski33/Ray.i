"""
Brand Licensing Analyzer Module
Manages brand/make licensing status by location, mapping Brand Licensing file
to inventory data naming conventions.

Key mapping:
- Brand Licensing "Division" -> Inventory "Manufacturer" (e.g., KEYSTONE, COACHMEN)
- Brand Licensing "BRAND SERIES SUBSERIES" -> Inventory "Make" (e.g., ALPINE, APEX)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import Config, default_config
from .data_loader import DataLoader


class BrandLicensingAnalyzer:
    """
    Analyze brand licensing status across locations.

    Maps Brand Licensing file nomenclature to inventory nomenclature:
    - Division -> Manufacturer
    - BRAND SERIES SUBSERIES -> Make
    """

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)
        self._licensing_df: Optional[pd.DataFrame] = None
        self._location_codes: Optional[List[str]] = None

    def _load_licensing_data(self) -> pd.DataFrame:
        """Load and normalize brand licensing data."""
        if self._licensing_df is not None:
            return self._licensing_df

        file_path = self.config.data_hub_path / "Brand Licensing.xlsx"
        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()

        # Identify location columns (after the standard columns)
        standard_cols = [
            "Manufacturer", "Division", "BRAND SERIES SUBSERIES",
            "Type", "DIVISION BRAND SERIES SUBSERIES", "Column1", "Dealers"
        ]
        self._location_codes = [c for c in df.columns if c not in standard_cols]

        # Normalize naming to match inventory data
        # Division -> maps to Inventory Manufacturer
        # BRAND SERIES SUBSERIES -> maps to Inventory Make
        df = df.rename(columns={
            "Division": "Inv_Manufacturer",           # What inventory calls Manufacturer
            "BRAND SERIES SUBSERIES": "Inv_Make",     # What inventory calls Make
            "Manufacturer": "OEM_Parent",             # Parent company (THOR, FOREST RIVER, etc.)
            "Type": "Veh_Type"
        })

        # Forward fill OEM_Parent for grouped rows
        df["OEM_Parent"] = df["OEM_Parent"].ffill()

        # Clean up string columns
        for col in ["OEM_Parent", "Inv_Manufacturer", "Inv_Make", "Veh_Type"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(["NAN", "NONE", ""], np.nan)

        self._licensing_df = df
        return df

    def get_location_codes(self) -> List[str]:
        """Get list of location codes from licensing file."""
        self._load_licensing_data()
        return self._location_codes or []

    def get_licensing_status(
        self,
        manufacturer: str,
        make: str,
        location: str
    ) -> str:
        """
        Get licensing status for a specific manufacturer/make at a location.

        Args:
            manufacturer: Inventory manufacturer name (e.g., KEYSTONE)
            make: Inventory make name (e.g., ALPINE)
            location: Location code (e.g., AMA)

        Returns:
            Status string: "Licensed", "Not Available", "Pending", etc.
        """
        df = self._load_licensing_data()
        if df.empty:
            return "Unknown"

        manufacturer = manufacturer.upper().strip()
        make = make.upper().strip()
        location = location.upper().strip()

        if location not in df.columns:
            return "Unknown"

        # Find matching row
        mask = (
            (df["Inv_Manufacturer"].str.upper() == manufacturer) &
            (df["Inv_Make"].str.upper() == make)
        )

        matches = df[mask]
        if matches.empty:
            # Try matching just manufacturer (for makes not in licensing file)
            return "Not Listed"

        status = matches[location].iloc[0]
        if pd.isna(status):
            return "Not Listed"

        return str(status).strip()

    def get_licensing_matrix(
        self,
        manufacturer: str = None,
        veh_type: str = None
    ) -> pd.DataFrame:
        """
        Get licensing matrix showing status by make and location.

        Args:
            manufacturer: Filter by inventory manufacturer name
            veh_type: Filter by vehicle type (TT, FW, etc.)

        Returns:
            DataFrame with makes as rows and locations as columns
        """
        df = self._load_licensing_data()
        if df.empty:
            return pd.DataFrame()

        # Apply filters
        mask = pd.Series(True, index=df.index)
        if manufacturer:
            mask &= df["Inv_Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= df["Veh_Type"].str.upper() == veh_type.upper()

        filtered = df[mask].copy()
        if filtered.empty:
            return pd.DataFrame()

        # Select columns for matrix
        info_cols = ["OEM_Parent", "Inv_Manufacturer", "Inv_Make", "Veh_Type"]
        location_cols = self.get_location_codes()

        available_cols = [c for c in info_cols + location_cols if c in filtered.columns]
        result = filtered[available_cols].copy()

        # Rename for clarity
        result = result.rename(columns={
            "Inv_Manufacturer": "Manufacturer",
            "Inv_Make": "Make",
            "Veh_Type": "Type",
            "OEM_Parent": "OEM"
        })

        return result

    def check_recommendations_licensing(
        self,
        recommendations: List[Dict],
        locations: List[str] = None
    ) -> List[Dict]:
        """
        Check licensing status for a list of recommendations.

        Adds licensing information to each recommendation.

        Args:
            recommendations: List of recommendation dicts with 'manufacturer' and 'make'
            locations: Optional list of locations to check (default: all)

        Returns:
            Recommendations with added licensing fields
        """
        df = self._load_licensing_data()
        if df.empty:
            return recommendations

        location_codes = locations or self.get_location_codes()

        for rec in recommendations:
            manufacturer = rec.get("manufacturer", "")
            make = rec.get("make", "")

            # Get licensing status at each location
            licensing_status = {}
            licensed_count = 0
            not_available_count = 0
            pending_count = 0

            for loc in location_codes:
                status = self.get_licensing_status(manufacturer, make, loc)
                licensing_status[loc] = status

                if "LICENSED" in status.upper():
                    licensed_count += 1
                elif "NOT AVAILABLE" in status.upper():
                    not_available_count += 1
                elif "PENDING" in status.upper():
                    pending_count += 1

            # Add summary to recommendation
            rec["licensing_summary"] = {
                "licensed_locations": licensed_count,
                "not_available_locations": not_available_count,
                "pending_locations": pending_count,
                "total_locations_checked": len(location_codes)
            }
            rec["licensing_detail"] = licensing_status

            # Add warning flag if not widely licensed
            if licensed_count < len(location_codes) * 0.5:
                rec["licensing_warning"] = f"Only licensed at {licensed_count}/{len(location_codes)} locations"
            else:
                rec["licensing_warning"] = None

        return recommendations

    def get_licensed_makes_for_location(self, location: str, manufacturer: str = None) -> List[str]:
        """
        Get list of makes licensed at a specific location.

        Args:
            location: Location code
            manufacturer: Optional filter by manufacturer

        Returns:
            List of make names that are licensed
        """
        df = self._load_licensing_data()
        if df.empty or location not in df.columns:
            return []

        mask = df[location].str.upper().str.contains("LICENSED", na=False)
        if manufacturer:
            mask &= df["Inv_Manufacturer"].str.upper() == manufacturer.upper()

        return df.loc[mask, "Inv_Make"].dropna().unique().tolist()

    def get_location_licensing_summary(self, location: str) -> Dict:
        """
        Get licensing summary for a specific location.

        Returns dict with counts by status.
        """
        df = self._load_licensing_data()
        if df.empty or location not in df.columns:
            return {}

        status_counts = df[location].value_counts().to_dict()
        return {
            "location": location,
            "licensed": sum(1 for s in df[location] if pd.notna(s) and "LICENSED" in str(s).upper()),
            "not_available": sum(1 for s in df[location] if pd.notna(s) and "NOT AVAILABLE" in str(s).upper()),
            "pending": sum(1 for s in df[location] if pd.notna(s) and "PENDING" in str(s).upper()),
            "status_breakdown": status_counts
        }

    def generate_licensing_report(self, manufacturer: str = None) -> Dict:
        """
        Generate comprehensive licensing report.

        Returns dict with:
        - Summary by location
        - Makes with limited availability
        - Location coverage gaps
        """
        df = self._load_licensing_data()
        if df.empty:
            return {"error": "No licensing data available"}

        if manufacturer:
            df = df[df["Inv_Manufacturer"].str.upper() == manufacturer.upper()]

        location_codes = self.get_location_codes()

        report = {
            "manufacturer": manufacturer or "All",
            "total_makes": len(df["Inv_Make"].dropna().unique()),
            "locations_analyzed": len(location_codes),
            "by_location": {},
            "limited_availability_makes": [],
            "fully_licensed_makes": []
        }

        # Analyze by location
        for loc in location_codes:
            report["by_location"][loc] = self.get_location_licensing_summary(loc)

        # Find makes with limited availability
        for _, row in df.iterrows():
            make = row.get("Inv_Make", "")
            if pd.isna(make):
                continue

            licensed_count = sum(
                1 for loc in location_codes
                if pd.notna(row.get(loc)) and "LICENSED" in str(row[loc]).upper()
            )

            if licensed_count < len(location_codes) * 0.5:
                report["limited_availability_makes"].append({
                    "make": make,
                    "manufacturer": row.get("Inv_Manufacturer", ""),
                    "licensed_locations": licensed_count,
                    "total_locations": len(location_codes)
                })
            elif licensed_count == len(location_codes):
                report["fully_licensed_makes"].append(make)

        return report
