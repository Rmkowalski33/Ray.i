"""
Data Loader Module
Loads and normalizes all data sources from the Data Hub into analysis-ready DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from .config import Config, default_config, get_price_group


class DataLoader:
    """Load and normalize all FTRV data sources from Data Hub."""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self._cache: Dict[str, pd.DataFrame] = {}

    def _get_path(self, relative_path: str) -> Path:
        """Get full path for a data file."""
        return self.config.data_hub_path / relative_path

    # =========================================================================
    # INVENTORY DATA
    # =========================================================================

    def load_current_inventory(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load current inventory from Inventory_Extract.xlsx.
        Returns DataFrame with normalized columns and calculated fields.
        """
        cache_key = "current_inventory"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.inventory_file)
        df = pd.read_excel(file_path, sheet_name=self.config.inventory_sheet)

        df = self._normalize_inventory_columns(df)
        df = self._add_calculated_fields(df)
        df = self._classify_inventory_status(df)
        df = self._apply_filters(df)

        self._cache[cache_key] = df
        return df

    def load_retail_history(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load retail sales history from Inventory_Extract.xlsx.
        Returns DataFrame with sales data for demand analysis.
        """
        cache_key = "retail_history"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.inventory_file)
        df = pd.read_excel(file_path, sheet_name=self.config.retail_sheet)

        df = self._normalize_inventory_columns(df)
        df = self._add_calculated_fields(df)

        # Parse sold date for time series analysis
        if "Sold Date" in df.columns:
            df["Sold Date"] = pd.to_datetime(df["Sold Date"], errors="coerce")
            df["Sold Month"] = df["Sold Date"].dt.to_period("M")
            df["Sold Year"] = df["Sold Date"].dt.year

        df = self._apply_filters(df)

        self._cache[cache_key] = df
        return df

    def load_on_order(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load units on order from Inventory_Extract.xlsx.
        Returns DataFrame with pipeline/on-order units.
        """
        cache_key = "on_order"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.inventory_file)

        try:
            df = pd.read_excel(file_path, sheet_name=self.config.on_order_sheet)
        except ValueError:
            # Sheet doesn't exist, return empty DataFrame
            self._cache[cache_key] = pd.DataFrame()
            return self._cache[cache_key]

        df = self._normalize_inventory_columns(df)
        df = self._add_calculated_fields(df)
        df = self._apply_filters(df)

        self._cache[cache_key] = df
        return df

    def _normalize_inventory_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types."""
        df.columns = df.columns.str.strip()

        # Convert numeric columns
        numeric_cols = [
            "Retail Price", "Purchase Price", "Total Cost",
            "Book Value Low", "Book Value High", "FP Balance",
            "Age", "Status Days", "PC Age", "Deal FrontEnd",
            "Deal BackEnd", "Deal Selling Price", "Deal Cost"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Clean string columns
        string_cols = [
            "Manufacturer", "Make", "Model", "Veh Type",
            "Status", "Floor layout", "Sub Floorplan", "PC"
        ]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace(["NAN", "NONE", ""], np.nan)

        return df

    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields for analysis."""
        # Age bucket classification
        if "Age" in df.columns:
            df["Age Bucket"] = df["Age"].apply(self.config.get_age_bucket)

        # Price group classification
        if "Retail Price" in df.columns:
            df["Price Group"] = df["Retail Price"].apply(get_price_group)

        # Margin calculation
        if "Retail Price" in df.columns and "Total Cost" in df.columns:
            df["Gross Margin"] = df["Retail Price"] - df["Total Cost"]
            df["Margin Pct"] = np.where(
                df["Retail Price"] > 0,
                df["Gross Margin"] / df["Retail Price"],
                0
            )

        # Product key for joining
        df["Product Key"] = (
            df.get("Manufacturer", pd.Series([""])).fillna("").astype(str) + "|" +
            df.get("Make", pd.Series([""])).fillna("").astype(str) + "|" +
            df.get("Model", pd.Series([""])).fillna("").astype(str)
        )

        return df

    def _classify_inventory_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify inventory into status categories."""
        if "Status" not in df.columns:
            return df

        df["Status Category"] = "Unknown"

        sellable = df["Status"].isin([s.upper() for s in self.config.sellable_statuses])
        constrained = df["Status"].isin([s.upper() for s in self.config.constrained_statuses])
        dead = df["Status"].isin([s.upper() for s in self.config.dead_statuses])

        df.loc[sellable, "Status Category"] = "Sellable"
        df.loc[constrained, "Status Category"] = "Constrained"
        df.loc[dead, "Status Category"] = "Dead/Exit"

        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply configured filters to dataframe."""
        if self.config.filter_manufacturer and "Manufacturer" in df.columns:
            df = df[df["Manufacturer"].str.upper() == self.config.filter_manufacturer.upper()]

        if self.config.filter_make and "Make" in df.columns:
            df = df[df["Make"].str.upper() == self.config.filter_make.upper()]

        if self.config.filter_veh_types and "Veh Type" in df.columns:
            veh_types_upper = [v.upper() for v in self.config.filter_veh_types]
            df = df[df["Veh Type"].isin(veh_types_upper)]

        if self.config.filter_locations and "PC" in df.columns:
            locations_upper = [loc.upper() for loc in self.config.filter_locations]
            df = df[df["PC"].isin(locations_upper)]

        if self.config.filter_zones:
            # Need to join with locations to filter by zone
            locations = self.load_locations(use_cache=True)
            zone_pcs = locations[
                locations["Zone"].isin(self.config.filter_zones)
            ]["PC"].unique()
            if "PC" in df.columns:
                df = df[df["PC"].isin(zone_pcs)]

        return df

    # =========================================================================
    # REFERENCE DATA
    # =========================================================================

    def load_locations(self, use_cache: bool = True) -> pd.DataFrame:
        """Load location/store reference data with zone information."""
        cache_key = "locations"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.references_file)
        df = pd.read_excel(file_path, sheet_name=self.config.locations_sheet, header=2)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Rename columns to standard names
        column_map = {
            "LOCATION": "Location ID",
            "PROFIT CENTER": "PC",
            "ZONE": "Zone",
            "ZONE GM": "Zone GM",
            "LOCATION NAME": "Location Name",
            "CITY": "City",
            "STATE": "State",
            "Dealer (Stat Surveys)": "SSI Dealer",
            "Dealer BTA (Stat Surveys)": "SSI BTA"
        }

        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Ensure NEW capacity column is numeric (lot capacity for new inventory)
        if "NEW" in df.columns:
            df["NEW"] = pd.to_numeric(df["NEW"], errors="coerce").fillna(0).astype(int)

        self._cache[cache_key] = df
        return df

    def load_product_master(self, use_cache: bool = True) -> pd.DataFrame:
        """Load product (manufacturer/make) reference data."""
        cache_key = "product_master"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.references_file)
        df = pd.read_excel(file_path, sheet_name=self.config.product_sheet)

        for col in ["Manufacturer", "Make", "OEM"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        self._cache[cache_key] = df
        return df

    def load_status_mappings(self, use_cache: bool = True) -> pd.DataFrame:
        """Load status normalization mappings."""
        cache_key = "status_mappings"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.references_file)
        df = pd.read_excel(file_path, sheet_name=self.config.status_sheet)

        self._cache[cache_key] = df
        return df

    # =========================================================================
    # SSI MARKET DATA
    # =========================================================================

    def load_ssi_market_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load SSI market data for competitive intelligence.
        This is the main market sales data.
        """
        cache_key = "ssi_market"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.ssi_data_file)

        # Try different encodings
        for encoding in ["utf-8", "latin1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not read SSI data with any encoding")

        # Skip duplicate header row if present (row 0 often has column names as values)
        if len(df) > 0 and df.iloc[0]["Manufacturer"] == "Manufacturer":
            df = df.iloc[1:].reset_index(drop=True)

        # The units column might be named after the year (e.g., "2025")
        year_cols = [col for col in df.columns if str(col).isdigit() and len(str(col)) == 4]
        if year_cols:
            # Use the most recent year column
            df["Units"] = pd.to_numeric(df[year_cols[-1]], errors="coerce").fillna(0).astype(int)
        elif "Units" in df.columns:
            df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0).astype(int)

        # Clean string columns
        string_cols = ["Manufacturer", "Division", "Model", "Type", "Sub Type",
                       "Price Group", "Dealer", "County", "Year-Month"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        # Extract BTA from Dealer name (format: "Dealer Name-City, ST" -> use locations mapping)
        # We'll add BTA in a separate step using locations data

        self._cache[cache_key] = df
        return df

    def load_ssi_dealers(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load SSI dealer reference data with BTA mappings.

        Returns DataFrame with columns:
        - Dealer: Dealer name (matches SSI Full Data)
        - DealerBTA: Basic Trading Area
        - DealerCity, County, etc.
        """
        cache_key = "ssi_dealers"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.ssi_fact_tables_file)

        # Read with header on row 1 (0-indexed), data starts row 2
        df = pd.read_excel(file_path, sheet_name="Dealers", header=0)

        # The first row is actually the header, second row has column names
        # Check if first data row looks like headers
        if len(df) > 0 and str(df.iloc[0, 0]).strip() == "Dealer":
            # Use first row as headers
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Clean string columns
        for col in ["Dealer", "DealerBTA", "DealerCity", "County"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        self._cache[cache_key] = df
        return df

    def get_dealers_in_bta(self, bta: str) -> List[str]:
        """Get list of all dealers in a specific BTA."""
        dealers = self.load_ssi_dealers()
        if "DealerBTA" not in dealers.columns or "Dealer" not in dealers.columns:
            return []

        bta_upper = bta.upper().strip()
        filtered = dealers[dealers["DealerBTA"].str.upper().str.strip() == bta_upper]
        return filtered["Dealer"].tolist()

    def get_bta_for_dealer(self, dealer: str) -> str:
        """Get the BTA for a specific dealer."""
        dealers = self.load_ssi_dealers()
        if "DealerBTA" not in dealers.columns or "Dealer" not in dealers.columns:
            return ""

        dealer_upper = dealer.upper().strip()
        match = dealers[dealers["Dealer"].str.upper().str.strip() == dealer_upper]
        if len(match) > 0:
            return match["DealerBTA"].iloc[0]
        return ""

    def load_ssi_btas(self, use_cache: bool = True) -> pd.DataFrame:
        """Load FTRV BTA (Basic Trading Area) mappings."""
        cache_key = "ssi_btas"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path(self.config.ssi_fact_tables_file)
        df = pd.read_excel(file_path, sheet_name="FTRV BTAs")

        self._cache[cache_key] = df
        return df

    # =========================================================================
    # BRAND LICENSING DATA
    # =========================================================================

    def load_brand_licensing(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load brand licensing matrix from Brand Licensing.xlsx.
        Returns DataFrame with licensing status by brand and location.

        Columns typically include:
        - Brand/Make name
        - Status per location (Licensed, Not Available, Pending)
        """
        cache_key = "brand_licensing"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        file_path = self._get_path("Brand Licensing.xlsx")

        if not file_path.exists():
            # Return empty DataFrame if file doesn't exist
            self._cache[cache_key] = pd.DataFrame()
            return self._cache[cache_key]

        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            self._cache[cache_key] = df
            return df
        except Exception:
            self._cache[cache_key] = pd.DataFrame()
            return self._cache[cache_key]

    def get_licensed_brands_for_location(self, location_pc: str) -> List[str]:
        """Get list of brands a location is licensed to sell."""
        licensing = self.load_brand_licensing()
        if licensing.empty:
            return []

        location_pc = location_pc.upper().strip()
        if location_pc not in licensing.columns:
            return []

        # Find rows where this location has "Licensed" or similar status
        licensed_mask = licensing[location_pc].str.upper().str.contains(
            "LICENSED|YES|ACTIVE", na=False, regex=True
        )
        if "Brand" in licensing.columns:
            return licensing.loc[licensed_mask, "Brand"].tolist()
        elif "Make" in licensing.columns:
            return licensing.loc[licensed_mask, "Make"].tolist()
        return []

    def get_licensing_status_by_brand(self, brand: str) -> Dict[str, str]:
        """Get licensing status for a brand across all locations."""
        licensing = self.load_brand_licensing()
        if licensing.empty:
            return {}

        brand = brand.upper().strip()
        brand_col = "Brand" if "Brand" in licensing.columns else "Make"

        if brand_col not in licensing.columns:
            return {}

        brand_row = licensing[licensing[brand_col].str.upper() == brand]
        if brand_row.empty:
            return {}

        # Get all location columns (exclude brand column)
        location_cols = [c for c in licensing.columns if c != brand_col]
        return {loc: brand_row[loc].iloc[0] for loc in location_cols if pd.notna(brand_row[loc].iloc[0])}

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_manufacturers(self) -> List[str]:
        """Get list of manufacturers from retail history."""
        retail = self.load_retail_history()
        if "Manufacturer" in retail.columns:
            return sorted(retail["Manufacturer"].dropna().unique().tolist())
        return []

    def get_makes_for_manufacturer(self, manufacturer: str) -> List[str]:
        """Get list of makes for a manufacturer."""
        retail = self.load_retail_history()
        filtered = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]
        if "Make" in filtered.columns:
            return sorted(filtered["Make"].dropna().unique().tolist())
        return []

    def get_zones(self) -> List[str]:
        """Get list of zones from locations."""
        locations = self.load_locations()
        if "Zone" in locations.columns:
            return sorted(locations["Zone"].dropna().unique().tolist())
        return []

    def get_locations_in_zone(self, zone: str) -> List[str]:
        """Get list of location PCs in a zone."""
        locations = self.load_locations()
        filtered = locations[locations["Zone"] == zone]
        if "PC" in filtered.columns:
            return filtered["PC"].tolist()
        return []

    def get_veh_types(self) -> List[str]:
        """Get list of vehicle types from inventory."""
        inventory = self.load_current_inventory()
        if "Veh Type" in inventory.columns:
            return sorted(inventory["Veh Type"].dropna().unique().tolist())
        return []

    def get_manufacturer_inventory(self, manufacturer: str) -> pd.DataFrame:
        """Get current inventory filtered by manufacturer."""
        inventory = self.load_current_inventory()
        return inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]

    def get_manufacturer_sales(self, manufacturer: str) -> pd.DataFrame:
        """Get sales history filtered by manufacturer."""
        retail = self.load_retail_history()
        return retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def refresh_data(self):
        """Clear cache and reload all data."""
        self.clear_cache()
        self.load_current_inventory()
        self.load_retail_history()
        self.load_locations()
