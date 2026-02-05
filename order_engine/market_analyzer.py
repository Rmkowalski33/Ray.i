"""
Market Analyzer Module
Analyzes market demand using SSI data and internal sales history.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .config import Config, default_config
from .data_loader import DataLoader


class MarketAnalyzer:
    """Analyze market demand and competitive landscape."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)

    # =========================================================================
    # DEMAND VELOCITY ANALYSIS
    # =========================================================================

    def calculate_demand_velocity(
        self,
        manufacturer: str = None,
        make: str = None,
        veh_type: str = None,
        floorplan: str = None,
        price_group: str = None,
        lookback_months: int = None
    ) -> Dict:
        """
        Calculate sales velocity (units per month) for a segment.

        Returns dict with:
        - monthly_velocity: Average units sold per month
        - total_units: Total units in period
        - months_with_sales: Number of months with at least 1 sale
        - trend: "increasing", "decreasing", or "stable"
        """
        lookback = lookback_months or self.config.lookback_months
        retail = self.data_loader.load_retail_history()

        # Filter by segment criteria
        mask = pd.Series(True, index=retail.index)

        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if make:
            mask &= retail["Make"].str.upper() == make.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()
        if floorplan:
            mask &= retail["Sub Floorplan"].str.upper() == floorplan.upper()
        if price_group:
            mask &= retail["Price Group"] == price_group

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=lookback * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return {
                "monthly_velocity": 0,
                "total_units": 0,
                "months_with_sales": 0,
                "trend": "no_data",
                "sample_size": 0
            }

        # Calculate monthly sales
        monthly_sales = filtered.groupby("Sold Month").size()
        total_units = len(filtered)
        months_with_sales = len(monthly_sales)
        monthly_velocity = total_units / max(lookback, 1)

        # Calculate trend (compare first half to second half)
        trend = self._calculate_trend(monthly_sales)

        return {
            "monthly_velocity": round(monthly_velocity, 2),
            "total_units": total_units,
            "months_with_sales": months_with_sales,
            "trend": trend,
            "sample_size": len(filtered)
        }

    def _calculate_trend(self, monthly_sales: pd.Series) -> str:
        """Calculate sales trend from monthly data."""
        if len(monthly_sales) < 4:
            return "insufficient_data"

        sorted_months = monthly_sales.sort_index()
        mid = len(sorted_months) // 2
        first_half_avg = sorted_months.iloc[:mid].mean()
        second_half_avg = sorted_months.iloc[mid:].mean()

        if second_half_avg > first_half_avg * 1.1:
            return "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def get_velocity_by_make(
        self,
        manufacturer: str = None,
        veh_type: str = None,
        make: str = None,
        lookback_months: int = None
    ) -> pd.DataFrame:
        """
        Get sales velocity for each make (brand/series).
        Returns DataFrame with make-level velocity stats.

        This is the primary aggregation level for order planning.
        """
        lookback = lookback_months or self.config.lookback_months
        retail = self.data_loader.load_retail_history()

        # Start with all data
        mask = pd.Series(True, index=retail.index)

        # Filter by manufacturer if specified
        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()

        # Filter by make if specified
        if make:
            mask &= retail["Make"].str.upper() == make.upper()

        # Filter by vehicle type
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        # Lookback filter
        cutoff_date = datetime.now() - timedelta(days=lookback * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        # Group by make (the key change - aggregate at make level, not model)
        group_cols = ["Manufacturer", "Make"]
        if "Veh Type" in filtered.columns:
            group_cols.append("Veh Type")

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = "mean"
        if "Deal FrontEnd" in filtered.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"
        if "Total Cost" in filtered.columns:
            agg_dict["Total Cost"] = "mean"

        make_stats = (
            filtered.groupby(group_cols)
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten column names
        make_stats.columns = self._flatten_columns(make_stats.columns, group_cols)

        # Calculate monthly velocity
        make_stats["Monthly Velocity"] = make_stats["Units Sold"] / lookback

        # Sort by volume
        make_stats = make_stats.sort_values("Units Sold", ascending=False)

        return make_stats

    def get_velocity_by_model(
        self,
        manufacturer: str,
        veh_type: str = None,
        lookback_months: int = None
    ) -> pd.DataFrame:
        """
        Get sales velocity for each model from a manufacturer.
        Returns DataFrame with model-level velocity stats.

        Note: For order planning, use get_velocity_by_make() instead.
        Model-level is too granular for ordering decisions.
        """
        lookback = lookback_months or self.config.lookback_months
        retail = self.data_loader.load_retail_history()

        # Filter
        mask = retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        # Lookback filter
        cutoff_date = datetime.now() - timedelta(days=lookback * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        # Group by model
        group_cols = ["Make", "Model"]
        if "Sub Floorplan" in filtered.columns:
            group_cols.append("Sub Floorplan")
        if "Veh Type" in filtered.columns:
            group_cols.append("Veh Type")

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = "mean"
        if "Deal FrontEnd" in filtered.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"

        model_stats = (
            filtered.groupby(group_cols)
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten column names
        model_stats.columns = self._flatten_columns(model_stats.columns, group_cols)

        # Calculate monthly velocity
        model_stats["Monthly Velocity"] = model_stats["Units Sold"] / lookback

        # Sort by volume
        model_stats = model_stats.sort_values("Units Sold", ascending=False)

        return model_stats

    def _flatten_columns(self, columns, group_cols):
        """Flatten multi-level column names."""
        flat = []
        for col in columns:
            if isinstance(col, tuple):
                if col[1] == "":
                    flat.append(col[0])
                elif col[0] == "Stock#":
                    flat.append("Units Sold")
                elif col[0] == "Deal FrontEnd" and col[1] == "mean":
                    flat.append("Avg Front-End")
                elif col[0] == "Deal FrontEnd" and col[1] == "sum":
                    flat.append("Total Front-End")
                elif col[0] == "Retail Price":
                    flat.append("Avg Price")
                elif col[0] == "Age":
                    flat.append("Avg Days to Sell")
                else:
                    flat.append(f"{col[0]}_{col[1]}")
            else:
                flat.append(col)
        return flat

    # =========================================================================
    # SEASONALITY ANALYSIS
    # =========================================================================

    def analyze_seasonality(
        self,
        manufacturer: str = None,
        veh_type: str = None
    ) -> pd.DataFrame:
        """
        Analyze monthly seasonality patterns from historical sales.
        Returns DataFrame with monthly index (100 = average).
        """
        retail = self.data_loader.load_retail_history()

        # Filter
        mask = pd.Series(True, index=retail.index)
        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        mask &= retail["Sold Date"].notna()
        filtered = retail[mask].copy()

        if len(filtered) == 0:
            return pd.DataFrame()

        filtered["Month"] = filtered["Sold Date"].dt.month

        # Calculate monthly totals
        monthly_counts = filtered.groupby("Month").size()
        overall_avg = monthly_counts.mean()

        # Create seasonality index
        seasonality = pd.DataFrame({
            "Month": range(1, 13),
            "Month Name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        })

        seasonality["Actual Sales"] = seasonality["Month"].map(monthly_counts).fillna(0)
        seasonality["Index"] = (seasonality["Actual Sales"] / overall_avg * 100).round(0)
        seasonality["Config Index"] = seasonality["Month"].map(self.config.seasonality_index)

        return seasonality

    def get_seasonality_adjusted_demand(
        self,
        base_monthly_demand: float,
        target_month: int
    ) -> float:
        """Adjust monthly demand for seasonality."""
        factor = self.config.get_seasonality_factor(target_month)
        return base_monthly_demand * factor

    # =========================================================================
    # MARKET SHARE ANALYSIS (SSI DATA)
    # =========================================================================

    def get_ftrv_btas(self) -> List[str]:
        """
        Get list of FTRV's BTAs (Basic Trading Areas) from locations data.
        These are the markets where FTRV competes.
        """
        try:
            locations = self.data_loader.load_locations()
            if "SSI BTA" in locations.columns:
                return locations["SSI BTA"].dropna().unique().tolist()
            return []
        except Exception:
            return []

    def get_ftrv_ssi_dealers(self) -> List[str]:
        """
        Get list of FTRV's SSI dealer names from locations data.
        Used to identify FTRV sales in SSI data.
        """
        try:
            locations = self.data_loader.load_locations()
            if "SSI Dealer" in locations.columns:
                return locations["SSI Dealer"].dropna().unique().tolist()
            return []
        except Exception:
            return []

    def get_dealers_in_ftrv_btas(self) -> List[str]:
        """
        Get all dealers that operate in FTRV's BTAs.
        Uses SSI Dealers fact table to map dealers to BTAs.
        """
        ftrv_btas = self.get_ftrv_btas()
        if not ftrv_btas:
            return []

        try:
            ssi_dealers = self.data_loader.load_ssi_dealers()
        except Exception:
            return []

        if "DealerBTA" not in ssi_dealers.columns or "Dealer" not in ssi_dealers.columns:
            return []

        # Find all dealers in FTRV's BTAs
        ftrv_btas_upper = [str(b).upper().strip() for b in ftrv_btas if pd.notna(b)]
        dealers_in_btas = ssi_dealers[
            ssi_dealers["DealerBTA"].str.upper().str.strip().isin(ftrv_btas_upper)
        ]["Dealer"].tolist()

        return dealers_in_btas

    def calculate_market_share(
        self,
        manufacturer: str,
        bta: str = None,
        veh_type: str = None,
        use_ftrv_btas: bool = True
    ) -> Dict:
        """
        Calculate FTRV market share vs. total market using SSI data.

        Uses SSI Dealers fact table to properly filter by BTA territory.

        Args:
            manufacturer: Filter by manufacturer name
            bta: Optional specific BTA to analyze
            veh_type: Optional vehicle type filter
            use_ftrv_btas: If True, only count market in FTRV's BTAs

        Returns:
            Dict with market share metrics
        """
        try:
            ssi = self.data_loader.load_ssi_market_data()
            retail = self.data_loader.load_retail_history()
            ssi_dealers = self.data_loader.load_ssi_dealers()
        except Exception as e:
            return {"error": f"Could not load market data: {e}"}

        if ssi.empty:
            return {"error": "SSI data is empty"}

        # Get FTRV's BTAs and dealers
        ftrv_btas = self.get_ftrv_btas()
        ftrv_dealers = self.get_ftrv_ssi_dealers()

        # === BUILD DEALER-TO-BTA MAPPING ===
        dealer_bta_map = {}
        if "Dealer" in ssi_dealers.columns and "DealerBTA" in ssi_dealers.columns:
            for _, row in ssi_dealers.iterrows():
                dealer = str(row["Dealer"]).strip()
                bta_val = str(row["DealerBTA"]).strip()
                dealer_bta_map[dealer.upper()] = bta_val

        # === SSI MARKET TOTAL ===
        ssi_mask = pd.Series(True, index=ssi.index)

        # Filter by manufacturer if specified
        if manufacturer and "Manufacturer" in ssi.columns:
            ssi_mask &= ssi["Manufacturer"].str.upper() == manufacturer.upper()

        # Filter by vehicle type
        if veh_type and "Type" in ssi.columns:
            ssi_mask &= ssi["Type"].str.upper().str.contains(veh_type.upper(), na=False)

        # Filter to FTRV's BTAs using dealer-to-BTA mapping
        if use_ftrv_btas and ftrv_btas and "Dealer" in ssi.columns:
            ftrv_btas_upper = set(str(b).upper().strip() for b in ftrv_btas if pd.notna(b))

            # Map each SSI record's dealer to its BTA
            def dealer_in_ftrv_bta(dealer_name):
                if pd.isna(dealer_name):
                    return False
                dealer_upper = str(dealer_name).strip().upper()
                dealer_bta = dealer_bta_map.get(dealer_upper, "")
                return dealer_bta.upper().strip() in ftrv_btas_upper

            bta_mask = ssi["Dealer"].apply(dealer_in_ftrv_bta)
            ssi_mask &= bta_mask

        # Specific BTA override
        if bta and "Dealer" in ssi.columns:
            bta_upper = bta.upper().strip()

            def dealer_in_specific_bta(dealer_name):
                if pd.isna(dealer_name):
                    return False
                dealer_upper = str(dealer_name).strip().upper()
                dealer_bta = dealer_bta_map.get(dealer_upper, "")
                return bta_upper in dealer_bta.upper()

            ssi_mask &= ssi["Dealer"].apply(dealer_in_specific_bta)

        ssi_filtered = ssi[ssi_mask]
        total_market = int(ssi_filtered["Units"].sum()) if "Units" in ssi_filtered.columns else 0

        # === FTRV SALES FROM SSI ===
        ftrv_ssi_sales = 0
        if ftrv_dealers and "Dealer" in ssi_filtered.columns:
            ftrv_dealers_upper = set(str(d).upper().strip() for d in ftrv_dealers if pd.notna(d))

            def is_ftrv_dealer(dealer_name):
                if pd.isna(dealer_name):
                    return False
                dealer_upper = str(dealer_name).strip().upper()
                # Check for exact match or partial match (Fun Town RV-)
                for fd in ftrv_dealers_upper:
                    if fd in dealer_upper or dealer_upper in fd:
                        return True
                return False

            ftrv_dealer_mask = ssi_filtered["Dealer"].apply(is_ftrv_dealer)
            ftrv_ssi_sales = int(ssi_filtered.loc[ftrv_dealer_mask, "Units"].sum()) if "Units" in ssi_filtered.columns else 0

        # Internal sales count as backup
        ftrv_mask = pd.Series(True, index=retail.index)
        if manufacturer:
            ftrv_mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type and "Veh Type" in retail.columns:
            ftrv_mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        cutoff_date = datetime.now() - timedelta(days=365)
        if "Sold Date" in retail.columns:
            ftrv_mask &= retail["Sold Date"] >= cutoff_date

        ftrv_internal_sales = len(retail[ftrv_mask])

        # Use SSI-matched sales if available, otherwise internal
        ftrv_sales = ftrv_ssi_sales if ftrv_ssi_sales > 0 else ftrv_internal_sales

        # Calculate share
        market_share = ftrv_sales / total_market if total_market > 0 else 0

        return {
            "ftrv_sales": ftrv_sales,
            "ftrv_ssi_matched": ftrv_ssi_sales,
            "ftrv_internal_count": ftrv_internal_sales,
            "total_market": total_market,
            "market_share": round(market_share, 4),
            "market_share_pct": f"{market_share:.1%}",
            "btas_used": len(ftrv_btas) if use_ftrv_btas else 0,
            "dealers_matched": len(ftrv_dealers),
            "data_source": "SSI" if ftrv_ssi_sales > 0 else "Internal"
        }

    def get_market_share_by_bta(
        self,
        manufacturer: str = None,
        veh_type: str = None
    ) -> pd.DataFrame:
        """
        Get market share breakdown by BTA (Basic Trading Area).
        Returns DataFrame with market share for each FTRV BTA.
        """
        try:
            ssi = self.data_loader.load_ssi_market_data()
            ssi_dealers = self.data_loader.load_ssi_dealers()
            locations = self.data_loader.load_locations()
        except Exception:
            return pd.DataFrame()

        if ssi.empty:
            return pd.DataFrame()

        ftrv_btas = self.get_ftrv_btas()
        ftrv_dealers = self.get_ftrv_ssi_dealers()

        if not ftrv_btas:
            return pd.DataFrame()

        # Build dealer-to-BTA mapping
        dealer_bta_map = {}
        if "Dealer" in ssi_dealers.columns and "DealerBTA" in ssi_dealers.columns:
            for _, row in ssi_dealers.iterrows():
                dealer = str(row["Dealer"]).strip().upper()
                bta_val = str(row["DealerBTA"]).strip()
                dealer_bta_map[dealer] = bta_val

        # Map SSI data to BTAs
        if "Dealer" in ssi.columns:
            ssi = ssi.copy()
            ssi["BTA"] = ssi["Dealer"].apply(
                lambda d: dealer_bta_map.get(str(d).strip().upper(), "") if pd.notna(d) else ""
            )

        # Apply manufacturer/veh_type filters
        ssi_mask = pd.Series(True, index=ssi.index)
        if manufacturer and "Manufacturer" in ssi.columns:
            ssi_mask &= ssi["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type and "Type" in ssi.columns:
            ssi_mask &= ssi["Type"].str.upper().str.contains(veh_type.upper(), na=False)
        ssi_filtered = ssi[ssi_mask]

        # Identify FTRV dealers
        ftrv_dealers_upper = set(str(d).upper().strip() for d in ftrv_dealers if pd.notna(d))

        def is_ftrv(dealer):
            if pd.isna(dealer):
                return False
            d_upper = str(dealer).strip().upper()
            for fd in ftrv_dealers_upper:
                if fd in d_upper or d_upper in fd:
                    return True
            return False

        ssi_filtered = ssi_filtered.copy()
        ssi_filtered["Is_FTRV"] = ssi_filtered["Dealer"].apply(is_ftrv)

        results = []
        for bta in ftrv_btas:
            if pd.isna(bta) or not str(bta).strip():
                continue

            bta_clean = str(bta).strip()

            # Filter to this BTA
            if "BTA" not in ssi_filtered.columns:
                continue

            bta_mask = ssi_filtered["BTA"].str.strip() == bta_clean
            bta_data = ssi_filtered[bta_mask]

            if len(bta_data) == 0:
                results.append({
                    "BTA": bta_clean,
                    "FTRV Sales": 0,
                    "Total Market": 0,
                    "Market Share": 0,
                    "Share %": "0.0%"
                })
                continue

            # Calculate totals - handle Units column safely
            try:
                if "Units" in bta_data.columns:
                    total_market = int(bta_data["Units"].sum())
                    ftrv_mask = bta_data["Is_FTRV"] == True
                    ftrv_sales = int(bta_data.loc[ftrv_mask, "Units"].sum()) if ftrv_mask.any() else 0
                else:
                    # Count rows as fallback
                    total_market = len(bta_data)
                    ftrv_sales = int(bta_data["Is_FTRV"].sum())
            except Exception:
                total_market = len(bta_data)
                ftrv_sales = 0

            share = ftrv_sales / total_market if total_market > 0 else 0

            results.append({
                "BTA": bta_clean,
                "FTRV Sales": ftrv_sales,
                "Total Market": total_market,
                "Market Share": round(share, 4),
                "Share %": f"{share:.1%}"
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("Total Market", ascending=False)
        return df

    def get_competitive_landscape(
        self,
        veh_type: str = None,
        price_group: str = None
    ) -> pd.DataFrame:
        """
        Get competitive landscape - top manufacturers by market volume.
        Returns DataFrame with manufacturer market stats.
        """
        try:
            ssi = self.data_loader.load_ssi_market_data()
        except Exception:
            return pd.DataFrame()

        mask = pd.Series(True, index=ssi.index)
        if veh_type and "Type" in ssi.columns:
            mask &= ssi["Type"].str.upper().str.contains(veh_type.upper(), na=False)
        if price_group and "Price Group" in ssi.columns:
            mask &= ssi["Price Group"].str.upper() == price_group.upper()

        filtered = ssi[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        mfg_stats = (
            filtered.groupby("Manufacturer")
            .agg({"Units": "sum"})
            .reset_index()
        )

        total = mfg_stats["Units"].sum()
        mfg_stats["Market Share"] = mfg_stats["Units"] / total
        mfg_stats = mfg_stats.sort_values("Units", ascending=False)

        return mfg_stats.head(20)

    # =========================================================================
    # LOCATION ANALYSIS
    # =========================================================================

    def get_sales_by_location(
        self,
        manufacturer: str = None,
        veh_type: str = None,
        lookback_months: int = None
    ) -> pd.DataFrame:
        """Get sales performance by location."""
        lookback = lookback_months or self.config.lookback_months
        retail = self.data_loader.load_retail_history()

        mask = pd.Series(True, index=retail.index)
        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        cutoff_date = datetime.now() - timedelta(days=lookback * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = "mean"
        if "Deal FrontEnd" in filtered.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"

        location_stats = (
            filtered.groupby("PC")
            .agg(agg_dict)
            .reset_index()
        )

        location_stats.columns = self._flatten_columns(location_stats.columns, ["PC"])

        # Rename PC column
        location_stats = location_stats.rename(columns={"PC": "Location"})

        location_stats["Monthly Velocity"] = location_stats["Units Sold"] / lookback
        location_stats = location_stats.sort_values("Units Sold", ascending=False)

        # Add zone information
        locations = self.data_loader.load_locations()
        if "PC" in locations.columns and "Zone" in locations.columns:
            zone_map = dict(zip(locations["PC"], locations["Zone"]))
            location_stats["Zone"] = location_stats["Location"].map(zone_map)

        return location_stats

    def get_sales_by_zone(
        self,
        manufacturer: str = None,
        veh_type: str = None,
        lookback_months: int = None
    ) -> pd.DataFrame:
        """Get sales aggregated by zone."""
        location_stats = self.get_sales_by_location(
            manufacturer=manufacturer,
            veh_type=veh_type,
            lookback_months=lookback_months
        )

        if location_stats.empty or "Zone" not in location_stats.columns:
            return pd.DataFrame()

        zone_stats = (
            location_stats.groupby("Zone")
            .agg({
                "Units Sold": "sum",
                "Monthly Velocity": "sum",
                "Location": "count"
            })
            .rename(columns={"Location": "Locations"})
            .reset_index()
            .sort_values("Units Sold", ascending=False)
        )

        return zone_stats

    # =========================================================================
    # GAP ANALYSIS
    # =========================================================================

    def identify_market_gaps(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """
        Identify segments where market demand exceeds FTRV inventory coverage.
        Returns DataFrame with gap analysis.
        """
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()

        if manufacturer:
            inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        # Get inventory by segment
        inv_by_segment = (
            inventory.groupby(["Veh Type", "Sub Floorplan", "Price Group"])
            .size()
            .reset_index(name="Current Inventory")
        )

        # Get sales velocity by segment
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        recent_retail = retail[retail["Sold Date"] >= cutoff_date] if "Sold Date" in retail.columns else retail

        sales_by_segment = (
            recent_retail.groupby(["Veh Type", "Sub Floorplan", "Price Group"])
            .size()
            .reset_index(name="Units Sold")
        )
        sales_by_segment["Monthly Velocity"] = sales_by_segment["Units Sold"] / self.config.lookback_months

        # Merge
        gaps = pd.merge(
            sales_by_segment,
            inv_by_segment,
            on=["Veh Type", "Sub Floorplan", "Price Group"],
            how="outer"
        ).fillna(0)

        # Calculate days supply
        gaps["Days Supply"] = np.where(
            gaps["Monthly Velocity"] > 0,
            gaps["Current Inventory"] / gaps["Monthly Velocity"] * 30,
            999
        )

        # Flag gaps
        gaps["Gap Status"] = "OK"
        gaps.loc[gaps["Days Supply"] < self.config.min_days_supply, "Gap Status"] = "UNDERSTOCKED"
        gaps.loc[gaps["Days Supply"] > self.config.max_days_supply, "Gap Status"] = "OVERSTOCKED"
        gaps.loc[gaps["Days Supply"] == 999, "Gap Status"] = "NO VELOCITY"

        gaps = gaps.sort_values(["Gap Status", "Days Supply"])

        return gaps
