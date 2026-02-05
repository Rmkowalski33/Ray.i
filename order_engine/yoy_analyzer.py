"""
Year-over-Year Analyzer Module
Compares performance across years with location-adjusted metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .config import Config, default_config
from .data_loader import DataLoader


class YoYAnalyzer:
    """Analyze year-over-year performance with location adjustments."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)

    # =========================================================================
    # ACTIVE LOCATION COUNTING
    # =========================================================================

    def get_active_locations_by_period(
        self,
        period: str = "month"  # "month", "quarter", "year"
    ) -> pd.DataFrame:
        """
        Get count of active locations by period.

        A location is "active" if it had inventory OR retail activity.
        This is used to adjust YoY comparisons for location growth.
        """
        retail = self.data_loader.load_retail_history()
        inventory = self.data_loader.load_current_inventory()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        # Get unique locations from retail by period
        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        if period == "month":
            retail["Period"] = retail["Sold Date"].dt.to_period("M")
        elif period == "quarter":
            retail["Period"] = retail["Sold Date"].dt.to_period("Q")
        else:
            retail["Period"] = retail["Sold Date"].dt.to_period("Y")

        # Count unique locations with sales per period
        active_by_period = (
            retail.groupby("Period")["PC"]
            .nunique()
            .reset_index()
        )
        active_by_period.columns = ["Period", "Active Locations"]

        return active_by_period

    def get_active_locations_for_year(self, year: int) -> int:
        """Get count of locations that were active during a given year."""
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns or "PC" not in retail.columns:
            return 0

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        year_retail = retail[retail["Sold Date"].dt.year == year]

        return year_retail["PC"].nunique()

    # =========================================================================
    # YEAR-OVER-YEAR COMPARISON
    # =========================================================================

    def compare_yoy(
        self,
        manufacturer: str = None,
        veh_type: str = None,
        adjust_for_locations: bool = True,
        use_ytd: bool = True
    ) -> Dict:
        """
        Compare this year to last year with location adjustments.

        Uses YTD (Year-to-Date) comparison by default to ensure apples-to-apples
        comparison when we're partway through the current year.

        Args:
            manufacturer: Filter by manufacturer
            veh_type: Filter by vehicle type
            adjust_for_locations: If True, calculate per-location metrics
            use_ytd: If True, compare YTD periods (same date range in both years)

        Returns:
            Dict with YoY comparison metrics
        """
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns:
            return {"error": "No sold date column in retail data"}

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        # Apply filters
        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]
        if veh_type and "Veh Type" in retail.columns:
            retail = retail[retail["Veh Type"].str.upper() == veh_type.upper()]

        if len(retail) == 0:
            return {"error": "No data after filters applied"}

        # Define periods
        now = datetime.now()
        current_year = now.year
        last_year = current_year - 1

        retail["Year"] = retail["Sold Date"].dt.year
        retail["DayOfYear"] = retail["Sold Date"].dt.dayofyear

        # YTD comparison - only compare same date range in both years
        if use_ytd:
            current_day_of_year = now.timetuple().tm_yday

            # Current year YTD
            current_year_sales = retail[
                (retail["Year"] == current_year) &
                (retail["DayOfYear"] <= current_day_of_year)
            ]

            # Last year same period (YTD as of same date)
            last_year_sales = retail[
                (retail["Year"] == last_year) &
                (retail["DayOfYear"] <= current_day_of_year)
            ]

            period_desc = f"YTD through {now.strftime('%b %d')}"
        else:
            # Full year comparison
            current_year_sales = retail[retail["Year"] == current_year]
            last_year_sales = retail[retail["Year"] == last_year]
            period_desc = "Full Year"

        cy_units = len(current_year_sales)
        ly_units = len(last_year_sales)

        # Revenue if available
        cy_revenue = current_year_sales["Retail Price"].sum() if "Retail Price" in current_year_sales.columns else 0
        ly_revenue = last_year_sales["Retail Price"].sum() if "Retail Price" in last_year_sales.columns else 0

        # Front-end if available
        cy_front_end = current_year_sales["Deal FrontEnd"].sum() if "Deal FrontEnd" in current_year_sales.columns else 0
        ly_front_end = last_year_sales["Deal FrontEnd"].sum() if "Deal FrontEnd" in last_year_sales.columns else 0

        # Calculate raw changes
        units_change = cy_units - ly_units
        units_change_pct = (units_change / ly_units) if ly_units > 0 else 0
        revenue_change_pct = ((cy_revenue - ly_revenue) / ly_revenue) if ly_revenue > 0 else 0

        result = {
            "comparison_type": period_desc,
            "current_year": current_year,
            "last_year": last_year,
            "current_year_units": cy_units,
            "last_year_units": ly_units,
            "units_change": units_change,
            "units_change_pct": round(units_change_pct, 4),
            "units_change_pct_formatted": f"{units_change_pct:+.1%}",
            "current_year_revenue": round(cy_revenue, 2),
            "last_year_revenue": round(ly_revenue, 2),
            "revenue_change_pct": round(revenue_change_pct, 4),
            "current_year_front_end": round(cy_front_end, 2),
            "last_year_front_end": round(ly_front_end, 2),
        }

        # Location-adjusted metrics
        if adjust_for_locations:
            # Get locations active in the YTD period for each year
            if use_ytd:
                cy_locations = current_year_sales["PC"].nunique() if "PC" in current_year_sales.columns else 0
                ly_locations = last_year_sales["PC"].nunique() if "PC" in last_year_sales.columns else 0
            else:
                cy_locations = self.get_active_locations_for_year(current_year)
                ly_locations = self.get_active_locations_for_year(last_year)

            if cy_locations > 0 and ly_locations > 0:
                cy_per_location = cy_units / cy_locations
                ly_per_location = ly_units / ly_locations
                adjusted_change = cy_per_location - ly_per_location
                adjusted_change_pct = (adjusted_change / ly_per_location) if ly_per_location > 0 else 0

                result["current_year_locations"] = cy_locations
                result["last_year_locations"] = ly_locations
                result["location_growth"] = cy_locations - ly_locations
                result["location_growth_pct"] = round((cy_locations - ly_locations) / ly_locations, 4) if ly_locations > 0 else 0
                result["current_year_per_location"] = round(cy_per_location, 2)
                result["last_year_per_location"] = round(ly_per_location, 2)
                result["adjusted_change_pct"] = round(adjusted_change_pct, 4)
                result["adjusted_change_pct_formatted"] = f"{adjusted_change_pct:+.1%}"
                result["interpretation"] = self._interpret_yoy(
                    units_change_pct, adjusted_change_pct, cy_locations - ly_locations
                )

        return result

    def _interpret_yoy(
        self,
        raw_change: float,
        adjusted_change: float,
        location_growth: int
    ) -> str:
        """Generate interpretation of YoY changes."""
        if location_growth > 0:
            if adjusted_change > 0.05:
                return f"Strong organic growth (+{adjusted_change:.1%}) beyond {location_growth} new locations"
            elif adjusted_change > 0:
                return f"Positive growth (+{adjusted_change:.1%}) on per-location basis with {location_growth} new locations"
            elif adjusted_change > -0.05:
                return f"Growth driven by {location_growth} new locations; per-location sales flat"
            else:
                return f"Despite {location_growth} new locations, per-location sales down {adjusted_change:.1%}"
        else:
            if raw_change > 0:
                return f"Sales growth of {raw_change:.1%} with stable location count"
            else:
                return f"Sales decline of {raw_change:.1%} - review demand patterns"

    def compare_yoy_by_quarter(
        self,
        manufacturer: str = None,
        adjust_for_locations: bool = True
    ) -> pd.DataFrame:
        """
        Get quarterly YoY comparison.

        Returns DataFrame with quarterly comparisons.
        """
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        retail["Year"] = retail["Sold Date"].dt.year
        retail["Quarter"] = retail["Sold Date"].dt.quarter

        now = datetime.now()
        current_year = now.year
        last_year = current_year - 1

        results = []
        for q in range(1, 5):
            cy_sales = retail[(retail["Year"] == current_year) & (retail["Quarter"] == q)]
            ly_sales = retail[(retail["Year"] == last_year) & (retail["Quarter"] == q)]

            cy_units = len(cy_sales)
            ly_units = len(ly_sales)
            change_pct = ((cy_units - ly_units) / ly_units) if ly_units > 0 else 0

            row = {
                "Quarter": f"Q{q}",
                f"{last_year}": ly_units,
                f"{current_year}": cy_units,
                "Change": cy_units - ly_units,
                "Change %": f"{change_pct:+.1%}"
            }

            if adjust_for_locations:
                # Get location counts for each quarter
                cy_locs = retail[
                    (retail["Year"] == current_year) & (retail["Quarter"] == q)
                ]["PC"].nunique()
                ly_locs = retail[
                    (retail["Year"] == last_year) & (retail["Quarter"] == q)
                ]["PC"].nunique()

                if cy_locs > 0 and ly_locs > 0:
                    cy_per_loc = cy_units / cy_locs
                    ly_per_loc = ly_units / ly_locs
                    adj_change = ((cy_per_loc - ly_per_loc) / ly_per_loc) if ly_per_loc > 0 else 0
                    row["Adj. Change %"] = f"{adj_change:+.1%}"
                    row[f"{last_year} Locs"] = ly_locs
                    row[f"{current_year} Locs"] = cy_locs

            results.append(row)

        return pd.DataFrame(results)

    def compare_yoy_by_month(
        self,
        manufacturer: str = None,
        months_back: int = 12,
        adjust_for_locations: bool = True
    ) -> pd.DataFrame:
        """
        Get monthly YoY comparison.

        Returns DataFrame with monthly comparisons.
        """
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        retail["Year"] = retail["Sold Date"].dt.year
        retail["Month"] = retail["Sold Date"].dt.month
        retail["YearMonth"] = retail["Sold Date"].dt.to_period("M")

        now = datetime.now()
        current_year = now.year
        last_year = current_year - 1

        results = []
        for m in range(1, 13):
            month_name = datetime(2000, m, 1).strftime("%b")

            cy_sales = retail[(retail["Year"] == current_year) & (retail["Month"] == m)]
            ly_sales = retail[(retail["Year"] == last_year) & (retail["Month"] == m)]

            cy_units = len(cy_sales)
            ly_units = len(ly_sales)

            # Skip future months in current year
            if current_year == now.year and m > now.month:
                continue

            change_pct = ((cy_units - ly_units) / ly_units) if ly_units > 0 else 0

            row = {
                "Month": month_name,
                f"{last_year}": ly_units,
                f"{current_year}": cy_units,
                "Change": cy_units - ly_units,
                "Change %": f"{change_pct:+.1%}"
            }

            if adjust_for_locations:
                cy_locs = cy_sales["PC"].nunique() if len(cy_sales) > 0 else 0
                ly_locs = ly_sales["PC"].nunique() if len(ly_sales) > 0 else 0

                if cy_locs > 0 and ly_locs > 0:
                    cy_per_loc = cy_units / cy_locs
                    ly_per_loc = ly_units / ly_locs
                    adj_change = ((cy_per_loc - ly_per_loc) / ly_per_loc) if ly_per_loc > 0 else 0
                    row["Adj. Change %"] = f"{adj_change:+.1%}"

            results.append(row)

        return pd.DataFrame(results)

    # =========================================================================
    # SEGMENT-LEVEL YoY
    # =========================================================================

    def compare_yoy_by_segment(
        self,
        manufacturer: str = None,
        segment_by: str = "veh_type"  # "veh_type", "make", "zone"
    ) -> pd.DataFrame:
        """
        Get YoY comparison by segment.

        Args:
            manufacturer: Filter by manufacturer
            segment_by: Column to segment by ("veh_type", "make", "zone")

        Returns:
            DataFrame with segment-level YoY comparisons
        """
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        # Map segment_by to column name
        segment_col_map = {
            "veh_type": "Veh Type",
            "make": "Make",
            "zone": "Zone"
        }
        segment_col = segment_col_map.get(segment_by, "Veh Type")

        # Add zone from locations if needed
        if segment_by == "zone" and "Zone" not in retail.columns:
            locations = self.data_loader.load_locations()
            if "PC" in locations.columns and "Zone" in locations.columns:
                zone_map = dict(zip(locations["PC"], locations["Zone"]))
                retail["Zone"] = retail["PC"].map(zone_map)

        if segment_col not in retail.columns:
            return pd.DataFrame()

        retail["Year"] = retail["Sold Date"].dt.year

        now = datetime.now()
        current_year = now.year
        last_year = current_year - 1

        # Group by segment and year
        segment_sales = (
            retail[retail["Year"].isin([current_year, last_year])]
            .groupby([segment_col, "Year"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        if current_year not in segment_sales.columns:
            segment_sales[current_year] = 0
        if last_year not in segment_sales.columns:
            segment_sales[last_year] = 0

        segment_sales["Change"] = segment_sales[current_year] - segment_sales[last_year]
        segment_sales["Change %"] = np.where(
            segment_sales[last_year] > 0,
            ((segment_sales[current_year] - segment_sales[last_year]) / segment_sales[last_year]),
            0
        )
        segment_sales["Change %"] = segment_sales["Change %"].apply(lambda x: f"{x:+.1%}")

        segment_sales = segment_sales.rename(columns={segment_col: "Segment"})
        segment_sales = segment_sales.sort_values(current_year, ascending=False)

        return segment_sales

    # =========================================================================
    # PERFORMANCE TRENDS
    # =========================================================================

    def get_rolling_trend(
        self,
        manufacturer: str = None,
        window_months: int = 3
    ) -> pd.DataFrame:
        """
        Get rolling trend of sales over time.

        Returns DataFrame with rolling average sales.
        """
        retail = self.data_loader.load_retail_history()

        if "Sold Date" not in retail.columns:
            return pd.DataFrame()

        retail = retail.copy()
        retail["Sold Date"] = pd.to_datetime(retail["Sold Date"], errors="coerce")
        retail = retail[retail["Sold Date"].notna()]

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        retail["YearMonth"] = retail["Sold Date"].dt.to_period("M")

        monthly = (
            retail.groupby("YearMonth")
            .size()
            .reset_index(name="Units")
        )

        monthly = monthly.sort_values("YearMonth")
        monthly[f"{window_months}mo Rolling Avg"] = monthly["Units"].rolling(window=window_months).mean()

        monthly["Period"] = monthly["YearMonth"].astype(str)

        return monthly[["Period", "Units", f"{window_months}mo Rolling Avg"]]
