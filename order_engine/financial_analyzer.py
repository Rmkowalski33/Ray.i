"""
Financial Analyzer Module
Evaluates profitability, deal economics, and ROI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .config import Config, default_config
from .data_loader import DataLoader


class FinancialAnalyzer:
    """Analyze financial performance and deal economics."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)

    # =========================================================================
    # MARGIN ANALYSIS
    # =========================================================================

    def calculate_segment_margin(
        self,
        manufacturer: str = None,
        make: str = None,
        veh_type: str = None,
        floorplan: str = None
    ) -> Dict:
        """
        Calculate average front-end margin for a segment.
        Returns dict with margin metrics.
        """
        retail = self.data_loader.load_retail_history()

        mask = pd.Series(True, index=retail.index)
        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if make:
            mask &= retail["Make"].str.upper() == make.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()
        if floorplan and "Sub Floorplan" in retail.columns:
            mask &= retail["Sub Floorplan"].str.upper() == floorplan.upper()

        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return {
                "avg_front_end": 0,
                "avg_margin_pct": 0,
                "total_front_end": 0,
                "sample_size": 0
            }

        avg_front_end = filtered["Deal FrontEnd"].mean() if "Deal FrontEnd" in filtered.columns else 0
        total_front_end = filtered["Deal FrontEnd"].sum() if "Deal FrontEnd" in filtered.columns else 0

        # Calculate margin percentage
        if "Retail Price" in filtered.columns and "Total Cost" in filtered.columns:
            filtered_valid = filtered[
                (filtered["Retail Price"] > 0) &
                (filtered["Total Cost"] > 0)
            ]
            if len(filtered_valid) > 0:
                avg_margin_pct = (
                    (filtered_valid["Retail Price"] - filtered_valid["Total Cost"]) /
                    filtered_valid["Retail Price"]
                ).mean()
            else:
                avg_margin_pct = 0
        else:
            avg_margin_pct = 0

        return {
            "avg_front_end": round(avg_front_end, 2),
            "avg_margin_pct": round(avg_margin_pct, 4),
            "total_front_end": round(total_front_end, 2),
            "sample_size": len(filtered),
            "avg_selling_price": round(filtered["Retail Price"].mean(), 2) if "Retail Price" in filtered.columns else 0,
            "avg_cost": round(filtered["Total Cost"].mean(), 2) if "Total Cost" in filtered.columns else 0
        }

    def get_margin_by_model(
        self,
        manufacturer: str,
        veh_type: str = None
    ) -> pd.DataFrame:
        """Get margin performance by model."""
        retail = self.data_loader.load_retail_history()

        mask = retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = "mean"
        if "Total Cost" in filtered.columns:
            agg_dict["Total Cost"] = "mean"
        if "Deal FrontEnd" in filtered.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"

        model_margins = (
            filtered.groupby(["Make", "Model"])
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten columns
        flat_cols = []
        for col in model_margins.columns:
            if isinstance(col, tuple):
                if col[1] == "" or col[1] == "count" or col[1] == "mean" or col[1] == "sum":
                    # Handle group columns and simple aggregations
                    if col[0] == "Stock#":
                        flat_cols.append("Units Sold")
                    elif col[0] == "Retail Price":
                        flat_cols.append("Avg Selling Price")
                    elif col[0] == "Total Cost":
                        flat_cols.append("Avg Cost")
                    elif col[0] == "Deal FrontEnd" and col[1] == "mean":
                        flat_cols.append("Avg Front-End")
                    elif col[0] == "Deal FrontEnd" and col[1] == "sum":
                        flat_cols.append("Total Front-End")
                    elif col[0] == "Age":
                        flat_cols.append("Avg Days to Sell")
                    elif col[1] == "":
                        flat_cols.append(col[0])  # Group columns like Make, Model
                    else:
                        flat_cols.append(f"{col[0]}_{col[1]}")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        model_margins.columns = flat_cols

        # Calculate margin percentage
        if "Avg Selling Price" in model_margins.columns and "Avg Cost" in model_margins.columns:
            model_margins["Margin %"] = (
                (model_margins["Avg Selling Price"] - model_margins["Avg Cost"]) /
                model_margins["Avg Selling Price"]
            ).clip(lower=0)

            # ROI score (margin adjusted for turn rate)
            if "Avg Days to Sell" in model_margins.columns:
                avg_days = model_margins["Avg Days to Sell"].mean()
                model_margins["ROI Score"] = (
                    model_margins["Margin %"] *
                    (avg_days / model_margins["Avg Days to Sell"].clip(lower=1))
                )

        return model_margins.sort_values("Total Front-End", ascending=False)

    def get_margin_by_make(
        self,
        manufacturer: str = None,
        veh_type: str = None
    ) -> pd.DataFrame:
        """
        Get margin performance by make (brand/series level).

        This is the appropriate aggregation level for ordering decisions.
        """
        retail = self.data_loader.load_retail_history()

        mask = pd.Series(True, index=retail.index)
        if manufacturer:
            mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= retail["Veh Type"].str.upper() == veh_type.upper()

        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        if "Sold Date" in retail.columns:
            mask &= retail["Sold Date"] >= cutoff_date

        filtered = retail[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        # Aggregate at MAKE level (not model)
        group_cols = ["Manufacturer", "Make"]
        if "Veh Type" in filtered.columns:
            group_cols.append("Veh Type")

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = "mean"
        if "Total Cost" in filtered.columns:
            agg_dict["Total Cost"] = "mean"
        if "Deal FrontEnd" in filtered.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"

        make_margins = (
            filtered.groupby(group_cols)
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten columns
        flat_cols = []
        for col in make_margins.columns:
            if isinstance(col, tuple):
                if col[1] == "" or col[1] == "count" or col[1] == "mean" or col[1] == "sum":
                    if col[0] == "Stock#":
                        flat_cols.append("Units Sold")
                    elif col[0] == "Retail Price":
                        flat_cols.append("Avg Selling Price")
                    elif col[0] == "Total Cost":
                        flat_cols.append("Avg Cost")
                    elif col[0] == "Deal FrontEnd" and col[1] == "mean":
                        flat_cols.append("Avg Front-End")
                    elif col[0] == "Deal FrontEnd" and col[1] == "sum":
                        flat_cols.append("Total Front-End")
                    elif col[0] == "Age":
                        flat_cols.append("Avg Days to Sell")
                    elif col[1] == "":
                        flat_cols.append(col[0])
                    else:
                        flat_cols.append(f"{col[0]}_{col[1]}")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        make_margins.columns = flat_cols

        # Calculate margin percentage
        if "Avg Selling Price" in make_margins.columns and "Avg Cost" in make_margins.columns:
            make_margins["Margin %"] = (
                (make_margins["Avg Selling Price"] - make_margins["Avg Cost"]) /
                make_margins["Avg Selling Price"]
            ).clip(lower=0)

            # ROI score (margin adjusted for turn rate)
            if "Avg Days to Sell" in make_margins.columns:
                avg_days = make_margins["Avg Days to Sell"].mean()
                make_margins["ROI Score"] = (
                    make_margins["Margin %"] *
                    (avg_days / make_margins["Avg Days to Sell"].clip(lower=1))
                )

        return make_margins.sort_values("Total Front-End", ascending=False)

    # =========================================================================
    # HOLDING COST ANALYSIS
    # =========================================================================

    def calculate_holding_cost(
        self,
        cost: float,
        days: int
    ) -> float:
        """Calculate floorplan interest cost for holding a unit."""
        return self.config.calculate_holding_cost(cost, days)

    def get_total_holding_cost(
        self,
        manufacturer: str = None
    ) -> Dict:
        """Calculate total accumulated holding costs for current inventory."""
        inventory = self.data_loader.load_current_inventory()

        if manufacturer:
            inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]

        if "Status Category" in inventory.columns:
            sellable = inventory[inventory["Status Category"] == "Sellable"]
        else:
            sellable = inventory

        if len(sellable) == 0:
            return {
                "total_holding_cost": 0,
                "avg_holding_cost_per_unit": 0,
                "total_units": 0
            }

        sellable = sellable.copy()
        if "Total Cost" in sellable.columns and "Age" in sellable.columns:
            sellable["Holding Cost"] = sellable.apply(
                lambda row: self.calculate_holding_cost(row["Total Cost"], row["Age"]),
                axis=1
            )
        else:
            sellable["Holding Cost"] = 0

        return {
            "total_holding_cost": round(sellable["Holding Cost"].sum(), 2),
            "avg_holding_cost_per_unit": round(sellable["Holding Cost"].mean(), 2),
            "total_units": len(sellable),
            "avg_unit_age": round(sellable["Age"].mean(), 1) if "Age" in sellable.columns else 0,
            "total_inventory_cost": round(sellable["Total Cost"].sum(), 2) if "Total Cost" in sellable.columns else 0
        }

    def get_holding_cost_by_segment(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """Get holding cost breakdown by segment."""
        inventory = self.data_loader.load_current_inventory()

        if manufacturer:
            inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]

        if "Status Category" in inventory.columns:
            sellable = inventory[inventory["Status Category"] == "Sellable"].copy()
        else:
            sellable = inventory.copy()

        if len(sellable) == 0:
            return pd.DataFrame()

        if "Total Cost" in sellable.columns and "Age" in sellable.columns:
            sellable["Holding Cost"] = sellable.apply(
                lambda row: self.calculate_holding_cost(row["Total Cost"], row["Age"]),
                axis=1
            )
        else:
            return pd.DataFrame()

        group_cols = ["Veh Type"]
        if "Sub Floorplan" in sellable.columns:
            group_cols.append("Sub Floorplan")

        segment_costs = (
            sellable.groupby(group_cols)
            .agg({
                "Stock#": "count",
                "Total Cost": "sum",
                "Holding Cost": "sum",
                "Age": "mean"
            })
            .reset_index()
        )

        segment_costs.columns = group_cols + ["Units", "Total Cost", "Total Holding Cost", "Avg Age"]

        segment_costs["Avg Holding Cost/Unit"] = segment_costs["Total Holding Cost"] / segment_costs["Units"]

        return segment_costs.sort_values("Total Holding Cost", ascending=False)

    # =========================================================================
    # ROI RANKING
    # =========================================================================

    def rank_segments_by_roi(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """Rank segments by ROI (margin adjusted for turn rate)."""
        retail = self.data_loader.load_retail_history()

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        recent = retail[retail["Sold Date"] >= cutoff_date] if "Sold Date" in retail.columns else retail

        if len(recent) == 0:
            return pd.DataFrame()

        group_cols = ["Veh Type"]
        if "Sub Floorplan" in recent.columns:
            group_cols.append("Sub Floorplan")
        if "Price Group" in recent.columns:
            group_cols.append("Price Group")

        agg_dict = {"Stock#": "count"}
        if "Deal FrontEnd" in recent.columns:
            agg_dict["Deal FrontEnd"] = ["mean", "sum"]
        if "Total Cost" in recent.columns:
            agg_dict["Total Cost"] = "mean"
        if "Age" in recent.columns:
            agg_dict["Age"] = "mean"

        segment_roi = (
            recent.groupby(group_cols)
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten columns
        flat_cols = []
        for col in segment_roi.columns:
            if isinstance(col, tuple):
                if col[0] == "Stock#":
                    flat_cols.append("Units Sold")
                elif col[0] == "Deal FrontEnd" and col[1] == "mean":
                    flat_cols.append("Avg Front-End")
                elif col[0] == "Deal FrontEnd" and col[1] == "sum":
                    flat_cols.append("Total Front-End")
                elif col[0] == "Total Cost":
                    flat_cols.append("Avg Cost")
                elif col[0] == "Age":
                    flat_cols.append("Avg Days to Sell")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        segment_roi.columns = flat_cols

        # Calculate ROI components
        if "Avg Days to Sell" in segment_roi.columns:
            segment_roi["Annual Turns"] = 365 / segment_roi["Avg Days to Sell"].clip(lower=30)

        if "Avg Front-End" in segment_roi.columns and "Avg Cost" in segment_roi.columns:
            segment_roi["Margin % of Cost"] = segment_roi["Avg Front-End"] / segment_roi["Avg Cost"].clip(lower=1)

        if "Margin % of Cost" in segment_roi.columns and "Annual Turns" in segment_roi.columns:
            segment_roi["ROI Score"] = segment_roi["Margin % of Cost"] * segment_roi["Annual Turns"]

        if "Avg Cost" in segment_roi.columns and "Avg Days to Sell" in segment_roi.columns:
            segment_roi["Holding Cost Drag"] = (
                segment_roi["Avg Cost"] *
                self.config.floorplan_rate *
                (segment_roi["Avg Days to Sell"] / 365)
            )

        if "Avg Front-End" in segment_roi.columns and "Holding Cost Drag" in segment_roi.columns:
            segment_roi["Net Margin"] = segment_roi["Avg Front-End"] - segment_roi["Holding Cost Drag"]

        return segment_roi.sort_values("ROI Score", ascending=False) if "ROI Score" in segment_roi.columns else segment_roi

    # =========================================================================
    # PROFITABILITY FORECAST
    # =========================================================================

    def forecast_order_profitability(
        self,
        units: int,
        avg_cost: float,
        expected_margin_pct: float,
        expected_days_to_sell: int
    ) -> Dict:
        """Forecast profitability for a potential order."""
        expected_selling_price = avg_cost / (1 - expected_margin_pct)
        expected_revenue = units * expected_selling_price
        expected_gross_margin = expected_revenue - (units * avg_cost)

        holding_cost = self.calculate_holding_cost(avg_cost, expected_days_to_sell) * units

        net_profit = expected_gross_margin - holding_cost

        turn_rate = 365 / expected_days_to_sell
        annualized_roi = (net_profit / (units * avg_cost)) * turn_rate

        return {
            "units": units,
            "total_cost": round(units * avg_cost, 2),
            "expected_revenue": round(expected_revenue, 2),
            "expected_gross_margin": round(expected_gross_margin, 2),
            "holding_cost": round(holding_cost, 2),
            "net_profit": round(net_profit, 2),
            "margin_pct": round(expected_margin_pct, 4),
            "holding_cost_pct": round(holding_cost / (units * avg_cost), 4),
            "net_margin_pct": round(net_profit / expected_revenue, 4),
            "expected_days_to_sell": expected_days_to_sell,
            "annualized_roi": round(annualized_roi, 4),
            "annualized_roi_pct": f"{annualized_roi:.1%}"
        }
