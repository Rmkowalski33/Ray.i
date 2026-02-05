"""
Pipeline Analyzer Module
Tracks inventory lifecycle, days supply, and pipeline health.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .config import Config, default_config
from .data_loader import DataLoader


class PipelineAnalyzer:
    """Analyze inventory pipeline and lifecycle."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)

    # =========================================================================
    # DAYS SUPPLY ANALYSIS
    # =========================================================================

    def calculate_days_supply(
        self,
        manufacturer: str = None,
        make: str = None,
        veh_type: str = None,
        floorplan: str = None,
        include_pipeline: bool = False
    ) -> Dict:
        """
        Calculate days supply for a segment.

        Days Supply = Current Inventory / (Monthly Sales Velocity / 30)

        Returns dict with:
        - days_supply: Current days of inventory
        - current_inventory: Units on lot
        - monthly_velocity: Sales rate
        - status: "critically_low", "low", "optimal", "high", "overstocked"
        """
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()

        # Filter inventory
        inv_mask = pd.Series(True, index=inventory.index)
        if manufacturer:
            inv_mask &= inventory["Manufacturer"].str.upper() == manufacturer.upper()
        if make:
            inv_mask &= inventory["Make"].str.upper() == make.upper()
        if veh_type:
            inv_mask &= inventory["Veh Type"].str.upper() == veh_type.upper()
        if floorplan and "Sub Floorplan" in inventory.columns:
            inv_mask &= inventory["Sub Floorplan"].str.upper() == floorplan.upper()

        # Only count sellable inventory
        if "Status Category" in inventory.columns:
            inv_mask &= inventory["Status Category"] == "Sellable"

        current_inventory = len(inventory[inv_mask])

        # Calculate velocity from retail
        retail_mask = pd.Series(True, index=retail.index)
        if manufacturer:
            retail_mask &= retail["Manufacturer"].str.upper() == manufacturer.upper()
        if make:
            retail_mask &= retail["Make"].str.upper() == make.upper()
        if veh_type:
            retail_mask &= retail["Veh Type"].str.upper() == veh_type.upper()
        if floorplan and "Sub Floorplan" in retail.columns:
            retail_mask &= retail["Sub Floorplan"].str.upper() == floorplan.upper()

        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        if "Sold Date" in retail.columns:
            retail_mask &= retail["Sold Date"] >= cutoff_date

        monthly_velocity = len(retail[retail_mask]) / self.config.lookback_months

        # Include pipeline if requested
        pipeline_units = 0
        if include_pipeline:
            on_order = self.data_loader.load_on_order()
            if not on_order.empty:
                pipe_mask = pd.Series(True, index=on_order.index)
                if manufacturer and "Manufacturer" in on_order.columns:
                    pipe_mask &= on_order["Manufacturer"].str.upper() == manufacturer.upper()
                if make and "Make" in on_order.columns:
                    pipe_mask &= on_order["Make"].str.upper() == make.upper()
                if veh_type and "Veh Type" in on_order.columns:
                    pipe_mask &= on_order["Veh Type"].str.upper() == veh_type.upper()
                pipeline_units = len(on_order[pipe_mask])

        # Calculate days supply
        if monthly_velocity > 0:
            daily_velocity = monthly_velocity / 30
            days_supply = current_inventory / daily_velocity
            total_days_supply = (current_inventory + pipeline_units) / daily_velocity
        else:
            days_supply = 999
            total_days_supply = 999

        # Determine status
        status = self._classify_days_supply(days_supply)

        return {
            "current_inventory": current_inventory,
            "pipeline_units": pipeline_units,
            "monthly_velocity": round(monthly_velocity, 2),
            "days_supply": round(days_supply, 1),
            "total_days_supply_with_pipeline": round(total_days_supply, 1),
            "status": status,
            "target_days_supply": self.config.target_days_supply
        }

    def _classify_days_supply(self, days_supply: float) -> str:
        """Classify days supply into status."""
        if days_supply < self.config.critical_days_supply:
            return "critically_low"
        elif days_supply < self.config.min_days_supply:
            return "low"
        elif days_supply <= self.config.max_days_supply:
            return "optimal"
        else:
            return "overstocked"

    def get_days_supply_by_segment(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """Get days supply for all segments."""
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()

        if manufacturer:
            inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        # Only sellable inventory
        if "Status Category" in inventory.columns:
            inventory = inventory[inventory["Status Category"] == "Sellable"]

        # Get inventory counts by segment
        group_cols = ["Veh Type"]
        if "Sub Floorplan" in inventory.columns:
            group_cols.append("Sub Floorplan")

        inv_by_segment = (
            inventory.groupby(group_cols)
            .agg({
                "Stock#": "count",
                "Age": "mean",
                "Retail Price": "mean"
            })
            .reset_index()
        )

        rename_map = {"Stock#": "Inventory", "Age": "Avg Age", "Retail Price": "Avg Price"}
        inv_by_segment = inv_by_segment.rename(columns=rename_map)

        # Get velocity by segment
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        recent_retail = retail[retail["Sold Date"] >= cutoff_date] if "Sold Date" in retail.columns else retail

        sales_by_segment = (
            recent_retail.groupby(group_cols)
            .size()
            .reset_index(name="Units Sold")
        )
        sales_by_segment["Monthly Velocity"] = sales_by_segment["Units Sold"] / self.config.lookback_months

        # Merge
        result = pd.merge(
            inv_by_segment,
            sales_by_segment,
            on=group_cols,
            how="outer"
        ).fillna(0)

        # Calculate days supply
        result["Days Supply"] = np.where(
            result["Monthly Velocity"] > 0,
            result["Inventory"] / result["Monthly Velocity"] * 30,
            999
        )

        # Classify status
        result["Status"] = result["Days Supply"].apply(self._classify_days_supply_label)

        # Sort by urgency
        status_order = {"Critical": 0, "Low": 1, "Optimal": 2, "Overstocked": 3}
        result["Sort Order"] = result["Status"].map(status_order)
        result = result.sort_values(["Sort Order", "Days Supply"]).drop("Sort Order", axis=1)

        return result

    def _classify_days_supply_label(self, days_supply: float) -> str:
        """Get label for days supply status."""
        if days_supply < self.config.critical_days_supply:
            return "Critical"
        elif days_supply < self.config.min_days_supply:
            return "Low"
        elif days_supply <= self.config.max_days_supply:
            return "Optimal"
        else:
            return "Overstocked"

    # =========================================================================
    # AGE DISTRIBUTION ANALYSIS
    # =========================================================================

    def get_age_distribution(
        self,
        manufacturer: str = None,
        veh_type: str = None
    ) -> pd.DataFrame:
        """Get inventory age distribution."""
        inventory = self.data_loader.load_current_inventory()

        mask = pd.Series(True, index=inventory.index)
        if manufacturer:
            mask &= inventory["Manufacturer"].str.upper() == manufacturer.upper()
        if veh_type:
            mask &= inventory["Veh Type"].str.upper() == veh_type.upper()
        if "Status Category" in inventory.columns:
            mask &= inventory["Status Category"] == "Sellable"

        filtered = inventory[mask]

        if len(filtered) == 0:
            return pd.DataFrame()

        if "Age Bucket" not in filtered.columns:
            filtered = filtered.copy()
            filtered["Age Bucket"] = filtered["Age"].apply(self.config.get_age_bucket)

        agg_dict = {"Stock#": "count"}
        if "Retail Price" in filtered.columns:
            agg_dict["Retail Price"] = ["sum", "mean"]
        if "Total Cost" in filtered.columns:
            agg_dict["Total Cost"] = "sum"
        if "Age" in filtered.columns:
            agg_dict["Age"] = "mean"

        age_dist = (
            filtered.groupby("Age Bucket")
            .agg(agg_dict)
            .reset_index()
        )

        # Flatten columns
        flat_cols = ["Age Bucket"]
        for col in age_dist.columns[1:]:
            if isinstance(col, tuple):
                if col[0] == "Stock#":
                    flat_cols.append("Units")
                elif col[0] == "Retail Price" and col[1] == "sum":
                    flat_cols.append("Total Retail Value")
                elif col[0] == "Retail Price" and col[1] == "mean":
                    flat_cols.append("Avg Retail Price")
                elif col[0] == "Total Cost":
                    flat_cols.append("Total Cost")
                elif col[0] == "Age":
                    flat_cols.append("Avg Age")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        age_dist.columns = flat_cols

        # Sort by age bucket
        bucket_order = list(self.config.age_buckets.keys())
        age_dist["Sort Order"] = age_dist["Age Bucket"].apply(
            lambda x: bucket_order.index(x) if x in bucket_order else 999
        )
        age_dist = age_dist.sort_values("Sort Order").drop("Sort Order", axis=1)

        # Calculate percentages
        total_units = age_dist["Units"].sum()
        age_dist["% of Inventory"] = age_dist["Units"] / total_units

        return age_dist

    def identify_aging_risk(
        self,
        manufacturer: str = None,
        age_threshold: int = 90
    ) -> pd.DataFrame:
        """Identify units at risk due to age."""
        inventory = self.data_loader.load_current_inventory()

        mask = inventory["Age"] > age_threshold
        if "Status Category" in inventory.columns:
            mask &= inventory["Status Category"] == "Sellable"
        if manufacturer:
            mask &= inventory["Manufacturer"].str.upper() == manufacturer.upper()

        aging = inventory[mask].copy()

        if len(aging) == 0:
            return pd.DataFrame()

        # Calculate accumulated holding cost
        if "Total Cost" in aging.columns and "Age" in aging.columns:
            aging["Est. Holding Cost"] = aging.apply(
                lambda row: self.config.calculate_holding_cost(row["Total Cost"], row["Age"]),
                axis=1
            )

        columns = [
            "Stock#", "Manufacturer", "Make", "Model", "Veh Type",
            "Age", "Retail Price", "Total Cost", "Est. Holding Cost",
            "PC", "Status"
        ]

        return aging[[c for c in columns if c in aging.columns]].sort_values("Age", ascending=False)

    # =========================================================================
    # INVENTORY POSITION SUMMARY
    # =========================================================================

    def get_inventory_position_summary(
        self,
        manufacturer: str = None
    ) -> Dict:
        """Get overall inventory position summary."""
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()
        on_order = self.data_loader.load_on_order()

        if manufacturer:
            inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]
            if not on_order.empty and "Manufacturer" in on_order.columns:
                on_order = on_order[on_order["Manufacturer"].str.upper() == manufacturer.upper()]

        # Sellable inventory
        if "Status Category" in inventory.columns:
            sellable = inventory[inventory["Status Category"] == "Sellable"]
        else:
            sellable = inventory

        # Recent sales
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_months * 30)
        recent_sales = retail[retail["Sold Date"] >= cutoff_date] if "Sold Date" in retail.columns else retail

        # Calculate metrics
        total_inventory = len(sellable)
        total_retail_value = sellable["Retail Price"].sum() if "Retail Price" in sellable.columns else 0
        total_cost = sellable["Total Cost"].sum() if "Total Cost" in sellable.columns else 0
        avg_age = sellable["Age"].mean() if "Age" in sellable.columns and len(sellable) > 0 else 0
        monthly_sales = len(recent_sales) / self.config.lookback_months
        pipeline_units = len(on_order)

        # Days supply
        days_supply = (total_inventory / (monthly_sales / 30)) if monthly_sales > 0 else 999

        # Age distribution
        aging_90_plus = len(sellable[sellable["Age"] > 90]) if "Age" in sellable.columns else 0
        aging_180_plus = len(sellable[sellable["Age"] > 180]) if "Age" in sellable.columns else 0

        return {
            "total_inventory": total_inventory,
            "total_retail_value": round(total_retail_value, 2),
            "total_cost": round(total_cost, 2),
            "avg_age_days": round(avg_age, 1),
            "days_supply": round(days_supply, 1),
            "monthly_sales_rate": round(monthly_sales, 1),
            "pipeline_units": pipeline_units,
            "aging_90_plus_units": aging_90_plus,
            "aging_180_plus_units": aging_180_plus,
            "aging_90_plus_pct": round(aging_90_plus / total_inventory, 3) if total_inventory > 0 else 0,
            "inventory_health": self._classify_inventory_health(days_supply, avg_age, aging_90_plus / total_inventory if total_inventory > 0 else 0)
        }

    def _classify_inventory_health(
        self,
        days_supply: float,
        avg_age: float,
        aging_pct: float
    ) -> str:
        """Classify overall inventory health."""
        score = 0

        # Days supply scoring
        if self.config.min_days_supply <= days_supply <= self.config.target_days_supply:
            score += 3
        elif self.config.target_days_supply < days_supply <= self.config.max_days_supply:
            score += 2
        elif days_supply < self.config.min_days_supply:
            score += 1

        # Age scoring
        if avg_age <= 60:
            score += 3
        elif avg_age <= 90:
            score += 2
        elif avg_age <= 120:
            score += 1

        # Aging percentage scoring
        if aging_pct <= 0.1:
            score += 3
        elif aging_pct <= 0.2:
            score += 2
        elif aging_pct <= 0.3:
            score += 1

        if score >= 8:
            return "Excellent"
        elif score >= 6:
            return "Good"
        elif score >= 4:
            return "Fair"
        else:
            return "Poor"

    # =========================================================================
    # ON ORDER / PIPELINE ANALYSIS
    # =========================================================================

    def get_on_order_summary(
        self,
        manufacturer: str = None
    ) -> Dict:
        """
        Get summary of units currently on order (in pipeline).

        Distinguishes between:
        - Units needing POs (ORDERED status - no PO yet)
        - Units with POs (PURCHASED, PO ISSUED - have PO)
        - Units needing location (YARD / PDI POOL - no location assigned)
        - Units in transit (DISPATCHED, SHIPPED, TRANSFER)

        Returns dict with:
        - total_on_order: Total units in pipeline
        - needs_po: Units that need PO assigned
        - has_po: Units with PO issued
        - needs_location: Units needing location assignment
        - in_transit: Units in transit to locations
        - by_status: Units broken down by order status
        - estimated_arrivals: Dict of arrival windows
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return {
                "total_on_order": 0,
                "needs_po": 0,
                "has_po": 0,
                "needs_location": 0,
                "in_transit": 0,
                "by_status": {},
                "avg_age": 0,
                "estimated_arrivals": {},
                "total_value": 0
            }

        if manufacturer and "Manufacturer" in on_order.columns:
            on_order = on_order[on_order["Manufacturer"].str.upper() == manufacturer.upper()]

        total_on_order = len(on_order)

        # Categorize by PO and location status
        needs_po = 0
        has_po = 0
        needs_location = 0
        in_transit = 0

        if "Status" in on_order.columns:
            status_upper = on_order["Status"].str.upper()

            # Units needing PO (ORDERED = unit entered but no PO)
            needs_po = len(on_order[status_upper == "ORDERED"])

            # Units with PO (PURCHASED, PO ISSUED)
            has_po = len(on_order[status_upper.isin(["PURCHASED", "PO ISSUED"])])

            # Units needing location assignment (YARD / PDI POOL)
            needs_location = len(on_order[
                status_upper.str.contains("YARD|PDI POOL|POOL", na=False)
            ])

            # Units in transit (DISPATCHED, SHIPPED, TRANSFER)
            in_transit = len(on_order[
                status_upper.isin(["DISPATCHED", "SHIPPED", "TRANSFER", "STORE-TO-STORE TRANSFER", "OPS TRANSFER"])
            ])

        # By status breakdown
        by_status = {}
        if "Status" in on_order.columns:
            status_counts = on_order["Status"].value_counts().to_dict()
            for status, count in status_counts.items():
                status_data = on_order[on_order["Status"] == status]
                by_status[status] = {
                    "count": count,
                    "avg_age": round(status_data["Age"].mean(), 1) if "Age" in status_data.columns else 0,
                    "total_cost": round(status_data["Total Cost"].sum(), 0) if "Total Cost" in status_data.columns else 0
                }

        # Average age
        avg_age = on_order["Age"].mean() if "Age" in on_order.columns and len(on_order) > 0 else 0

        # Estimate arrivals based on age/status
        estimated_arrivals = self._estimate_arrival_windows(on_order)

        # Total value
        total_value = on_order["Total Cost"].sum() if "Total Cost" in on_order.columns else 0

        return {
            "total_on_order": total_on_order,
            "needs_po": needs_po,
            "has_po": has_po,
            "needs_location": needs_location,
            "in_transit": in_transit,
            "by_status": by_status,
            "avg_age": round(avg_age, 1),
            "estimated_arrivals": estimated_arrivals,
            "total_value": round(total_value, 2)
        }

    def _estimate_arrival_windows(self, on_order: pd.DataFrame) -> Dict:
        """
        Estimate arrival windows based on order age and status.

        Uses lead time assumptions from config to project arrivals.
        """
        if on_order.empty:
            return {}

        now = datetime.now()

        # Estimate days until arrival based on status
        status_days_remaining = {
            "PO ISSUED": self.config.total_lead_time,  # Full lead time
            "IN PRODUCTION": self.config.transit_to_pdi_days + self.config.pdi_processing_days + self.config.transit_to_store_days,
            "IN TRANSIT": self.config.pdi_processing_days + self.config.transit_to_store_days,
            "AT PDI": self.config.pdi_processing_days + self.config.transit_to_store_days,
            "PDI COMPLETE": self.config.transit_to_store_days,
            "DISPATCHED": 7,  # Usually arrives within a week
        }

        arrivals = {}
        for status, days in status_days_remaining.items():
            if "Status" in on_order.columns:
                status_units = on_order[on_order["Status"].str.upper().str.contains(status, na=False)]
            else:
                continue

            if len(status_units) == 0:
                continue

            # Calculate arrival month
            arrival_date = now + timedelta(days=days)
            arrival_month = arrival_date.strftime("%b %Y")

            if arrival_month not in arrivals:
                arrivals[arrival_month] = 0
            arrivals[arrival_month] += len(status_units)

        return arrivals

    def get_on_order_details(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """
        Get detailed breakdown of units on order.

        Returns DataFrame with on-order units grouped by status, make, model.
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return pd.DataFrame()

        if manufacturer and "Manufacturer" in on_order.columns:
            on_order = on_order[on_order["Manufacturer"].str.upper() == manufacturer.upper()]

        group_cols = []
        for col in ["Status", "Make", "Model", "Veh Type"]:
            if col in on_order.columns:
                group_cols.append(col)

        if not group_cols:
            return pd.DataFrame()

        agg_dict = {"Stock#": "count"} if "Stock#" in on_order.columns else {}
        if "Age" in on_order.columns:
            agg_dict["Age"] = "mean"
        if "Total Cost" in on_order.columns:
            agg_dict["Total Cost"] = "sum"
        if "Retail Price" in on_order.columns:
            agg_dict["Retail Price"] = ["sum", "mean"]

        if not agg_dict:
            return on_order.groupby(group_cols).size().reset_index(name="Units")

        details = on_order.groupby(group_cols).agg(agg_dict).reset_index()

        # Flatten columns
        flat_cols = list(group_cols)
        for col in details.columns[len(group_cols):]:
            if isinstance(col, tuple):
                if col[0] == "Stock#":
                    flat_cols.append("Units")
                elif col[0] == "Age":
                    flat_cols.append("Avg Age")
                elif col[0] == "Total Cost":
                    flat_cols.append("Total Cost")
                elif col[0] == "Retail Price" and col[1] == "sum":
                    flat_cols.append("Total Retail")
                elif col[0] == "Retail Price" and col[1] == "mean":
                    flat_cols.append("Avg Retail")
                else:
                    flat_cols.append(f"{col[0]}_{col[1]}")
            else:
                flat_cols.append(col)
        details.columns = flat_cols

        return details.sort_values("Units", ascending=False)

    def get_pipeline_flow_health(
        self,
        manufacturer: str = None,
        days: int = 30
    ) -> Dict:
        """
        Calculate pipeline flow health - ratio of receipts to orders.

        A healthy flow ratio is 0.8-1.2 (receiving roughly what you order).

        Returns dict with:
        - orders_placed: Units ordered in period
        - units_received: Units received (added to inventory) in period
        - flow_ratio: receipts / orders
        - health_status: "healthy", "backed_up", "over_receiving"
        """
        inventory = self.data_loader.load_current_inventory()
        on_order = self.data_loader.load_on_order()

        if manufacturer:
            if not inventory.empty and "Manufacturer" in inventory.columns:
                inventory = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]
            if not on_order.empty and "Manufacturer" in on_order.columns:
                on_order = on_order[on_order["Manufacturer"].str.upper() == manufacturer.upper()]

        # Estimate recent receipts from inventory age
        # Units with age <= days are recently received
        if "Age" in inventory.columns:
            recent_receipts = len(inventory[inventory["Age"] <= days])
        else:
            recent_receipts = 0

        # Estimate recent orders from on-order age
        # This is approximate since we may not have exact order dates
        if "Age" in on_order.columns and not on_order.empty:
            recent_orders = len(on_order[on_order["Age"] <= days])
        else:
            recent_orders = len(on_order) if not on_order.empty else 0

        # Calculate flow ratio
        if recent_orders > 0:
            flow_ratio = recent_receipts / recent_orders
        else:
            flow_ratio = 1.0  # No orders = neutral

        # Determine health status
        if 0.8 <= flow_ratio <= 1.2:
            health_status = "healthy"
        elif flow_ratio < 0.8:
            health_status = "backed_up"  # More orders than receipts
        else:
            health_status = "over_receiving"  # More receipts than orders

        return {
            "period_days": days,
            "orders_placed": recent_orders,
            "units_received": recent_receipts,
            "flow_ratio": round(flow_ratio, 2),
            "health_status": health_status,
            "total_pipeline": len(on_order) if not on_order.empty else 0,
            "interpretation": self._interpret_flow_health(flow_ratio, recent_orders, recent_receipts)
        }

    def _interpret_flow_health(
        self,
        flow_ratio: float,
        orders: int,
        receipts: int
    ) -> str:
        """Generate human-readable interpretation of pipeline flow."""
        if orders == 0 and receipts == 0:
            return "No recent activity - review ordering strategy"
        elif orders == 0:
            return f"Receiving {receipts} units with no new orders placed"
        elif receipts == 0:
            return f"Placed {orders} orders but no units received yet"
        elif flow_ratio < 0.5:
            return f"Pipeline backing up - orders ({orders}) significantly exceed receipts ({receipts})"
        elif flow_ratio < 0.8:
            return f"Slight backup - orders ({orders}) slightly exceed receipts ({receipts})"
        elif flow_ratio <= 1.2:
            return f"Healthy flow - orders ({orders}) and receipts ({receipts}) are balanced"
        elif flow_ratio <= 1.5:
            return f"Receiving slightly ahead of orders - receipts ({receipts}) vs orders ({orders})"
        else:
            return f"Rapid receipts - may need to slow ordering. Receipts ({receipts}) vs orders ({orders})"

    def get_pipeline_by_arrival_window(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """
        Get pipeline units grouped by estimated arrival window.

        Returns DataFrame with units expected by month.
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return pd.DataFrame()

        if manufacturer and "Manufacturer" in on_order.columns:
            on_order = on_order[on_order["Manufacturer"].str.upper() == manufacturer.upper()]

        on_order = on_order.copy()

        # Estimate arrival based on status and age
        now = datetime.now()

        def estimate_arrival(row):
            status = str(row.get("Status", "")).upper() if "Status" in row.index else ""
            age = row.get("Age", 0) if "Age" in row.index else 0

            # Days remaining based on status
            if "PO ISSUED" in status or "ORDERED" in status:
                days_left = max(0, self.config.total_lead_time - age)
            elif "PRODUCTION" in status or "BUILD" in status:
                days_left = self.config.transit_to_pdi_days + self.config.pdi_processing_days + self.config.transit_to_store_days
            elif "TRANSIT" in status:
                days_left = self.config.pdi_processing_days + self.config.transit_to_store_days
            elif "PDI" in status:
                days_left = self.config.transit_to_store_days
            elif "DISPATCH" in status:
                days_left = 7
            else:
                days_left = self.config.total_lead_time  # Default to full lead time

            arrival = now + timedelta(days=days_left)
            return arrival.strftime("%Y-%m")

        on_order["Est Arrival"] = on_order.apply(estimate_arrival, axis=1)

        # Group by arrival window
        arrival_summary = (
            on_order.groupby("Est Arrival")
            .agg({
                "Stock#": "count" if "Stock#" in on_order.columns else lambda x: len(x)
            })
            .reset_index()
        )
        arrival_summary.columns = ["Arrival Window", "Units"]

        # Add month name for readability
        arrival_summary["Month"] = pd.to_datetime(arrival_summary["Arrival Window"]).dt.strftime("%b %Y")

        return arrival_summary.sort_values("Arrival Window")

    # =========================================================================
    # REORDER NEEDS IDENTIFICATION
    # =========================================================================

    def identify_reorder_needs(
        self,
        manufacturer: str = None
    ) -> pd.DataFrame:
        """Identify segments that need reordering based on days supply."""
        days_supply_df = self.get_days_supply_by_segment(manufacturer)

        needs_reorder = days_supply_df[
            days_supply_df["Status"].isin(["Critical", "Low"])
        ].copy()

        if len(needs_reorder) == 0:
            return pd.DataFrame()

        # Calculate recommended order quantity
        needs_reorder["Units Needed"] = (
            (self.config.target_days_supply / 30 * needs_reorder["Monthly Velocity"]) -
            needs_reorder["Inventory"]
        ).clip(lower=0).round(0).astype(int)

        needs_reorder["Priority"] = needs_reorder["Status"].map({
            "Critical": 1,
            "Low": 2
        })

        return needs_reorder.sort_values(["Priority", "Days Supply"])
