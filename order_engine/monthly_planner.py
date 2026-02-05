"""
Monthly Planner Module
Breaks order recommendations into monthly distribution across the planning horizon.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from dateutil.relativedelta import relativedelta

from .config import Config, default_config
from .data_loader import DataLoader
from .pipeline_analyzer import PipelineAnalyzer
from .market_analyzer import MarketAnalyzer


@dataclass
class MonthlyBucket:
    """A single month's order recommendation."""
    month: str  # "Mar 2026"
    month_num: int  # 3
    year: int  # 2026

    # Demand
    projected_demand: float
    seasonality_index: int

    # Supply
    starting_inventory: int
    pipeline_arriving: int
    ending_inventory: int

    # Gap & Recommendation
    gap: int
    recommended_order: int

    # Cumulative
    cumulative_demand: float
    cumulative_ordered: int


class MonthlyPlanner:
    """Break order recommendations into monthly distribution."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)
        self.pipeline_analyzer = PipelineAnalyzer(data_loader=self.data_loader, config=self.config)

    def generate_monthly_plan(
        self,
        manufacturer: str,
        start_month: datetime = None,
        num_months: int = None,
        veh_type: str = None,
        make: str = None
    ) -> Dict:
        """
        Generate monthly order distribution plan.

        Args:
            manufacturer: Manufacturer to plan for
            start_month: First month of plan (default: next month)
            num_months: Number of months to plan (default: config.forecast_months)
            veh_type: Optional vehicle type filter
            make: Optional make filter

        Returns:
            Dict with monthly breakdown and totals
        """
        num_months = num_months or self.config.forecast_months
        start_month = start_month or (datetime.now() + relativedelta(months=1)).replace(day=1)

        # Get base data
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()
        on_order = self.data_loader.load_on_order()

        # Apply filters
        inv_mask = inventory["Manufacturer"].str.upper() == manufacturer.upper()
        retail_mask = retail["Manufacturer"].str.upper() == manufacturer.upper()

        if veh_type:
            inv_mask &= inventory["Veh Type"].str.upper() == veh_type.upper()
            retail_mask &= retail["Veh Type"].str.upper() == veh_type.upper()
        if make:
            inv_mask &= inventory["Make"].str.upper() == make.upper()
            retail_mask &= retail["Make"].str.upper() == make.upper()

        if "Status Category" in inventory.columns:
            inv_mask &= inventory["Status Category"] == "Sellable"

        filtered_inv = inventory[inv_mask]
        filtered_retail = retail[retail_mask]

        # Filter on-order
        pipe_mask = pd.Series(True, index=on_order.index) if not on_order.empty else pd.Series(dtype=bool)
        if not on_order.empty and "Manufacturer" in on_order.columns:
            pipe_mask = on_order["Manufacturer"].str.upper() == manufacturer.upper()
            if veh_type and "Veh Type" in on_order.columns:
                pipe_mask &= on_order["Veh Type"].str.upper() == veh_type.upper()
            if make and "Make" in on_order.columns:
                pipe_mask &= on_order["Make"].str.upper() == make.upper()
        filtered_pipeline = on_order[pipe_mask] if not on_order.empty else pd.DataFrame()

        # Calculate base monthly velocity
        lookback_days = self.config.lookback_months * 30
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        if "Sold Date" in filtered_retail.columns:
            recent_sales = filtered_retail[filtered_retail["Sold Date"] >= cutoff_date]
        else:
            recent_sales = filtered_retail

        base_monthly_velocity = len(recent_sales) / self.config.lookback_months if len(recent_sales) > 0 else 0

        # Get current inventory count
        current_inventory = len(filtered_inv)

        # Estimate pipeline arrivals by month
        pipeline_by_month = self._estimate_pipeline_arrivals(filtered_pipeline, start_month, num_months)

        # Build monthly buckets
        monthly_buckets = []
        running_inventory = current_inventory
        cumulative_demand = 0
        cumulative_ordered = 0

        for i in range(num_months):
            month_date = start_month + relativedelta(months=i)
            month_num = month_date.month
            month_str = month_date.strftime("%b %Y")

            # Seasonality adjustment
            seasonality_index = self.config.seasonality_index.get(month_num, 100)
            seasonality_factor = seasonality_index / 100.0

            # Projected demand for this month
            projected_demand = base_monthly_velocity * seasonality_factor

            # Pipeline arriving this month
            pipeline_arriving = pipeline_by_month.get(month_str, 0)

            # Starting inventory (from previous month's ending)
            starting_inventory = running_inventory

            # Calculate gap
            # We want to end the month with target days supply worth of inventory
            target_ending_inventory = (self.config.target_days_supply / 30) * projected_demand
            available_supply = starting_inventory + pipeline_arriving
            ending_inventory_if_no_order = max(0, available_supply - projected_demand)

            gap = max(0, int(target_ending_inventory - ending_inventory_if_no_order))

            # Recommended order (with safety buffer)
            recommended_order = int(gap * (1 + self.config.safety_buffer_pct))

            # Update running inventory for next month
            ending_inventory = int(available_supply - projected_demand + recommended_order)
            running_inventory = max(0, ending_inventory)

            # Update cumulative
            cumulative_demand += projected_demand
            cumulative_ordered += recommended_order

            bucket = MonthlyBucket(
                month=month_str,
                month_num=month_num,
                year=month_date.year,
                projected_demand=round(projected_demand, 1),
                seasonality_index=seasonality_index,
                starting_inventory=starting_inventory,
                pipeline_arriving=pipeline_arriving,
                ending_inventory=ending_inventory,
                gap=gap,
                recommended_order=recommended_order,
                cumulative_demand=round(cumulative_demand, 1),
                cumulative_ordered=cumulative_ordered
            )
            monthly_buckets.append(bucket)

        return {
            "manufacturer": manufacturer,
            "veh_type": veh_type,
            "make": make,
            "start_month": start_month.strftime("%b %Y"),
            "num_months": num_months,
            "base_monthly_velocity": round(base_monthly_velocity, 2),
            "current_inventory": current_inventory,
            "total_pipeline": len(filtered_pipeline),
            "monthly_buckets": [self._bucket_to_dict(b) for b in monthly_buckets],
            "totals": {
                "total_projected_demand": round(cumulative_demand, 1),
                "total_recommended_order": cumulative_ordered,
                "total_pipeline_arriving": sum(pipeline_by_month.values()),
                "avg_monthly_order": round(cumulative_ordered / num_months, 1) if num_months > 0 else 0
            }
        }

    def _estimate_pipeline_arrivals(
        self,
        pipeline: pd.DataFrame,
        start_month: datetime,
        num_months: int
    ) -> Dict[str, int]:
        """Estimate when pipeline units will arrive by month."""
        arrivals = {}

        if pipeline.empty:
            return arrivals

        now = datetime.now()

        # Status to days-until-arrival mapping
        status_days = {
            "ORDERED": self.config.total_lead_time,
            "PURCHASED": self.config.transit_to_pdi_days + self.config.pdi_processing_days + self.config.transit_to_store_days,
            "PO ISSUED": self.config.pdi_processing_days + self.config.transit_to_store_days,
            "YARD": self.config.transit_to_store_days,
            "PDI POOL": self.config.transit_to_store_days,
            "DISPATCHED": 7,
            "SHIPPED": self.config.pdi_processing_days + self.config.transit_to_store_days,
            "TRANSFER": 7,
        }

        for _, row in pipeline.iterrows():
            status = str(row.get("Status", "")).upper() if "Status" in row.index else ""
            age = row.get("Age", 0) if "Age" in row.index else 0

            # Estimate days until arrival
            days_remaining = self.config.total_lead_time  # Default
            for status_key, days in status_days.items():
                if status_key in status:
                    days_remaining = max(0, days - age) if "ORDERED" in status else days
                    break

            arrival_date = now + timedelta(days=days_remaining)
            arrival_month = arrival_date.strftime("%b %Y")

            # Only count if within planning horizon
            month_start = start_month
            month_end = start_month + relativedelta(months=num_months)
            if month_start <= arrival_date < month_end:
                arrivals[arrival_month] = arrivals.get(arrival_month, 0) + 1

        return arrivals

    def _bucket_to_dict(self, bucket: MonthlyBucket) -> Dict:
        """Convert MonthlyBucket to dict."""
        return {
            "month": bucket.month,
            "month_num": bucket.month_num,
            "year": bucket.year,
            "projected_demand": bucket.projected_demand,
            "seasonality_index": bucket.seasonality_index,
            "starting_inventory": bucket.starting_inventory,
            "pipeline_arriving": bucket.pipeline_arriving,
            "ending_inventory": bucket.ending_inventory,
            "gap": bucket.gap,
            "recommended_order": bucket.recommended_order,
            "cumulative_demand": bucket.cumulative_demand,
            "cumulative_ordered": bucket.cumulative_ordered
        }

    def _generate_zone_monthly_plan(
        self,
        manufacturer: str,
        zone: str,
        start_month: datetime,
        num_months: int
    ) -> Dict:
        """
        Generate monthly plan for a specific zone.

        Args:
            manufacturer: Manufacturer to plan for
            zone: Zone name to filter by
            start_month: First month of plan
            num_months: Number of months to plan

        Returns:
            Dict with monthly breakdown for the zone
        """
        # Get locations in this zone
        locations = self.data_loader.get_locations_in_zone(zone)
        if not locations:
            return {
                "manufacturer": manufacturer,
                "zone": zone,
                "error": f"No locations found in zone '{zone}'"
            }

        # Get base data
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()
        on_order = self.data_loader.load_on_order()

        # Filter inventory to manufacturer and zone locations
        inv_mask = (
            (inventory["Manufacturer"].str.upper() == manufacturer.upper()) &
            (inventory["PC"].isin(locations))
        )
        if "Status Category" in inventory.columns:
            inv_mask &= inventory["Status Category"] == "Sellable"
        filtered_inv = inventory[inv_mask]

        # Filter retail to manufacturer and zone locations
        retail_mask = (
            (retail["Manufacturer"].str.upper() == manufacturer.upper()) &
            (retail["PC"].isin(locations))
        )
        filtered_retail = retail[retail_mask]

        # Filter on-order to manufacturer and zone locations
        if not on_order.empty and "Manufacturer" in on_order.columns:
            pipe_mask = (
                (on_order["Manufacturer"].str.upper() == manufacturer.upper()) &
                (on_order["PC"].isin(locations))
            )
            filtered_pipeline = on_order[pipe_mask]
        else:
            filtered_pipeline = pd.DataFrame()

        # Calculate zone-specific monthly velocity
        lookback_days = self.config.lookback_months * 30
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        if "Sold Date" in filtered_retail.columns:
            recent_sales = filtered_retail[filtered_retail["Sold Date"] >= cutoff_date]
        else:
            recent_sales = filtered_retail

        base_monthly_velocity = len(recent_sales) / self.config.lookback_months if len(recent_sales) > 0 else 0

        # Get current inventory count
        current_inventory = len(filtered_inv)

        # Estimate pipeline arrivals by month
        pipeline_by_month = self._estimate_pipeline_arrivals(filtered_pipeline, start_month, num_months)

        # Build monthly buckets
        monthly_buckets = []
        running_inventory = current_inventory
        cumulative_demand = 0
        cumulative_ordered = 0

        for i in range(num_months):
            month_date = start_month + relativedelta(months=i)
            month_num = month_date.month
            month_str = month_date.strftime("%b %Y")

            # Seasonality adjustment
            seasonality_index = self.config.seasonality_index.get(month_num, 100)
            seasonality_factor = seasonality_index / 100.0

            # Projected demand for this month
            projected_demand = base_monthly_velocity * seasonality_factor

            # Pipeline arriving this month
            pipeline_arriving = pipeline_by_month.get(month_str, 0)

            # Starting inventory (from previous month's ending)
            starting_inventory = running_inventory

            # Calculate gap
            target_ending_inventory = (self.config.target_days_supply / 30) * projected_demand
            available_supply = starting_inventory + pipeline_arriving
            ending_inventory_if_no_order = max(0, available_supply - projected_demand)

            gap = max(0, int(target_ending_inventory - ending_inventory_if_no_order))

            # Recommended order (with safety buffer)
            recommended_order = int(gap * (1 + self.config.safety_buffer_pct))

            # Update running inventory for next month
            ending_inventory = int(available_supply - projected_demand + recommended_order)
            running_inventory = max(0, ending_inventory)

            # Update cumulative
            cumulative_demand += projected_demand
            cumulative_ordered += recommended_order

            bucket = MonthlyBucket(
                month=month_str,
                month_num=month_num,
                year=month_date.year,
                projected_demand=round(projected_demand, 1),
                seasonality_index=seasonality_index,
                starting_inventory=starting_inventory,
                pipeline_arriving=pipeline_arriving,
                ending_inventory=ending_inventory,
                gap=gap,
                recommended_order=recommended_order,
                cumulative_demand=round(cumulative_demand, 1),
                cumulative_ordered=cumulative_ordered
            )
            monthly_buckets.append(bucket)

        return {
            "manufacturer": manufacturer,
            "zone": zone,
            "locations": locations,
            "location_count": len(locations),
            "start_month": start_month.strftime("%b %Y"),
            "num_months": num_months,
            "base_monthly_velocity": round(base_monthly_velocity, 2),
            "current_inventory": current_inventory,
            "total_pipeline": len(filtered_pipeline),
            "monthly_buckets": [self._bucket_to_dict(b) for b in monthly_buckets],
            "totals": {
                "total_projected_demand": round(cumulative_demand, 1),
                "total_recommended_order": cumulative_ordered,
                "total_pipeline_arriving": sum(pipeline_by_month.values()),
                "avg_monthly_order": round(cumulative_ordered / num_months, 1) if num_months > 0 else 0
            }
        }

    def generate_monthly_plan_by_segment(
        self,
        manufacturer: str,
        segment_by: str = "veh_type",
        start_month: datetime = None,
        num_months: int = None
    ) -> Dict:
        """
        Generate monthly plans segmented by vehicle type, make, or zone.

        Args:
            manufacturer: Manufacturer to plan for
            segment_by: "veh_type", "make", or "zone"
            start_month: First month of plan
            num_months: Number of months

        Returns:
            Dict with monthly plans for each segment
        """
        num_months = num_months or self.config.forecast_months
        start_month = start_month or (datetime.now() + relativedelta(months=1)).replace(day=1)

        inventory = self.data_loader.load_current_inventory()
        mfg_inv = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]

        # Get segments
        if segment_by == "veh_type":
            segments = mfg_inv["Veh Type"].dropna().unique().tolist()
            segment_col = "veh_type"
        elif segment_by == "make":
            segments = mfg_inv["Make"].dropna().unique().tolist()
            segment_col = "make"
        else:
            # Zone - need to add zone from locations
            locations = self.data_loader.load_locations()
            if "PC" in locations.columns and "Zone" in locations.columns:
                zone_map = dict(zip(locations["PC"], locations["Zone"]))
                mfg_inv = mfg_inv.copy()
                mfg_inv["Zone"] = mfg_inv["PC"].map(zone_map)
                segments = mfg_inv["Zone"].dropna().unique().tolist()
                segment_col = "zone"
            else:
                segments = []
                segment_col = "zone"

        segment_plans = {}
        for segment in segments:
            if segment_by == "veh_type":
                plan = self.generate_monthly_plan(
                    manufacturer=manufacturer,
                    veh_type=segment,
                    start_month=start_month,
                    num_months=num_months
                )
            elif segment_by == "make":
                plan = self.generate_monthly_plan(
                    manufacturer=manufacturer,
                    make=segment,
                    start_month=start_month,
                    num_months=num_months
                )
            else:
                # Zone-level planning
                plan = self._generate_zone_monthly_plan(
                    manufacturer=manufacturer,
                    zone=segment,
                    start_month=start_month,
                    num_months=num_months
                )

            segment_plans[segment] = plan

        return {
            "manufacturer": manufacturer,
            "segment_by": segment_by,
            "start_month": start_month.strftime("%b %Y"),
            "num_months": num_months,
            "segments": segment_plans
        }

    def _generate_location_monthly_plan(
        self,
        manufacturer: str,
        location_pc: str,
        start_month: datetime,
        num_months: int
    ) -> Dict:
        """
        Generate monthly plan for a specific location.

        Args:
            manufacturer: Manufacturer to plan for
            location_pc: Location PC code
            start_month: First month of plan
            num_months: Number of months to plan

        Returns:
            Dict with monthly breakdown for the location
        """
        # Get base data
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()
        on_order = self.data_loader.load_on_order()

        # Get location info
        locations_df = self.data_loader.load_locations()
        loc_info = locations_df[locations_df["PC"] == location_pc]
        location_name = loc_info["Name"].iloc[0] if not loc_info.empty and "Name" in loc_info.columns else location_pc
        location_zone = loc_info["Zone"].iloc[0] if not loc_info.empty and "Zone" in loc_info.columns else "Unknown"

        # Filter inventory to manufacturer and location
        inv_mask = (
            (inventory["Manufacturer"].str.upper() == manufacturer.upper()) &
            (inventory["PC"] == location_pc)
        )
        if "Status Category" in inventory.columns:
            inv_mask &= inventory["Status Category"] == "Sellable"
        filtered_inv = inventory[inv_mask]

        # Filter retail to manufacturer and location
        retail_mask = (
            (retail["Manufacturer"].str.upper() == manufacturer.upper()) &
            (retail["PC"] == location_pc)
        )
        filtered_retail = retail[retail_mask]

        # Filter on-order to manufacturer and location
        if not on_order.empty and "Manufacturer" in on_order.columns:
            pipe_mask = (
                (on_order["Manufacturer"].str.upper() == manufacturer.upper()) &
                (on_order["PC"] == location_pc)
            )
            filtered_pipeline = on_order[pipe_mask]
        else:
            filtered_pipeline = pd.DataFrame()

        # Calculate location-specific monthly velocity
        lookback_days = self.config.lookback_months * 30
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        if "Sold Date" in filtered_retail.columns:
            recent_sales = filtered_retail[filtered_retail["Sold Date"] >= cutoff_date]
        else:
            recent_sales = filtered_retail

        base_monthly_velocity = len(recent_sales) / self.config.lookback_months if len(recent_sales) > 0 else 0

        # Get current inventory count
        current_inventory = len(filtered_inv)

        # Estimate pipeline arrivals by month
        pipeline_by_month = self._estimate_pipeline_arrivals(filtered_pipeline, start_month, num_months)

        # Build monthly buckets
        monthly_buckets = []
        running_inventory = current_inventory
        cumulative_demand = 0
        cumulative_ordered = 0

        for i in range(num_months):
            month_date = start_month + relativedelta(months=i)
            month_num = month_date.month
            month_str = month_date.strftime("%b %Y")

            # Seasonality adjustment
            seasonality_index = self.config.seasonality_index.get(month_num, 100)
            seasonality_factor = seasonality_index / 100.0

            # Projected demand for this month
            projected_demand = base_monthly_velocity * seasonality_factor

            # Pipeline arriving this month
            pipeline_arriving = pipeline_by_month.get(month_str, 0)

            # Starting inventory (from previous month's ending)
            starting_inventory = running_inventory

            # Calculate gap
            target_ending_inventory = (self.config.target_days_supply / 30) * projected_demand
            available_supply = starting_inventory + pipeline_arriving
            ending_inventory_if_no_order = max(0, available_supply - projected_demand)

            gap = max(0, int(target_ending_inventory - ending_inventory_if_no_order))

            # Recommended order (with safety buffer)
            recommended_order = int(gap * (1 + self.config.safety_buffer_pct))

            # Update running inventory for next month
            ending_inventory = int(available_supply - projected_demand + recommended_order)
            running_inventory = max(0, ending_inventory)

            # Update cumulative
            cumulative_demand += projected_demand
            cumulative_ordered += recommended_order

            bucket = MonthlyBucket(
                month=month_str,
                month_num=month_num,
                year=month_date.year,
                projected_demand=round(projected_demand, 1),
                seasonality_index=seasonality_index,
                starting_inventory=starting_inventory,
                pipeline_arriving=pipeline_arriving,
                ending_inventory=ending_inventory,
                gap=gap,
                recommended_order=recommended_order,
                cumulative_demand=round(cumulative_demand, 1),
                cumulative_ordered=cumulative_ordered
            )
            monthly_buckets.append(bucket)

        return {
            "manufacturer": manufacturer,
            "location_pc": location_pc,
            "location_name": location_name,
            "zone": location_zone,
            "start_month": start_month.strftime("%b %Y"),
            "num_months": num_months,
            "base_monthly_velocity": round(base_monthly_velocity, 2),
            "current_inventory": current_inventory,
            "total_pipeline": len(filtered_pipeline),
            "monthly_buckets": [self._bucket_to_dict(b) for b in monthly_buckets],
            "totals": {
                "total_projected_demand": round(cumulative_demand, 1),
                "total_recommended_order": cumulative_ordered,
                "total_pipeline_arriving": sum(pipeline_by_month.values()),
                "avg_monthly_order": round(cumulative_ordered / num_months, 1) if num_months > 0 else 0
            }
        }

    def generate_monthly_plan_by_location(
        self,
        manufacturer: str,
        start_month: datetime = None,
        num_months: int = None,
        zone: str = None
    ) -> Dict:
        """
        Generate monthly plans for each individual location.

        Args:
            manufacturer: Manufacturer to plan for
            start_month: First month of plan (default: next month)
            num_months: Number of months to plan (default: config.forecast_months)
            zone: Optional zone filter to limit locations

        Returns:
            Dict with location_plans: {location_pc: monthly_plan_dict}
        """
        num_months = num_months or self.config.forecast_months
        start_month = start_month or (datetime.now() + relativedelta(months=1)).replace(day=1)

        # Get locations to process
        if zone:
            location_pcs = self.data_loader.get_locations_in_zone(zone)
        else:
            locations_df = self.data_loader.load_locations()
            location_pcs = locations_df["PC"].tolist() if "PC" in locations_df.columns else []

        # Get inventory to filter to only locations with this manufacturer's inventory
        inventory = self.data_loader.load_current_inventory()
        mfg_inv = inventory[inventory["Manufacturer"].str.upper() == manufacturer.upper()]
        locations_with_inventory = mfg_inv["PC"].unique().tolist()

        # Limit to locations that have inventory for this manufacturer
        location_pcs = [pc for pc in location_pcs if pc in locations_with_inventory]

        # Generate plans for each location
        location_plans = {}
        for pc in location_pcs:
            plan = self._generate_location_monthly_plan(
                manufacturer=manufacturer,
                location_pc=pc,
                start_month=start_month,
                num_months=num_months
            )
            location_plans[pc] = plan

        # Calculate aggregate totals
        total_inventory = sum(p.get("current_inventory", 0) for p in location_plans.values())
        total_pipeline = sum(p.get("total_pipeline", 0) for p in location_plans.values())
        total_recommended = sum(p.get("totals", {}).get("total_recommended_order", 0) for p in location_plans.values())
        total_demand = sum(p.get("totals", {}).get("total_projected_demand", 0) for p in location_plans.values())

        return {
            "manufacturer": manufacturer,
            "zone": zone,
            "start_month": start_month.strftime("%b %Y"),
            "num_months": num_months,
            "location_count": len(location_plans),
            "location_plans": location_plans,
            "aggregate_totals": {
                "total_current_inventory": total_inventory,
                "total_pipeline": total_pipeline,
                "total_projected_demand": round(total_demand, 1),
                "total_recommended_order": total_recommended
            }
        }

    def to_dataframe(self, monthly_plan: Dict) -> pd.DataFrame:
        """Convert monthly plan to DataFrame for easy viewing/export."""
        buckets = monthly_plan.get("monthly_buckets", [])
        if not buckets:
            return pd.DataFrame()

        df = pd.DataFrame(buckets)

        # Rename columns for readability
        df = df.rename(columns={
            "month": "Month",
            "projected_demand": "Projected Demand",
            "seasonality_index": "Seasonality",
            "starting_inventory": "Starting Inv",
            "pipeline_arriving": "Pipeline Arriving",
            "gap": "Gap",
            "recommended_order": "Recommended Order",
            "ending_inventory": "Ending Inv",
            "cumulative_demand": "Cumul. Demand",
            "cumulative_ordered": "Cumul. Ordered"
        })

        # Select and order columns
        columns = [
            "Month", "Seasonality", "Projected Demand", "Starting Inv",
            "Pipeline Arriving", "Gap", "Recommended Order", "Ending Inv"
        ]
        df = df[[c for c in columns if c in df.columns]]

        return df
