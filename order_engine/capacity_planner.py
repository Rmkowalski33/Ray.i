"""
Capacity-Based Order Planner Module
====================================

Top-down approach to inventory planning:
1. Start with lot capacity (what we CAN hold)
2. Account for current inventory, pipeline, and in-service units
3. Calculate gaps by Type → Zone → Location
4. Identify which licensed brands can fill each gap

This is the REVERSE of manufacturer-centric planning:
- Instead of "what should we order from Forest River?"
- Ask "what do we need, and who can supply it?"

Key Data Points:
- Lot Capacity: "NEW" column from Locations table
- Current Inventory: Inventory Extract (NEW units only)
- Pipeline: On Order units by status
- In Service: Units temporarily out for PDI/repairs (will return)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .config import Config, default_config
from .data_loader import DataLoader
from .brand_licensing import BrandLicensingAnalyzer


@dataclass
class LocationCapacity:
    """Capacity analysis for a single location."""
    location: str
    zone: str
    lot_capacity: int  # Total NEW spots available
    current_inventory: int  # NEW units on ground
    on_order_pipeline: int  # Units arriving (all pipeline statuses)
    in_service: int  # Units temporarily out (PDI, repairs)
    available_position: int  # Current + Pipeline - InService
    gap: int  # Capacity - Available Position
    gap_by_type: Dict[str, int] = field(default_factory=dict)  # Gap broken down by TT/FW/MH
    seasonality_adjusted_gap: int = 0  # Gap adjusted for current season
    licensed_brands: List[str] = field(default_factory=list)  # Brands licensed at this location


@dataclass
class TypeNeed:
    """Company-wide need for a vehicle type."""
    veh_type: str
    total_capacity: int
    total_current: int
    total_pipeline: int
    total_in_service: int
    total_gap: int
    by_zone: Dict[str, Dict] = field(default_factory=dict)
    by_location: Dict[str, Dict] = field(default_factory=dict)


class CapacityPlanner:
    """
    Capacity-based inventory planner.

    Works in REVERSE of manufacturer planning:
    1. What do we NEED company-wide?
    2. Where do we need it (zone/location)?
    3. What's available to fill it (licensed brands)?
    """

    # Status classifications based on Unit Status reference table
    # These match the "Status Name" column in FTRV Index & References

    # Sellable on-lot inventory (counts toward current inventory)
    SELLABLE_STATUSES = [
        "RVASAP (CAMP READY)", "RVASAP", "CAMP READY",
        "READY FOR SALE",
        "SHOWROOM",
        "HAIL DAMAGE"
    ]

    # Pipeline statuses (on order, will arrive)
    PIPELINE_STATUSES = [
        "ORDERED",           # Unit ordered, not shipped
        "PURCHASED",         # PO issued, floored
        "SHIPPED",           # In transit from factory
        "DRIVER NEEDED",     # Await driver assignment
        "OPS TRANSFER",      # Operational transfer (PDI, service)
        "STORE-TO-STORE TRANSFER",  # Retail balancing
        "RET TRANSFER"       # Retail transfer
    ]

    # In service statuses (temporarily unavailable, will return to sellable)
    IN_SERVICE_STATUSES = [
        "IN SERVICE",        # Repair/refurbishment, open RO
        "PRE-PDI",           # Await inspection
        "QC NEEDED",         # Post-PDI validation
        "DAMAGED (DEALER)"   # Dealer-caused damage repair
    ]

    # Hold/pending statuses (not immediately sellable)
    HOLD_STATUSES = [
        "FLEET PENDING",     # Hold for fleet/corporate deal
        "SALE PENDING",      # Deal in progress
        "RV SHOW UNIT",      # Show inventory
        "RV SHOW BACKUP",    # Backup for show
        "FACTORY REVIEW"     # Manufacturer damage review
    ]

    # Exit statuses (no longer in inventory)
    EXIT_STATUSES = [
        "SOLD",
        "WHOLESALE",
        "BUYBACK",
        "DEAL VOIDED"
    ]

    # Non-inventory statuses
    NON_INVENTORY_STATUSES = [
        "TRADE-IN"  # Inbound trade, not new inventory
    ]

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.data_loader = DataLoader(config=self.config)
        self.brand_licensing = BrandLicensingAnalyzer(
            data_loader=self.data_loader,
            config=self.config
        )

    def get_location_capacities(self) -> pd.DataFrame:
        """
        Load location lot capacities from reference data.

        Returns DataFrame with:
        - Location/PC
        - Zone
        - NEW capacity (lot spots for new inventory)
        """
        locations = self.data_loader.load_locations()

        # Ensure we have the NEW capacity column
        if "NEW" not in locations.columns:
            # Try alternate column names
            for alt in ["New", "NEW CAPACITY", "New Capacity", "Lot Capacity"]:
                if alt in locations.columns:
                    locations = locations.rename(columns={alt: "NEW"})
                    break

        if "NEW" not in locations.columns:
            # Create placeholder if not found
            locations["NEW"] = 0

        # Clean up
        locations["NEW"] = pd.to_numeric(locations["NEW"], errors="coerce").fillna(0).astype(int)

        return locations

    def get_current_inventory_by_location(self) -> pd.DataFrame:
        """
        Get current NEW inventory counts by location and type.

        Note: Inventory Extract contains NEW units only.
        Filters to sellable statuses (RVASAP, Ready for Sale, Showroom, etc.)
        """
        inventory = self.data_loader.load_current_inventory()

        # Filter to sellable status using our defined list
        if "Status" in inventory.columns:
            sellable_upper = [s.upper() for s in self.SELLABLE_STATUSES]
            # Use contains for partial matching (e.g., "RVASAP" matches "RVASAP (CAMP READY)")
            sellable_mask = inventory["Status"].str.upper().apply(
                lambda x: any(s in str(x) for s in sellable_upper) if pd.notna(x) else False
            )
            inventory = inventory[sellable_mask]
        elif "Status Category" in inventory.columns:
            inventory = inventory[inventory["Status Category"] == "Sellable"]

        # Group by location and type
        group_cols = ["PC"]
        if "Veh Type" in inventory.columns:
            group_cols.append("Veh Type")

        counts = inventory.groupby(group_cols).size().reset_index(name="Current_Inventory")

        return counts

    def get_pipeline_by_location(self) -> pd.DataFrame:
        """
        Get on-order/pipeline units by location and type.

        Includes all pipeline statuses that represent incoming inventory.
        Note: Uses destination PC for units in transit.
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return pd.DataFrame(columns=["PC", "Veh Type", "Pipeline_Units"])

        # Filter to pipeline statuses
        if "Status" in on_order.columns:
            pipeline_upper = [s.upper() for s in self.PIPELINE_STATUSES]
            pipeline_mask = on_order["Status"].str.upper().apply(
                lambda x: any(s in str(x) for s in pipeline_upper) if pd.notna(x) else False
            )
            on_order = on_order[pipeline_mask]

        # Group by location and type
        group_cols = ["PC"] if "PC" in on_order.columns else []
        if "Veh Type" in on_order.columns:
            group_cols.append("Veh Type")

        if not group_cols:
            return pd.DataFrame(columns=["PC", "Veh Type", "Pipeline_Units"])

        counts = on_order.groupby(group_cols).size().reset_index(name="Pipeline_Units")

        return counts

    def get_hold_units_by_location(self) -> pd.DataFrame:
        """
        Get units on hold (fleet pending, sale pending, show units, etc.)

        These units are on-lot but not immediately sellable.
        """
        inventory = self.data_loader.load_current_inventory()

        if "Status" not in inventory.columns:
            return pd.DataFrame(columns=["PC", "Veh Type", "Hold_Units"])

        # Filter to hold statuses
        hold_upper = [s.upper() for s in self.HOLD_STATUSES]
        hold_mask = inventory["Status"].str.upper().apply(
            lambda x: any(s in str(x) for s in hold_upper) if pd.notna(x) else False
        )
        hold_units = inventory[hold_mask]

        if hold_units.empty:
            return pd.DataFrame(columns=["PC", "Veh Type", "Hold_Units"])

        # Group by location and type
        group_cols = ["PC"]
        if "Veh Type" in hold_units.columns:
            group_cols.append("Veh Type")

        counts = hold_units.groupby(group_cols).size().reset_index(name="Hold_Units")

        return counts

    def get_in_service_by_location(self) -> pd.DataFrame:
        """
        Get units currently in service (will return to available inventory).

        These are units temporarily out for PDI, repairs, detailing, etc.
        """
        inventory = self.data_loader.load_current_inventory()

        if "Status" not in inventory.columns:
            return pd.DataFrame(columns=["PC", "Veh Type", "In_Service_Units"])

        # Filter to in-service statuses
        in_service_mask = inventory["Status"].str.upper().isin(
            [s.upper() for s in self.IN_SERVICE_STATUSES]
        )
        in_service = inventory[in_service_mask]

        if in_service.empty:
            return pd.DataFrame(columns=["PC", "Veh Type", "In_Service_Units"])

        # Group by location and type
        group_cols = ["PC"]
        if "Veh Type" in in_service.columns:
            group_cols.append("Veh Type")

        counts = in_service.groupby(group_cols).size().reset_index(name="In_Service_Units")

        return counts

    def calculate_company_needs(self) -> Dict[str, TypeNeed]:
        """
        Calculate company-wide inventory needs by vehicle type.

        Returns dict mapping vehicle type to TypeNeed dataclass.
        """
        # Load all data
        locations = self.get_location_capacities()
        current_inv = self.get_current_inventory_by_location()
        pipeline = self.get_pipeline_by_location()
        in_service = self.get_in_service_by_location()

        # Get vehicle types from inventory
        veh_types = self.data_loader.get_veh_types()

        results = {}

        for veh_type in veh_types:
            # Filter to this vehicle type
            vt_current = current_inv[current_inv.get("Veh Type", pd.Series()) == veh_type] if "Veh Type" in current_inv.columns else current_inv
            vt_pipeline = pipeline[pipeline.get("Veh Type", pd.Series()) == veh_type] if "Veh Type" in pipeline.columns else pipeline
            vt_in_service = in_service[in_service.get("Veh Type", pd.Series()) == veh_type] if "Veh Type" in in_service.columns else in_service

            # Calculate totals
            total_current = vt_current["Current_Inventory"].sum() if not vt_current.empty else 0
            total_pipeline = vt_pipeline["Pipeline_Units"].sum() if not vt_pipeline.empty else 0
            total_in_service = vt_in_service["In_Service_Units"].sum() if not vt_in_service.empty else 0

            # For capacity, we need to allocate lot space by type
            # Use current inventory mix as proxy for type allocation
            total_inv = current_inv["Current_Inventory"].sum() if not current_inv.empty else 1
            type_pct = total_current / total_inv if total_inv > 0 else 0.33

            total_capacity = int(locations["NEW"].sum() * type_pct)

            # Calculate gap
            available_position = total_current + total_pipeline - total_in_service
            total_gap = max(0, total_capacity - available_position)

            # Calculate by zone
            by_zone = {}
            if "Zone" in locations.columns:
                for zone in locations["Zone"].dropna().unique():
                    zone_locs = locations[locations["Zone"] == zone]["PC"].tolist()
                    zone_capacity = int(locations[locations["Zone"] == zone]["NEW"].sum() * type_pct)
                    zone_current = vt_current[vt_current["PC"].isin(zone_locs)]["Current_Inventory"].sum() if not vt_current.empty and "PC" in vt_current.columns else 0
                    zone_pipeline = vt_pipeline[vt_pipeline["PC"].isin(zone_locs)]["Pipeline_Units"].sum() if not vt_pipeline.empty and "PC" in vt_pipeline.columns else 0
                    zone_in_service = vt_in_service[vt_in_service["PC"].isin(zone_locs)]["In_Service_Units"].sum() if not vt_in_service.empty and "PC" in vt_in_service.columns else 0

                    zone_available = zone_current + zone_pipeline - zone_in_service
                    zone_gap = max(0, zone_capacity - zone_available)

                    by_zone[zone] = {
                        "capacity": zone_capacity,
                        "current": int(zone_current),
                        "pipeline": int(zone_pipeline),
                        "in_service": int(zone_in_service),
                        "available": int(zone_available),
                        "gap": int(zone_gap)
                    }

            # Calculate by location
            by_location = {}
            for _, loc_row in locations.iterrows():
                pc = loc_row.get("PC", "")
                if not pc:
                    continue

                loc_capacity = int(loc_row.get("NEW", 0) * type_pct)
                loc_current = vt_current[vt_current["PC"] == pc]["Current_Inventory"].sum() if not vt_current.empty and "PC" in vt_current.columns else 0
                loc_pipeline = vt_pipeline[vt_pipeline["PC"] == pc]["Pipeline_Units"].sum() if not vt_pipeline.empty and "PC" in vt_pipeline.columns else 0
                loc_in_service = vt_in_service[vt_in_service["PC"] == pc]["In_Service_Units"].sum() if not vt_in_service.empty and "PC" in vt_in_service.columns else 0

                loc_available = loc_current + loc_pipeline - loc_in_service
                loc_gap = max(0, loc_capacity - loc_available)

                # Get licensed brands for this location and type
                licensed_brands = self._get_licensed_brands_for_type(pc, veh_type)

                by_location[pc] = {
                    "zone": loc_row.get("Zone", ""),
                    "capacity": loc_capacity,
                    "current": int(loc_current),
                    "pipeline": int(loc_pipeline),
                    "in_service": int(loc_in_service),
                    "available": int(loc_available),
                    "gap": int(loc_gap),
                    "licensed_brands": licensed_brands
                }

            results[veh_type] = TypeNeed(
                veh_type=veh_type,
                total_capacity=total_capacity,
                total_current=int(total_current),
                total_pipeline=int(total_pipeline),
                total_in_service=int(total_in_service),
                total_gap=total_gap,
                by_zone=by_zone,
                by_location=by_location
            )

        return results

    def _get_licensed_brands_for_type(self, location: str, veh_type: str) -> List[str]:
        """Get brands licensed at a location for a specific vehicle type."""
        try:
            matrix = self.brand_licensing.get_licensing_matrix(veh_type=veh_type)
            if matrix.empty or location not in matrix.columns:
                return []

            licensed_mask = matrix[location].str.upper().str.contains("LICENSED", na=False)
            return matrix.loc[licensed_mask, "Make"].dropna().unique().tolist()
        except Exception:
            return []

    def generate_capacity_report(self) -> Dict:
        """
        Generate comprehensive capacity-based order planning report.

        Returns dict with:
        - Company-wide summary by type
        - Zone-level breakdown
        - Location-level gaps with licensed brand options
        - Prioritized recommendations
        """
        needs = self.calculate_company_needs()

        # Build report structure
        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_lot_capacity": 0,
                "total_current_inventory": 0,
                "total_pipeline": 0,
                "total_in_service": 0,
                "total_gap": 0
            },
            "by_type": {},
            "by_zone": {},
            "location_gaps": [],
            "fill_recommendations": []
        }

        # Aggregate summary
        for veh_type, need in needs.items():
            report["summary"]["total_lot_capacity"] += need.total_capacity
            report["summary"]["total_current_inventory"] += need.total_current
            report["summary"]["total_pipeline"] += need.total_pipeline
            report["summary"]["total_in_service"] += need.total_in_service
            report["summary"]["total_gap"] += need.total_gap

            report["by_type"][veh_type] = {
                "capacity": need.total_capacity,
                "current": need.total_current,
                "pipeline": need.total_pipeline,
                "in_service": need.total_in_service,
                "gap": need.total_gap,
                "fill_pct": round((need.total_current + need.total_pipeline) / need.total_capacity * 100, 1) if need.total_capacity > 0 else 0
            }

            # Aggregate zones
            for zone, zone_data in need.by_zone.items():
                if zone not in report["by_zone"]:
                    report["by_zone"][zone] = {
                        "capacity": 0, "current": 0, "pipeline": 0,
                        "in_service": 0, "gap": 0, "by_type": {}
                    }
                report["by_zone"][zone]["capacity"] += zone_data["capacity"]
                report["by_zone"][zone]["current"] += zone_data["current"]
                report["by_zone"][zone]["pipeline"] += zone_data["pipeline"]
                report["by_zone"][zone]["in_service"] += zone_data["in_service"]
                report["by_zone"][zone]["gap"] += zone_data["gap"]
                report["by_zone"][zone]["by_type"][veh_type] = zone_data

            # Collect location gaps
            for location, loc_data in need.by_location.items():
                if loc_data["gap"] > 0:
                    report["location_gaps"].append({
                        "location": location,
                        "zone": loc_data["zone"],
                        "veh_type": veh_type,
                        "capacity": loc_data["capacity"],
                        "current": loc_data["current"],
                        "pipeline": loc_data["pipeline"],
                        "in_service": loc_data["in_service"],
                        "gap": loc_data["gap"],
                        "licensed_brands": loc_data["licensed_brands"]
                    })

        # Sort location gaps by gap size (largest first)
        report["location_gaps"] = sorted(
            report["location_gaps"],
            key=lambda x: x["gap"],
            reverse=True
        )

        # Generate fill recommendations
        report["fill_recommendations"] = self._generate_fill_recommendations(
            report["location_gaps"]
        )

        # Add unassigned pipeline analysis
        report["unassigned_pipeline"] = self.get_unassigned_pipeline_summary()

        # Add assignment recommendations (summary level - kept for compatibility)
        report["assignment_recommendations"] = self.generate_assignment_recommendations()
        report["assignments_by_location"] = self.get_assignment_summary_by_location()

        # Add VIN-level detail
        report["inventory_detail"] = self.get_current_inventory_detail()
        report["on_order_detail"] = self.get_on_order_detail()
        report["unassigned_detail"] = self.get_unassigned_pipeline_detail()

        # Add VIN-level assignment recommendations
        gap_fill, overflow = self.generate_vin_level_assignments()
        report["vin_assignments_gap_fill"] = gap_fill
        report["vin_assignments_overflow"] = overflow

        # Add inventory flow projection
        report["inventory_flow"] = self.generate_inventory_flow_projection(months=6)

        return report

    def _generate_fill_recommendations(self, location_gaps: List[Dict]) -> List[Dict]:
        """
        Generate prioritized recommendations for filling gaps.

        Groups gaps by zone/type and suggests brands to fill.
        """
        recommendations = []

        # Group gaps by zone and type
        zone_type_gaps = {}
        for gap in location_gaps:
            key = (gap["zone"], gap["veh_type"])
            if key not in zone_type_gaps:
                zone_type_gaps[key] = {
                    "zone": gap["zone"],
                    "veh_type": gap["veh_type"],
                    "total_gap": 0,
                    "locations": [],
                    "all_licensed_brands": set()
                }
            zone_type_gaps[key]["total_gap"] += gap["gap"]
            zone_type_gaps[key]["locations"].append(gap["location"])
            zone_type_gaps[key]["all_licensed_brands"].update(gap["licensed_brands"])

        # Convert to recommendations
        for (zone, veh_type), data in sorted(
            zone_type_gaps.items(),
            key=lambda x: x[1]["total_gap"],
            reverse=True
        ):
            recommendations.append({
                "zone": zone,
                "veh_type": veh_type,
                "total_units_needed": data["total_gap"],
                "locations_with_gaps": len(data["locations"]),
                "location_list": data["locations"],
                "available_brands": sorted(data["all_licensed_brands"]),
                "recommendation": f"Need {data['total_gap']} {veh_type} units across {len(data['locations'])} locations in {zone}"
            })

        return recommendations

    def get_unassigned_pipeline(self) -> pd.DataFrame:
        """
        Get pipeline units that are unassigned (at PDI/YARD/CORP locations).

        These units are ready or nearly ready but haven't been assigned
        to a retail location yet.
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return pd.DataFrame()

        # Non-retail location codes (PDI centers, corporate, etc.)
        non_retail_pcs = ["CORP", "ELK", "ELK2", "MSC", "99", "96", "75", "72", "80", "81", "15"]

        # Also check for YARD/PDI POOL in status
        if "Status" in on_order.columns and "PC" in on_order.columns:
            unassigned_mask = (
                on_order["PC"].isin(non_retail_pcs) |
                on_order["PC"].isna() |
                on_order["Status"].str.upper().str.contains("YARD|PDI|POOL", na=False)
            )
            unassigned = on_order[unassigned_mask].copy()
        elif "PC" in on_order.columns:
            unassigned = on_order[on_order["PC"].isin(non_retail_pcs) | on_order["PC"].isna()].copy()
        else:
            return pd.DataFrame()

        return unassigned

    def get_unassigned_pipeline_summary(self) -> Dict:
        """
        Get summary of unassigned pipeline units by type, make, and status.
        """
        unassigned = self.get_unassigned_pipeline()

        if unassigned.empty:
            return {
                "total_unassigned": 0,
                "by_type": {},
                "by_make": {},
                "by_status": {},
                "by_manufacturer": {}
            }

        summary = {
            "total_unassigned": len(unassigned),
            "by_type": {},
            "by_make": {},
            "by_status": {},
            "by_manufacturer": {}
        }

        # By vehicle type
        if "Veh Type" in unassigned.columns:
            summary["by_type"] = unassigned.groupby("Veh Type").size().to_dict()

        # By make
        if "Make" in unassigned.columns:
            make_counts = unassigned.groupby("Make").size().sort_values(ascending=False)
            summary["by_make"] = make_counts.head(20).to_dict()

        # By status
        if "Status" in unassigned.columns:
            summary["by_status"] = unassigned.groupby("Status").size().to_dict()

        # By manufacturer
        if "Manufacturer" in unassigned.columns:
            summary["by_manufacturer"] = unassigned.groupby("Manufacturer").size().to_dict()

        return summary

    def generate_assignment_recommendations(self) -> List[Dict]:
        """
        Generate recommendations for assigning unassigned pipeline units to locations.

        Matches unassigned units to location gaps based on:
        - Vehicle type match
        - Brand licensing at destination
        - Gap priority (largest gaps first)
        """
        # Get unassigned pipeline
        unassigned = self.get_unassigned_pipeline()
        if unassigned.empty:
            return []

        # Get location gaps
        needs = self.calculate_company_needs()

        # Build list of gaps sorted by size
        gaps = []
        for veh_type, need in needs.items():
            for location, loc_data in need.by_location.items():
                if loc_data["gap"] > 0:
                    gaps.append({
                        "location": location,
                        "zone": loc_data["zone"],
                        "veh_type": veh_type,
                        "gap": loc_data["gap"],
                        "licensed_brands": set(loc_data.get("licensed_brands", []))
                    })

        # Sort gaps by size (largest first)
        gaps = sorted(gaps, key=lambda x: x["gap"], reverse=True)

        # Match unassigned units to gaps
        recommendations = []

        for _, unit in unassigned.iterrows():
            unit_type = unit.get("Veh Type", "")
            unit_make = unit.get("Make", "")
            unit_mfg = unit.get("Manufacturer", "")
            unit_status = unit.get("Status", "")
            stock_num = unit.get("Stock#", "")

            # Find best matching gap
            best_match = None
            for gap in gaps:
                if gap["veh_type"] != unit_type:
                    continue
                if gap["gap"] <= 0:
                    continue
                # Check if make is licensed at location
                if gap["licensed_brands"] and unit_make and unit_make.upper() not in [b.upper() for b in gap["licensed_brands"]]:
                    continue

                best_match = gap
                break

            if best_match:
                recommendations.append({
                    "stock_num": stock_num,
                    "veh_type": unit_type,
                    "manufacturer": unit_mfg,
                    "make": unit_make,
                    "current_status": unit_status,
                    "recommended_location": best_match["location"],
                    "recommended_zone": best_match["zone"],
                    "location_gap": best_match["gap"],
                    "reason": f"Fill {unit_type} gap at {best_match['location']} ({best_match['zone']})"
                })
                # Decrement the gap
                best_match["gap"] -= 1

        return recommendations

    def get_assignment_summary_by_location(self) -> Dict[str, Dict]:
        """
        Summarize assignment recommendations by destination location.
        """
        recommendations = self.generate_assignment_recommendations()

        by_location = {}
        for rec in recommendations:
            loc = rec["recommended_location"]
            if loc not in by_location:
                by_location[loc] = {
                    "zone": rec["recommended_zone"],
                    "total_units": 0,
                    "by_type": {},
                    "by_make": {},
                    "units": []
                }

            by_location[loc]["total_units"] += 1

            vt = rec["veh_type"]
            by_location[loc]["by_type"][vt] = by_location[loc]["by_type"].get(vt, 0) + 1

            make = rec["make"]
            by_location[loc]["by_make"][make] = by_location[loc]["by_make"].get(make, 0) + 1

            by_location[loc]["units"].append(rec)

        return by_location

    def get_unassigned_pipeline_detail(self) -> pd.DataFrame:
        """
        Get detailed unassigned pipeline with VIN, Model, Stock# for unit-level tracking.
        """
        unassigned = self.get_unassigned_pipeline()

        if unassigned.empty:
            return pd.DataFrame()

        # Select key columns for detail view
        detail_cols = [
            "Stock#", "VIN", "Yr", "Manufacturer", "Make", "Model",
            "Veh Type", "Status", "Status Days", "Age", "PC",
            "Retail Price", "Total Cost", "Purch. Date"
        ]
        available_cols = [c for c in detail_cols if c in unassigned.columns]

        return unassigned[available_cols].copy()

    def get_current_inventory_detail(self) -> pd.DataFrame:
        """
        Get detailed current inventory with VIN, Model, Stock# for unit-level tracking.
        """
        inventory = self.data_loader.load_current_inventory()

        # Select key columns for detail view
        detail_cols = [
            "Stock#", "VIN", "Yr", "Manufacturer", "Make", "Model",
            "Veh Type", "Status", "Status Days", "Age", "PC Age",
            "PC", "Lot Location", "Retail Price", "Total Cost",
            "Purch. Date", "Status Category"
        ]
        available_cols = [c for c in detail_cols if c in inventory.columns]

        return inventory[available_cols].copy()

    def get_on_order_detail(self) -> pd.DataFrame:
        """
        Get detailed on-order units with VIN, Model, Stock# for unit-level tracking.
        """
        on_order = self.data_loader.load_on_order()

        if on_order.empty:
            return pd.DataFrame()

        # Select key columns for detail view
        detail_cols = [
            "Stock#", "VIN", "Yr", "Manufacturer", "Make", "Model",
            "Veh Type", "Status", "Status Days", "Age", "PC",
            "Retail Price", "Total Cost", "Purch. Date"
        ]
        available_cols = [c for c in detail_cols if c in on_order.columns]

        return on_order[available_cols].copy()

    def generate_vin_level_assignments(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate VIN-specific assignment recommendations.

        Returns:
            Tuple of (gap_filling_assignments, remaining_yard_allocations)
            - gap_filling_assignments: VINs assigned to fill location gaps
            - remaining_yard_allocations: VINs that couldn't fill gaps, suggested allocation
        """
        # Get unassigned pipeline with full detail
        unassigned = self.get_unassigned_pipeline()
        if unassigned.empty:
            return [], []

        # Get location gaps
        needs = self.calculate_company_needs()

        # Build list of gaps sorted by size
        gaps = []
        for veh_type, need in needs.items():
            for location, loc_data in need.by_location.items():
                if loc_data["gap"] > 0:
                    gaps.append({
                        "location": location,
                        "zone": loc_data["zone"],
                        "veh_type": veh_type,
                        "gap": loc_data["gap"],
                        "original_gap": loc_data["gap"],
                        "licensed_brands": set(b.upper() for b in loc_data.get("licensed_brands", []))
                    })

        # Sort gaps by size (largest first)
        gaps = sorted(gaps, key=lambda x: x["gap"], reverse=True)

        gap_filling = []
        remaining = []

        for _, unit in unassigned.iterrows():
            unit_type = str(unit.get("Veh Type", "")).upper()
            unit_make = str(unit.get("Make", "")).upper()
            unit_mfg = str(unit.get("Manufacturer", ""))
            unit_model = str(unit.get("Model", ""))
            unit_status = str(unit.get("Status", ""))
            stock_num = unit.get("Stock#", "")
            vin = str(unit.get("VIN", ""))
            year = unit.get("Yr", "")
            retail_price = unit.get("Retail Price", 0)
            total_cost = unit.get("Total Cost", 0)

            unit_detail = {
                "stock_num": stock_num,
                "vin": vin,
                "year": year,
                "manufacturer": unit_mfg,
                "make": unit_make,
                "model": unit_model,
                "veh_type": unit_type,
                "current_status": unit_status,
                "retail_price": retail_price,
                "total_cost": total_cost
            }

            # Find best matching gap
            best_match = None
            for gap in gaps:
                if gap["veh_type"].upper() != unit_type:
                    continue
                if gap["gap"] <= 0:
                    continue
                # Check if make is licensed at location
                if gap["licensed_brands"] and unit_make and unit_make not in gap["licensed_brands"]:
                    continue

                best_match = gap
                break

            if best_match:
                assignment = {
                    **unit_detail,
                    "assigned_location": best_match["location"],
                    "assigned_zone": best_match["zone"],
                    "fills_gap_of": best_match["original_gap"],
                    "remaining_gap_after": best_match["gap"] - 1,
                    "assignment_type": "GAP_FILL",
                    "reason": f"Fill {unit_type} gap at {best_match['location']}"
                }
                gap_filling.append(assignment)
                best_match["gap"] -= 1
            else:
                # No gap to fill - suggest distribution based on velocity/capacity
                suggested_loc = self._suggest_overflow_location(unit_type, unit_make)
                allocation = {
                    **unit_detail,
                    "suggested_location": suggested_loc.get("location", "TBD"),
                    "suggested_zone": suggested_loc.get("zone", ""),
                    "assignment_type": "OVERFLOW",
                    "reason": suggested_loc.get("reason", "No matching gap - distribute based on capacity")
                }
                remaining.append(allocation)

        return gap_filling, remaining

    def _suggest_overflow_location(self, veh_type: str, make: str) -> Dict:
        """
        Suggest location for overflow units (those that don't fill gaps).
        Based on historical velocity and available capacity.
        """
        # Get locations with capacity and velocity data
        locations = self.get_location_capacities()

        # For now, return a simple suggestion - could be enhanced with velocity data
        # Prioritize locations with higher capacity and lower fill rates
        if locations.empty:
            return {"location": "TBD", "zone": "", "reason": "No location data available"}

        # Find location with most available capacity
        if "NEW" in locations.columns and "PC" in locations.columns:
            locations_sorted = locations.sort_values("NEW", ascending=False)
            if not locations_sorted.empty:
                top_loc = locations_sorted.iloc[0]
                return {
                    "location": top_loc.get("PC", "TBD"),
                    "zone": top_loc.get("Zone", ""),
                    "reason": f"High capacity location ({top_loc.get('NEW', 0)} spots)"
                }

        return {"location": "TBD", "zone": "", "reason": "Distribute based on sales velocity"}

    def generate_inventory_flow_projection(self, months: int = 6) -> Dict:
        """
        Generate inventory flow projection showing:
        - Current inventory depletion based on velocity
        - Pipeline arrivals by estimated month
        - Projected gaps and reorder points

        Args:
            months: Number of months to project

        Returns:
            Dict with monthly projections and reorder recommendations
        """
        from datetime import datetime, timedelta
        from dateutil.relativedelta import relativedelta

        # Get current state
        current_inv = self.data_loader.load_current_inventory()
        on_order = self.data_loader.load_on_order()
        retail = self.data_loader.load_retail_history()

        # Calculate monthly velocity by type
        velocity_by_type = {}
        if "Sold Date" in retail.columns and "Veh Type" in retail.columns:
            # Last 12 months
            cutoff = datetime.now() - timedelta(days=365)
            recent = retail[retail["Sold Date"] >= cutoff]
            velocity_by_type = recent.groupby("Veh Type").size().to_dict()
            # Convert to monthly
            velocity_by_type = {k: v / 12 for k, v in velocity_by_type.items()}

        # Get lot capacity
        locations = self.get_location_capacities()
        total_capacity = locations["NEW"].sum() if "NEW" in locations.columns else 5000

        # Build monthly projection
        projections = []
        now = datetime.now()

        # Starting position
        sellable_mask = current_inv["Status Category"] == "Sellable" if "Status Category" in current_inv.columns else pd.Series(True, index=current_inv.index)
        current_sellable = len(current_inv[sellable_mask])
        current_pipeline = len(on_order) if not on_order.empty else 0

        # Estimate pipeline arrival schedule (simplified: assume even distribution)
        pipeline_per_month = current_pipeline / 3 if current_pipeline > 0 else 0  # Assume 3-month lead time

        total_monthly_velocity = sum(velocity_by_type.values())

        running_inventory = current_sellable
        running_pipeline = current_pipeline

        for i in range(months + 1):
            month_date = now + relativedelta(months=i)
            month_str = month_date.strftime("%b %Y")

            if i == 0:
                # Current month
                proj = {
                    "month": month_str,
                    "month_num": i,
                    "starting_inventory": current_sellable,
                    "projected_sales": 0,
                    "pipeline_arriving": 0,
                    "ending_inventory": current_sellable,
                    "pipeline_remaining": current_pipeline,
                    "capacity": total_capacity,
                    "fill_rate": current_sellable / total_capacity * 100 if total_capacity > 0 else 0,
                    "gap": max(0, total_capacity - current_sellable - current_pipeline),
                    "reorder_needed": 0
                }
            else:
                # Apply seasonality
                seasonality = self.config.get_seasonality_factor(month_date.month)
                adjusted_sales = total_monthly_velocity * seasonality

                # Pipeline arrivals (spread over months 1-3)
                arrivals = pipeline_per_month if i <= 3 else 0

                ending_inv = running_inventory - adjusted_sales + arrivals
                ending_inv = max(0, ending_inv)

                running_pipeline = max(0, running_pipeline - arrivals)

                # Calculate gap and reorder need
                gap = max(0, total_capacity - ending_inv - running_pipeline)

                # Reorder trigger: when ending inventory < 60 days supply
                days_supply = (ending_inv / (adjusted_sales / 30)) if adjusted_sales > 0 else 999
                reorder_needed = gap if days_supply < 60 else 0

                proj = {
                    "month": month_str,
                    "month_num": i,
                    "starting_inventory": int(running_inventory),
                    "projected_sales": int(adjusted_sales),
                    "pipeline_arriving": int(arrivals),
                    "ending_inventory": int(ending_inv),
                    "pipeline_remaining": int(running_pipeline),
                    "capacity": total_capacity,
                    "fill_rate": ending_inv / total_capacity * 100 if total_capacity > 0 else 0,
                    "gap": int(gap),
                    "days_supply": int(days_supply),
                    "reorder_needed": int(reorder_needed),
                    "seasonality_factor": round(seasonality, 2)
                }

                running_inventory = ending_inv

            projections.append(proj)

        # Summary
        total_projected_sales = sum(p.get("projected_sales", 0) for p in projections)
        total_arrivals = sum(p.get("pipeline_arriving", 0) for p in projections)
        total_reorder = sum(p.get("reorder_needed", 0) for p in projections)

        return {
            "projection_months": months,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "starting_position": {
                "current_inventory": current_sellable,
                "current_pipeline": current_pipeline,
                "total_capacity": int(total_capacity),
                "monthly_velocity": int(total_monthly_velocity)
            },
            "velocity_by_type": {k: round(v, 1) for k, v in velocity_by_type.items()},
            "monthly_projections": projections,
            "summary": {
                "total_projected_sales": total_projected_sales,
                "total_pipeline_arrivals": total_arrivals,
                "total_reorder_recommended": total_reorder,
                "ending_inventory": projections[-1]["ending_inventory"] if projections else 0,
                "ending_fill_rate": projections[-1]["fill_rate"] if projections else 0
            }
        }

    def get_fill_options_for_gap(
        self,
        location: str,
        veh_type: str,
        gap_units: int
    ) -> List[Dict]:
        """
        Get specific brand/make options to fill a gap at a location.

        Returns ranked list of makes that could fill the gap, based on:
        - Licensing status at location
        - Historical velocity
        - Margin performance
        """
        # Get licensed brands
        licensed_makes = self._get_licensed_brands_for_type(location, veh_type)

        if not licensed_makes:
            return []

        # Get velocity data for ranking
        from .market_analyzer import MarketAnalyzer
        market = MarketAnalyzer(data_loader=self.data_loader, config=self.config)

        options = []
        for make in licensed_makes:
            # Get velocity for this make
            velocity = market.get_velocity_by_make(make=make, veh_type=veh_type)

            if velocity.empty:
                monthly_vel = 0
            else:
                monthly_vel = velocity["Monthly Velocity"].sum()

            options.append({
                "make": make,
                "veh_type": veh_type,
                "location": location,
                "gap_to_fill": gap_units,
                "monthly_velocity": round(monthly_vel, 2),
                "priority": "HIGH" if monthly_vel > 5 else "MEDIUM" if monthly_vel > 1 else "LOW"
            })

        # Sort by velocity (best sellers first)
        options = sorted(options, key=lambda x: x["monthly_velocity"], reverse=True)

        return options
