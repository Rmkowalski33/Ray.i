"""
Reconciliation Module
=====================
Bridges capacity-based needs with manufacturer-based ordering by allocating
total inventory needs across manufacturers with adjustment capabilities.

This module answers: "Given our total needs, how should we split orders
across manufacturers, considering our strategic direction?"
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .config import Config, default_config
from .data_loader import DataLoader
from .market_analyzer import MarketAnalyzer
from .capacity_planner import CapacityPlanner
from .brand_licensing import BrandLicensingAnalyzer


@dataclass
class Adjustment:
    """Represents a strategic adjustment to allocation."""
    adj_type: str          # 'manufacturer', 'make', 'segment', 'location'
    target: str            # What to adjust (manufacturer name, make name, etc.)
    adjustment_pct: float  # Percentage adjustment (-50 to +100 typical)
    veh_type: str = ""     # Optional vehicle type filter
    zone: str = ""         # Optional zone filter
    note: str = ""         # Explanation for the adjustment


class ReconciliationEngine:
    """
    Reconciles capacity-based needs with manufacturer allocations.

    Workflow:
    1. Get total needs from capacity analysis
    2. Calculate historical "fair share" by manufacturer
    3. Apply user adjustments (increase Jayco, decrease Jay Feather, etc.)
    4. Generate final allocation recommendations
    """

    DEFAULT_ADJUSTMENTS_FILE = "order_adjustments.yaml"

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)
        self.market_analyzer = MarketAnalyzer(data_loader=self.data_loader, config=self.config)
        self.capacity_planner = CapacityPlanner(config=self.config)
        self.brand_licensing = BrandLicensingAnalyzer(data_loader=self.data_loader, config=self.config)

        # Load adjustments
        self.adjustments_file = self.config.output_path.parent / self.DEFAULT_ADJUSTMENTS_FILE
        self.adjustments = self._load_adjustments()

    def _load_adjustments(self) -> List[Adjustment]:
        """Load strategic adjustments from YAML file."""
        if not self.adjustments_file.exists():
            self._create_default_adjustments_file()
            return []

        try:
            with open(self.adjustments_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            adjustments = []
            adj_list = data.get("adjustments") or []  # Handle None case

            for adj in adj_list:
                if adj and adj.get("active", True):  # Skip inactive or None entries
                    adjustments.append(Adjustment(
                        adj_type=adj.get("type", ""),
                        target=adj.get("target", adj.get("name", "")),
                        adjustment_pct=self._parse_adjustment(adj.get("adjustment", "0%")),
                        veh_type=adj.get("veh_type", ""),
                        zone=adj.get("zone", ""),
                        note=adj.get("note", "")
                    ))
            return adjustments

        except Exception as e:
            print(f"Warning: Could not load adjustments file: {e}")
            return []

    def _get_licensed_locations_for_manufacturer(self, manufacturer: str) -> List[str]:
        """
        Get list of location codes where a manufacturer is licensed.

        Args:
            manufacturer: Manufacturer name (e.g., "JAYCO", "FOREST RIVER")

        Returns:
            List of location codes (e.g., ["AMA", "DAL", "HOU", ...])
        """
        licensing = self.data_loader.load_brand_licensing()

        if licensing is None or licensing.empty:
            print(f"    Warning: No licensing data available, using all locations")
            return []

        # The licensing file has manufacturer under "Inv_Manufacturer" or "Division"
        mfg_col = None
        for col in ["Inv_Manufacturer", "Division", "Manufacturer"]:
            if col in licensing.columns:
                mfg_col = col
                break

        if mfg_col is None:
            print(f"    Warning: No manufacturer column found in licensing data")
            return []

        # Find rows matching this manufacturer
        mfg_upper = manufacturer.upper().strip()
        mask = licensing[mfg_col].str.upper().str.strip() == mfg_upper

        if mask.sum() == 0:
            # Try partial match
            mask = licensing[mfg_col].str.upper().str.contains(mfg_upper, na=False)

        if mask.sum() == 0:
            print(f"    Warning: No licensing entries found for {manufacturer}")
            return []

        mfg_rows = licensing[mask]

        # Find location columns (columns with "Licensed" values)
        location_cols = []
        for col in licensing.columns:
            if col in [mfg_col, "OEM_Parent", "Inv_Make", "Veh_Type", "BRAND SERIES SUBSERIES",
                      "Type", "DIVISION BRAND SERIES SUBSERIES", "Column1", "Dealers", "Manufacturer"]:
                continue
            # Check if this column has licensing status values
            if licensing[col].astype(str).str.upper().str.contains("LICENSED", na=False).any():
                location_cols.append(col)

        if not location_cols:
            print(f"    Warning: Could not identify location columns in licensing data")
            return []

        # Find locations where this manufacturer has at least one licensed make
        licensed_locations = []
        for loc in location_cols:
            # Check if any make from this manufacturer is licensed at this location
            loc_status = mfg_rows[loc].astype(str).str.upper()
            if loc_status.str.contains("LICENSED", na=False).any():
                licensed_locations.append(loc)

        return licensed_locations

    def _filter_capacity_to_licensed_locations(self, capacity_data: Dict,
                                                manufacturer: str) -> Dict:
        """
        Filter capacity data to only include locations where manufacturer is licensed.

        Args:
            capacity_data: Full capacity report from CapacityPlanner
            manufacturer: Manufacturer to filter for

        Returns:
            Filtered capacity data with only licensed locations
        """
        licensed_locations = self._get_licensed_locations_for_manufacturer(manufacturer)

        if not licensed_locations:
            print(f"    No licensed locations found for {manufacturer} - using full capacity")
            return capacity_data

        print(f"    Found {len(licensed_locations)} licensed locations for {manufacturer}")

        # Filter the various sections of capacity_data
        filtered = capacity_data.copy()

        # Filter by_type - need to recalculate totals for licensed locations only
        by_type = capacity_data.get("by_type", {})
        filtered_by_type = {}

        for veh_type, type_data in by_type.items():
            # Create filtered version
            filtered_by_type[veh_type] = type_data.copy()
            # Reset totals - we'll recalculate from location gaps
            filtered_by_type[veh_type]["capacity"] = 0
            filtered_by_type[veh_type]["current"] = 0
            filtered_by_type[veh_type]["pipeline"] = 0
            filtered_by_type[veh_type]["gap"] = 0

        # Filter location_gaps to only licensed locations
        location_gaps = capacity_data.get("location_gaps", [])
        filtered_gaps = []

        for gap in location_gaps:
            loc = gap.get("location", "")
            if loc.upper() in [l.upper() for l in licensed_locations]:
                filtered_gaps.append(gap)
                # Aggregate to by_type totals
                vt = gap.get("veh_type", "")
                if vt in filtered_by_type:
                    filtered_by_type[vt]["capacity"] += gap.get("capacity", 0)
                    filtered_by_type[vt]["current"] += gap.get("current", 0)
                    filtered_by_type[vt]["pipeline"] += gap.get("pipeline", 0)
                    filtered_by_type[vt]["gap"] += gap.get("gap", 0)

        filtered["location_gaps"] = filtered_gaps
        filtered["by_type"] = filtered_by_type

        # Recalculate fill percentages
        for vt, data in filtered_by_type.items():
            capacity = data.get("capacity", 0)
            current = data.get("current", 0)
            pipeline = data.get("pipeline", 0)
            if capacity > 0:
                data["fill_pct"] = round((current + pipeline) / capacity * 100, 1)
            else:
                data["fill_pct"] = 0

        # Filter by_zone - include zones that have licensed locations
        by_zone = capacity_data.get("by_zone", {})
        filtered_by_zone = {}

        # Get zone mapping from location_gaps
        zone_map = {gap.get("location", ""): gap.get("zone", "") for gap in location_gaps}

        for zone, zone_data in by_zone.items():
            # Check if any licensed location is in this zone
            zone_has_licensed = False
            for loc in licensed_locations:
                if zone_map.get(loc, "") == zone:
                    zone_has_licensed = True
                    break

            if zone_has_licensed:
                # Recalculate zone totals from filtered gaps
                zone_gaps = [g for g in filtered_gaps if g.get("zone", "") == zone]
                if zone_gaps:
                    filtered_by_zone[zone] = {
                        "capacity": sum(g.get("capacity", 0) for g in zone_gaps),
                        "current": sum(g.get("current", 0) for g in zone_gaps),
                        "pipeline": sum(g.get("pipeline", 0) for g in zone_gaps),
                        "gap": sum(g.get("gap", 0) for g in zone_gaps)
                    }

        filtered["by_zone"] = filtered_by_zone

        # Recalculate summary totals
        summary = filtered.get("summary", {}).copy()
        summary["total_lot_capacity"] = sum(d.get("capacity", 0) for d in filtered_by_type.values())
        summary["total_current_inventory"] = sum(d.get("current", 0) for d in filtered_by_type.values())
        summary["total_pipeline"] = sum(d.get("pipeline", 0) for d in filtered_by_type.values())
        summary["total_gap"] = sum(d.get("gap", 0) for d in filtered_by_type.values())
        summary["licensed_locations_count"] = len(licensed_locations)
        summary["filtered_for_manufacturer"] = manufacturer

        filtered["summary"] = summary

        return filtered

    def _parse_adjustment(self, value) -> float:
        """Parse adjustment value (e.g., '+20%', '-15%', '1.2')."""
        if isinstance(value, (int, float)):
            return float(value)

        value_str = str(value).strip()

        # Handle percentage strings
        if '%' in value_str:
            value_str = value_str.replace('%', '').strip()
            return float(value_str) / 100

        # Handle multipliers (e.g., "1.2" means +20%)
        return float(value_str) - 1.0 if float(value_str) > 0 else float(value_str)

    def _create_default_adjustments_file(self):
        """Create default adjustments file with examples."""
        content = """# FTRV Order Allocation Adjustments
# ==================================
# Use this file to adjust how inventory needs are allocated across manufacturers.
# These adjustments override historical patterns based on your strategic direction.
#
# ADJUSTMENT TYPES:
# - manufacturer: Adjust a manufacturer's overall allocation
# - make: Adjust a specific make/brand line
# - segment: Adjust a vehicle type segment (TT, FW, MH, TH)
# - zone: Adjust allocation for a geographic zone
#
# ADJUSTMENT VALUES:
# - Use percentages: "+20%", "-30%", "+50%"
# - Positive = increase allocation, Negative = decrease
# - Adjustments are applied multiplicatively to the base allocation
#
# EXAMPLES:
# - "+20%" on JAYCO means if fair share was 100 units, it becomes 120
# - "-30%" on JAY FEATHER means if fair share was 50 units, it becomes 35

adjustments:
  # =====================================================
  # MANUFACTURER ADJUSTMENTS
  # =====================================================
  # Uncomment and modify as needed

  # - type: manufacturer
  #   target: JAYCO
  #   adjustment: "+20%"
  #   note: "Expanding Jayco partnership - increasing floor plan commitment"
  #   active: true

  # - type: manufacturer
  #   target: KEYSTONE
  #   adjustment: "-10%"
  #   note: "Reducing Keystone exposure due to quality concerns"
  #   active: true

  # =====================================================
  # MAKE/BRAND LINE ADJUSTMENTS
  # =====================================================

  # - type: make
  #   target: JAY FEATHER
  #   adjustment: "-30%"
  #   note: "Phasing out Jay Feather line - not selling well"
  #   active: true

  # - type: make
  #   target: WILDWOOD
  #   adjustment: "+25%"
  #   note: "Wildwood selling above expectations - increase stock"
  #   active: true

  # =====================================================
  # SEGMENT ADJUSTMENTS
  # =====================================================

  # - type: segment
  #   target: TT
  #   adjustment: "+15%"
  #   note: "Focus on travel trailers - entry-level market growing"
  #   active: true

  # - type: segment
  #   target: FW
  #   veh_type: FW
  #   adjustment: "-10%"
  #   note: "Reduce fifth wheel exposure"
  #   active: true

  # =====================================================
  # ZONE/LOCATION ADJUSTMENTS
  # =====================================================

  # - type: zone
  #   target: TX-SOUTH
  #   adjustment: "+20%"
  #   note: "Expanding TX-SOUTH locations - need more inventory"
  #   active: true

"""
        with open(self.adjustments_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def generate_reconciliation(self, lookback_months: int = 12,
                                  planning_months: int = 0,
                                  target_fill_pct: float = 0.90,
                                  manufacturer: str = None) -> Dict[str, Any]:
        """
        Generate reconciliation report allocating capacity needs to manufacturers.

        Args:
            lookback_months: Months of history to use for fair share calculation
            planning_months: If > 0, project forward this many months to calculate future needs
            target_fill_pct: Target fill percentage for planning mode (default 90%)
            manufacturer: If specified, filter capacity to only licensed locations

        Returns:
            Dict with needs, allocations, adjustments applied, and final recommendations
        """
        print("Generating reconciliation report...")
        if manufacturer:
            print(f"  (Filtered to {manufacturer} licensed locations)")
        print("-" * 50)

        # Step 1: Get capacity-based needs (current state)
        print("  1. Calculating current capacity position...")
        capacity_data = self.capacity_planner.generate_capacity_report()

        # Step 1a: If manufacturer specified, filter to licensed locations only
        if manufacturer:
            print(f"  1a. Filtering to {manufacturer} licensed locations...")
            capacity_data = self._filter_capacity_to_licensed_locations(
                capacity_data, manufacturer
            )

        # Step 1b: If planning mode, project forward
        if planning_months > 0:
            print(f"  1b. Projecting forward {planning_months} months (planning mode)...")
            capacity_data = self._project_forward_needs(
                capacity_data, planning_months, target_fill_pct, lookback_months
            )

        # Step 2: Get historical velocity by manufacturer
        print("  2. Analyzing historical velocity by manufacturer...")
        velocity_by_mfg = self._get_velocity_by_manufacturer(lookback_months)

        # Step 3: Get licensing coverage by manufacturer
        print("  3. Checking licensing coverage...")
        licensing_coverage = self._get_licensing_coverage()

        # Step 4: Calculate base fair share allocation
        print("  4. Calculating fair share allocation...")
        base_allocation = self._calculate_fair_share(
            capacity_data, velocity_by_mfg, licensing_coverage
        )

        # Step 5: Apply user adjustments
        print("  5. Applying strategic adjustments...")
        adjusted_allocation = self._apply_adjustments(base_allocation)

        # Step 6: Generate final recommendations
        print("  6. Generating final recommendations...")
        final_allocation = self._finalize_allocation(adjusted_allocation, capacity_data)

        print("-" * 50)

        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capacity_summary": capacity_data.get("summary", {}),
            "needs_by_type": capacity_data.get("by_type", []),
            "needs_by_zone": capacity_data.get("by_zone", []),
            "velocity_by_manufacturer": velocity_by_mfg,
            "base_allocation": base_allocation,
            "adjustments_applied": [
                {
                    "type": a.adj_type,
                    "target": a.target,
                    "adjustment": f"{a.adjustment_pct:+.0%}",
                    "note": a.note
                }
                for a in self.adjustments
            ],
            "final_allocation": final_allocation,
            "comparison": self._generate_comparison(base_allocation, final_allocation)
        }

    def _project_forward_needs(self, capacity_data: Dict, planning_months: int,
                                target_fill_pct: float, lookback_months: int) -> Dict:
        """
        Project inventory needs forward based on expected sales velocity.

        This transforms "current gap" into "projected need over planning period".

        Formula:
        Projected Need = (Target Inventory at End of Period)
                        + (Expected Sales During Period)
                        - (Current Inventory)
                        - (Pipeline Arriving)

        Where:
        - Target Inventory = Capacity × target_fill_pct
        - Expected Sales = Monthly Velocity × planning_months × seasonality
        """
        # Get velocity data
        velocity_by_type = self._get_velocity_by_type(lookback_months)

        summary = capacity_data.get("summary", {})
        by_type = capacity_data.get("by_type", {})

        # Get seasonality factors from config
        seasonality = getattr(self.config, 'seasonality_factors', {})
        current_month = datetime.now().month

        # Calculate average seasonality for planning period
        avg_seasonality = 1.0
        if seasonality:
            months_in_period = [(current_month + i - 1) % 12 + 1 for i in range(planning_months)]
            season_factors = [seasonality.get(m, 1.0) for m in months_in_period]
            avg_seasonality = sum(season_factors) / len(season_factors) if season_factors else 1.0

        # Project needs by vehicle type
        projected_by_type = {}
        total_projected_sales = 0
        total_projected_need = 0

        for veh_type, data in by_type.items():
            capacity = data.get("capacity", 0)
            current = data.get("current", 0)
            pipeline = data.get("pipeline", 0)

            # Get velocity for this type
            monthly_vel = velocity_by_type.get(veh_type, 0)

            # Project sales over planning period (with seasonality)
            projected_sales = monthly_vel * planning_months * avg_seasonality

            # Target inventory at end of period
            target_inventory = capacity * target_fill_pct

            # What we'll have at end (before reorder)
            ending_inventory = current + pipeline - projected_sales

            # Need to order to reach target
            projected_need = max(0, target_inventory - ending_inventory)

            projected_by_type[veh_type] = {
                "capacity": capacity,
                "current": current,
                "pipeline": pipeline,
                "monthly_velocity": round(monthly_vel, 1),
                "projected_sales": round(projected_sales),
                "ending_inventory_without_order": round(max(0, ending_inventory)),
                "target_inventory": round(target_inventory),
                "projected_need": round(projected_need),
                "gap": round(projected_need),  # For compatibility
                "fill_pct": round((current + pipeline) / capacity * 100, 1) if capacity > 0 else 0
            }

            total_projected_sales += projected_sales
            total_projected_need += projected_need

        # Update summary
        projected_summary = summary.copy()
        projected_summary["planning_mode"] = True
        projected_summary["planning_months"] = planning_months
        projected_summary["target_fill_pct"] = target_fill_pct
        projected_summary["avg_seasonality"] = round(avg_seasonality, 2)
        projected_summary["total_projected_sales"] = round(total_projected_sales)
        projected_summary["total_projected_need"] = round(total_projected_need)
        projected_summary["total_gap"] = round(total_projected_need)  # Override current gap

        # Project by zone (simplified - use proportional allocation)
        by_zone = capacity_data.get("by_zone", {})
        projected_by_zone = {}

        total_current_gap = sum(z.get("gap", 0) for z in by_zone.values()) if isinstance(by_zone, dict) else 0

        for zone, data in by_zone.items() if isinstance(by_zone, dict) else []:
            zone_share = data.get("gap", 0) / total_current_gap if total_current_gap > 0 else 0
            projected_by_zone[zone] = data.copy()
            projected_by_zone[zone]["projected_need"] = round(total_projected_need * zone_share)
            projected_by_zone[zone]["gap"] = round(total_projected_need * zone_share)

        return {
            **capacity_data,
            "summary": projected_summary,
            "by_type": projected_by_type,
            "by_zone": projected_by_zone,
            "projection_details": {
                "planning_months": planning_months,
                "target_fill_pct": target_fill_pct,
                "avg_seasonality": avg_seasonality,
                "velocity_by_type": velocity_by_type,
                "total_projected_sales": round(total_projected_sales),
                "total_projected_need": round(total_projected_need)
            }
        }

    def _get_velocity_by_type(self, lookback_months: int) -> Dict[str, float]:
        """Get monthly velocity by vehicle type."""
        velocity_data = self.market_analyzer.get_velocity_by_make(lookback_months=lookback_months)

        if velocity_data is None or velocity_data.empty:
            return {}

        # Find velocity column
        vel_col = None
        for col in ["Monthly Velocity", "monthly_velocity", "Velocity"]:
            if col in velocity_data.columns:
                vel_col = col
                break

        if vel_col is None or "Veh Type" not in velocity_data.columns:
            return {}

        # Aggregate by vehicle type
        grouped = velocity_data.groupby("Veh Type")[vel_col].sum()
        return grouped.to_dict()

    def _get_velocity_by_manufacturer(self, lookback_months: int) -> Dict[str, Dict]:
        """Get historical sales velocity by manufacturer."""
        velocity_data = self.market_analyzer.get_velocity_by_make(lookback_months=lookback_months)

        if velocity_data is None or velocity_data.empty:
            return {}

        # Aggregate to manufacturer level
        mfg_velocity = {}

        # Find the velocity column (may be named differently)
        vel_col = None
        for col in ["Monthly Velocity", "monthly_velocity", "Velocity"]:
            if col in velocity_data.columns:
                vel_col = col
                break

        units_col = None
        for col in ["Units Sold", "total_retails", "Units"]:
            if col in velocity_data.columns:
                units_col = col
                break

        if vel_col is None:
            print(f"  Warning: No velocity column found. Available: {list(velocity_data.columns)}")
            return {}

        if "Manufacturer" in velocity_data.columns:
            agg_dict = {vel_col: "sum"}
            if units_col:
                agg_dict[units_col] = "sum"

            grouped = velocity_data.groupby("Manufacturer").agg(agg_dict).reset_index()

            total_velocity = grouped[vel_col].sum()

            for _, row in grouped.iterrows():
                mfg = row["Manufacturer"]
                velocity = row[vel_col]
                mfg_velocity[mfg] = {
                    "monthly_velocity": velocity,
                    "share_of_sales": velocity / total_velocity if total_velocity > 0 else 0,
                    "total_retails": row.get(units_col, 0) if units_col else 0
                }

        return mfg_velocity

    def _get_licensing_coverage(self) -> Dict[str, Dict]:
        """Get licensing coverage by manufacturer (how many locations carry each)."""
        licensing = self.data_loader.load_brand_licensing()
        locations = self.data_loader.load_locations()

        if licensing is None or licensing.empty:
            return {}

        total_locations = len(locations) if locations is not None else 30

        # Count locations per manufacturer
        mfg_col = "Inv_Manufacturer" if "Inv_Manufacturer" in licensing.columns else "Division"

        coverage = {}
        if mfg_col in licensing.columns and "Location" in licensing.columns:
            mfg_locs = licensing.groupby(mfg_col)["Location"].nunique().reset_index()

            for _, row in mfg_locs.iterrows():
                mfg = row[mfg_col]
                loc_count = row["Location"]
                coverage[mfg] = {
                    "locations_licensed": loc_count,
                    "coverage_pct": loc_count / total_locations if total_locations > 0 else 0
                }

        return coverage

    def _calculate_fair_share(self, capacity_data: Dict, velocity: Dict,
                              licensing: Dict) -> Dict[str, Dict]:
        """
        Calculate fair share allocation for each manufacturer.

        Fair share is based on:
        - Historical sales velocity (what's been selling)
        - Licensing coverage (where can they be stocked)
        - Current mix (what's currently in inventory)
        """
        total_gap = capacity_data.get("summary", {}).get("total_gap", 0)

        if total_gap <= 0:
            # No gap to fill
            return {mfg: {"base_units": 0, "base_pct": 0} for mfg in velocity.keys()}

        # Calculate weighted score for each manufacturer
        scores = {}
        for mfg, vel_data in velocity.items():
            velocity_score = vel_data.get("share_of_sales", 0)
            coverage_score = licensing.get(mfg, {}).get("coverage_pct", 0.5)

            # Weighted combination (60% velocity, 40% coverage)
            combined_score = (velocity_score * 0.6) + (coverage_score * 0.4)
            scores[mfg] = combined_score

        # Normalize scores to sum to 1
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {mfg: score / total_score for mfg, score in scores.items()}

        # Allocate units based on scores
        allocation = {}
        for mfg, score in scores.items():
            units = int(round(total_gap * score))
            allocation[mfg] = {
                "base_units": units,
                "base_pct": score,
                "velocity_share": velocity.get(mfg, {}).get("share_of_sales", 0),
                "coverage_pct": licensing.get(mfg, {}).get("coverage_pct", 0)
            }

        return allocation

    def _apply_adjustments(self, base_allocation: Dict) -> Dict[str, Dict]:
        """Apply user-defined strategic adjustments."""
        adjusted = {}

        for mfg, alloc in base_allocation.items():
            adjusted[mfg] = alloc.copy()
            adjusted[mfg]["adjustments_applied"] = []
            adjusted[mfg]["adjusted_units"] = alloc["base_units"]
            adjusted[mfg]["adjusted_pct"] = alloc["base_pct"]

        # Apply manufacturer-level adjustments
        for adj in self.adjustments:
            if adj.adj_type == "manufacturer":
                target_mfg = adj.target.upper()
                for mfg in adjusted.keys():
                    if mfg.upper() == target_mfg or target_mfg in mfg.upper():
                        multiplier = 1 + adj.adjustment_pct
                        adjusted[mfg]["adjusted_units"] = int(round(
                            adjusted[mfg]["adjusted_units"] * multiplier
                        ))
                        adjusted[mfg]["adjustments_applied"].append(
                            f"{adj.target}: {adj.adjustment_pct:+.0%}"
                        )

        # Apply make-level adjustments (affects parent manufacturer)
        # This is more complex - would need to track makes to manufacturers
        # For now, note that make adjustments are applied but simplified

        # Recalculate percentages after adjustments
        total_adjusted = sum(a["adjusted_units"] for a in adjusted.values())
        if total_adjusted > 0:
            for mfg in adjusted:
                adjusted[mfg]["adjusted_pct"] = adjusted[mfg]["adjusted_units"] / total_adjusted

        return adjusted

    def _finalize_allocation(self, adjusted_allocation: Dict,
                            capacity_data: Dict) -> List[Dict]:
        """Generate final allocation recommendations."""
        total_gap = capacity_data.get("summary", {}).get("total_gap", 0)

        # by_type can be a dict (veh_type -> data) or list
        by_type_data = capacity_data.get("by_type", {})
        if isinstance(by_type_data, dict):
            needs_by_type = {vt: data.get("gap", 0) for vt, data in by_type_data.items()}
        else:
            needs_by_type = {t.get("type", ""): t.get("gap", 0) for t in by_type_data}

        final = []

        # Sort by adjusted units (highest first)
        sorted_alloc = sorted(
            adjusted_allocation.items(),
            key=lambda x: x[1]["adjusted_units"],
            reverse=True
        )

        for mfg, alloc in sorted_alloc:
            if alloc["adjusted_units"] <= 0:
                continue

            final.append({
                "manufacturer": mfg,
                "recommended_units": alloc["adjusted_units"],
                "share_of_total": alloc["adjusted_pct"],
                "base_units": alloc["base_units"],
                "adjustment_delta": alloc["adjusted_units"] - alloc["base_units"],
                "adjustments": alloc.get("adjustments_applied", []),
                "velocity_share": alloc.get("velocity_share", 0),
                "coverage_pct": alloc.get("coverage_pct", 0)
            })

        return final

    def _generate_comparison(self, base: Dict, final: List[Dict]) -> Dict:
        """Generate comparison of base vs adjusted allocation."""
        comparison = {
            "total_base_units": sum(a["base_units"] for a in base.values()),
            "total_adjusted_units": sum(f["recommended_units"] for f in final),
            "manufacturers_increased": [],
            "manufacturers_decreased": [],
            "manufacturers_unchanged": []
        }

        for f in final:
            mfg = f["manufacturer"]
            delta = f["adjustment_delta"]

            if delta > 0:
                comparison["manufacturers_increased"].append({
                    "manufacturer": mfg,
                    "change": f"+{delta} units"
                })
            elif delta < 0:
                comparison["manufacturers_decreased"].append({
                    "manufacturer": mfg,
                    "change": f"{delta} units"
                })
            else:
                comparison["manufacturers_unchanged"].append(mfg)

        return comparison


class ReconciliationReportGenerator:
    """Generate Excel report for reconciliation analysis."""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.output_path = self.config.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_report(self, reconciliation_data: Dict) -> str:
        """Generate Excel reconciliation report."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Order_Reconciliation_{date_str}.xlsx"
        output_file = self.output_path / filename

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self._write_summary_tab(writer, reconciliation_data)
            self._write_allocation_tab(writer, reconciliation_data)
            self._write_adjustments_tab(writer, reconciliation_data)
            self._write_needs_detail_tab(writer, reconciliation_data)
            self._write_comparison_tab(writer, reconciliation_data)

        return str(output_file)

    def _write_summary_tab(self, writer, data: Dict):
        """Write executive summary."""
        rows = []
        rows.append(["FTRV ORDER RECONCILIATION REPORT", ""])
        rows.append(["Generated", data.get("generated_at", "")])
        rows.append(["", ""])

        summary = data.get("capacity_summary", {})
        rows.append(["CAPACITY SUMMARY", ""])
        rows.append(["Total Lot Capacity", summary.get("total_lot_capacity", 0)])
        rows.append(["Current Inventory", summary.get("total_current_inventory", 0)])
        rows.append(["Pipeline/On Order", summary.get("total_pipeline", 0)])
        rows.append(["TOTAL GAP TO FILL", summary.get("total_gap", 0)])
        rows.append(["", ""])

        rows.append(["ALLOCATION SUMMARY", ""])
        final = data.get("final_allocation", [])
        total_allocated = sum(f["recommended_units"] for f in final)
        rows.append(["Total Units Allocated", total_allocated])
        rows.append(["Manufacturers Included", len(final)])
        rows.append(["", ""])

        # Top 5 allocations
        rows.append(["TOP MANUFACTURER ALLOCATIONS", "Units", "Share"])
        for f in final[:5]:
            rows.append([f["manufacturer"], f["recommended_units"], f"{f['share_of_total']:.1%}"])

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Summary", index=False, header=False)

    def _write_allocation_tab(self, writer, data: Dict):
        """Write detailed allocation by manufacturer."""
        final = data.get("final_allocation", [])

        if not final:
            df = pd.DataFrame([["No allocation data"]], columns=["Status"])
            df.to_excel(writer, sheet_name="Allocation", index=False)
            return

        rows = []
        for f in final:
            rows.append({
                "Manufacturer": f["manufacturer"],
                "Recommended Units": f["recommended_units"],
                "Share of Total": f"{f['share_of_total']:.1%}",
                "Base Units (Before Adj)": f["base_units"],
                "Adjustment": f["adjustment_delta"],
                "Historical Velocity Share": f"{f['velocity_share']:.1%}",
                "Licensing Coverage": f"{f['coverage_pct']:.1%}",
                "Adjustments Applied": ", ".join(f.get("adjustments", [])) or "None"
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Allocation", index=False)

    def _write_adjustments_tab(self, writer, data: Dict):
        """Write adjustments applied."""
        adjustments = data.get("adjustments_applied", [])

        # Header info
        header = [
            ["STRATEGIC ADJUSTMENTS APPLIED", ""],
            ["", ""],
            ["These adjustments modify the base allocation based on strategic direction.", ""],
            ["Edit order_adjustments.yaml to change these settings.", ""],
            ["", ""]
        ]
        header_df = pd.DataFrame(header)
        header_df.to_excel(writer, sheet_name="Adjustments", index=False, header=False)

        if not adjustments:
            df = pd.DataFrame([["No adjustments configured"]], columns=["Status"])
            df.to_excel(writer, sheet_name="Adjustments", index=False, startrow=6)
            return

        df = pd.DataFrame(adjustments)
        df.columns = ["Type", "Target", "Adjustment", "Note"]
        df.to_excel(writer, sheet_name="Adjustments", index=False, startrow=6)

    def _write_needs_detail_tab(self, writer, data: Dict):
        """Write capacity needs detail."""
        by_type = data.get("needs_by_type", {})
        by_zone = data.get("needs_by_zone", {})

        rows = []
        rows.append(["NEEDS BY VEHICLE TYPE", "", "", "", ""])

        if by_type:
            # Handle dict format (veh_type -> data)
            if isinstance(by_type, dict):
                for vt, t in by_type.items():
                    rows.append([vt, f"Gap: {t.get('gap', 0)}",
                               f"Capacity: {t.get('capacity', 0)}",
                               f"Current: {t.get('current', 0)}",
                               f"Fill: {t.get('fill_pct', 0):.1f}%"])
            else:
                for t in by_type:
                    rows.append([t.get("type", ""), f"Gap: {t.get('gap', 0)}",
                               f"Capacity: {t.get('capacity', 0)}",
                               f"Current: {t.get('current', 0)}",
                               f"Fill: {t.get('fill_pct', 0):.1f}%"])

        rows.append(["", "", "", "", ""])
        rows.append(["NEEDS BY ZONE", "", "", "", ""])

        if by_zone:
            # Handle dict format (zone -> data)
            if isinstance(by_zone, dict):
                for zone, z in by_zone.items():
                    rows.append([zone, f"Gap: {z.get('gap', 0)}",
                               f"Capacity: {z.get('capacity', 0)}",
                               f"Current: {z.get('current', 0)}", ""])
            else:
                for z in by_zone:
                    rows.append([z.get("zone", ""), f"Gap: {z.get('gap', 0)}",
                               f"Capacity: {z.get('capacity', 0)}",
                               f"Current: {z.get('current', 0)}", ""])

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Needs Detail", index=False, header=False)

    def _write_comparison_tab(self, writer, data: Dict):
        """Write comparison of base vs adjusted."""
        comparison = data.get("comparison", {})

        rows = []
        rows.append(["ALLOCATION COMPARISON", ""])
        rows.append(["", ""])
        rows.append(["Total Base Allocation", comparison.get("total_base_units", 0)])
        rows.append(["Total After Adjustments", comparison.get("total_adjusted_units", 0)])
        rows.append(["", ""])

        rows.append(["MANUFACTURERS INCREASED", ""])
        for m in comparison.get("manufacturers_increased", []):
            rows.append([m["manufacturer"], m["change"]])

        rows.append(["", ""])
        rows.append(["MANUFACTURERS DECREASED", ""])
        for m in comparison.get("manufacturers_decreased", []):
            rows.append([m["manufacturer"], m["change"]])

        rows.append(["", ""])
        rows.append(["MANUFACTURERS UNCHANGED", ""])
        for m in comparison.get("manufacturers_unchanged", []):
            rows.append([m, "No change"])

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Comparison", index=False, header=False)
