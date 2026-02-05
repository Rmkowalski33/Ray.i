"""
Recommendation Engine Module
Generates actionable order recommendations based on demand, inventory, and profitability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .config import Config, default_config
from .data_loader import DataLoader
from .market_analyzer import MarketAnalyzer
from .pipeline_analyzer import PipelineAnalyzer
from .financial_analyzer import FinancialAnalyzer


class RecommendationEngine:
    """Generate intelligent order recommendations."""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.data_loader = DataLoader(config=self.config)
        self.market_analyzer = MarketAnalyzer(
            data_loader=self.data_loader,
            config=self.config
        )
        self.pipeline_analyzer = PipelineAnalyzer(
            data_loader=self.data_loader,
            config=self.config
        )
        self.financial_analyzer = FinancialAnalyzer(
            data_loader=self.data_loader,
            config=self.config
        )

    # =========================================================================
    # OPTIMAL ORDER QUANTITY CALCULATION
    # =========================================================================

    def calculate_optimal_order(
        self,
        manufacturer: str = None,
        make: str = None,
        veh_type: str = None,
        floorplan: str = None,
        target_days_supply: int = None,
        forecast_months: int = None
    ) -> Dict:
        """
        Calculate optimal order quantity for a segment.

        Formula:
        Recommended Qty = (Forecast Period × Monthly Velocity)
                         - Current Inventory
                         - Pipeline Arriving in Window
                         + Seasonality Adjustment
                         + Safety Buffer

        Returns dict with recommended quantity and supporting metrics.
        """
        target = target_days_supply or self.config.target_days_supply
        forecast = forecast_months or self.config.forecast_months

        # Get current inventory position
        days_supply = self.pipeline_analyzer.calculate_days_supply(
            manufacturer=manufacturer,
            make=make,
            veh_type=veh_type,
            floorplan=floorplan,
            include_pipeline=True
        )

        # Get demand velocity
        velocity = self.market_analyzer.calculate_demand_velocity(
            manufacturer=manufacturer,
            make=make,
            veh_type=veh_type,
            floorplan=floorplan
        )

        monthly_velocity = velocity.get("monthly_velocity", 0)
        current_inventory = days_supply.get("current_inventory", 0)
        pipeline_units = days_supply.get("pipeline_units", 0)

        # Calculate forecast period demand
        forecast_demand = forecast * monthly_velocity

        # Calculate target inventory level
        target_inventory = (target / 30) * monthly_velocity

        # Calculate gap
        gap = target_inventory - current_inventory - pipeline_units

        # Seasonality adjustment - look ahead to when units will arrive
        arrival_month = (datetime.now() + timedelta(days=self.config.total_lead_time)).month
        seasonality_factor = self.config.get_seasonality_factor(arrival_month)
        adjusted_gap = gap * seasonality_factor

        # Safety stock buffer
        safety_buffer = target_inventory * self.config.safety_buffer_pct

        # Final recommendation
        recommended_qty = max(0, round(adjusted_gap + safety_buffer))

        return {
            "segment": {
                "manufacturer": manufacturer,
                "make": make,
                "veh_type": veh_type,
                "floorplan": floorplan
            },
            "current_position": {
                "inventory": current_inventory,
                "pipeline": pipeline_units,
                "total_position": current_inventory + pipeline_units,
                "days_supply": days_supply.get("days_supply", 0)
            },
            "demand": {
                "monthly_velocity": round(monthly_velocity, 2),
                "forecast_demand": round(forecast_demand, 1),
                "trend": velocity.get("trend", "unknown"),
                "sample_size": velocity.get("sample_size", 0)
            },
            "calculation": {
                "target_days_supply": target,
                "forecast_months": forecast,
                "target_inventory": round(target_inventory, 1),
                "gap_before_adjustment": round(gap, 1),
                "seasonality_factor": round(seasonality_factor, 2),
                "arrival_month": arrival_month,
                "safety_buffer": round(safety_buffer, 1)
            },
            "recommendation": {
                "quantity": recommended_qty,
                "urgency": self._classify_urgency(days_supply.get("days_supply", 999)),
                "confidence": self._assess_confidence(velocity.get("sample_size", 0))
            }
        }

    def _classify_urgency(self, days_supply: float) -> str:
        """Classify order urgency based on days supply."""
        if days_supply < self.config.critical_days_supply:
            return "CRITICAL"
        elif days_supply < self.config.min_days_supply:
            return "HIGH"
        elif days_supply <= self.config.target_days_supply:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_confidence(self, sample_size: int) -> str:
        """Assess recommendation confidence based on data quality."""
        if sample_size >= 20:
            return "HIGH"
        elif sample_size >= 10:
            return "MEDIUM"
        elif sample_size >= 5:
            return "LOW"
        else:
            return "VERY LOW - Limited data"

    # =========================================================================
    # ORDER PLAN GENERATION
    # =========================================================================

    def generate_order_plan(
        self,
        manufacturer: str = None,
        make: str = None,
        makes: List[str] = None,
        budget: float = None,
        target_units: int = None,
        forecast_months: int = None,
        start_month: str = None,
        end_month: str = None,
        veh_types: List[str] = None,
        view_by: str = "make"
    ) -> Dict:
        """
        Generate a complete order plan at the MAKE level.

        Orders are aggregated at: Type → Manufacturer → Make
        This is the appropriate granularity for ordering decisions.
        Models/floorplans change too frequently.

        Args:
            manufacturer: Optional manufacturer filter (e.g., "FOREST RIVER")
            make: Optional specific make to order (e.g., "SALEM") - single make
            makes: Optional list of makes to include - for multi-select
                   When specified, volume from excluded makes is redistributed
                   to selected makes (handles discontinued brands)
            budget: Optional budget constraint
            target_units: Optional max units constraint
            forecast_months: Months to forecast (default: config.forecast_months)
            start_month: Start month for calendar-based period (e.g., "Mar 2026")
            end_month: End month for calendar-based period (e.g., "May 2026")
            veh_types: Optional list of vehicle types to include
            view_by: Grouping for report output (make, type, zone, model)

        Returns:
            Comprehensive order plan with make-level recommendations.
        """
        # Handle calendar-based period
        if start_month and end_month:
            from dateutil import parser as date_parser
            try:
                start_dt = date_parser.parse(start_month)
                end_dt = date_parser.parse(end_month)
                months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
                forecast = max(1, months_diff)
            except Exception:
                forecast = forecast_months or self.config.forecast_months
        else:
            forecast = forecast_months or self.config.forecast_months

        # Handle single make vs multi-select makes
        single_make = make  # Backward compatibility
        selected_makes = makes  # New multi-select parameter

        # Get MAKE-level analysis for ALL makes first (to calculate total need)
        all_make_velocity = self.market_analyzer.get_velocity_by_make(
            manufacturer=manufacturer,
            make=single_make,  # Only filter if single make specified
            veh_type=veh_types[0] if veh_types and len(veh_types) == 1 else None
        )

        if all_make_velocity.empty:
            filter_desc = single_make if single_make else manufacturer
            return {
                "status": "error",
                "message": f"No historical sales data for {filter_desc}. Cannot generate order plan."
            }

        # Filter by vehicle type if specified (multiple types)
        if veh_types and "Veh Type" in all_make_velocity.columns:
            veh_types_upper = [v.upper() for v in veh_types]
            all_make_velocity = all_make_velocity[all_make_velocity["Veh Type"].isin(veh_types_upper)]

        # Calculate total manufacturer velocity (all makes)
        total_mfg_velocity = all_make_velocity["Monthly Velocity"].sum()

        # Handle multi-select makes with volume redistribution
        excluded_makes = []
        redistribution_factor = 1.0

        if selected_makes and len(selected_makes) > 0:
            # Filter to only selected makes
            selected_makes_upper = [m.upper() for m in selected_makes]
            make_velocity = all_make_velocity[
                all_make_velocity["Make"].str.upper().isin(selected_makes_upper)
            ].copy()

            # Calculate velocity of excluded makes (to redistribute)
            excluded_velocity = all_make_velocity[
                ~all_make_velocity["Make"].str.upper().isin(selected_makes_upper)
            ]["Monthly Velocity"].sum()

            excluded_makes = all_make_velocity[
                ~all_make_velocity["Make"].str.upper().isin(selected_makes_upper)
            ]["Make"].unique().tolist()

            # Calculate redistribution factor
            # Selected makes absorb the volume from excluded/discontinued makes
            selected_velocity = make_velocity["Monthly Velocity"].sum()
            if selected_velocity > 0 and excluded_velocity > 0:
                # Redistributed volume = each selected make gets proportional share of excluded volume
                redistribution_factor = (selected_velocity + excluded_velocity) / selected_velocity

            if make_velocity.empty:
                return {
                    "status": "error",
                    "message": f"None of the selected makes have sales history. Cannot generate order plan."
                }

            # Warn if redistribution factor is very high
            if redistribution_factor > 3.0:
                print(f"  Note: High redistribution factor ({redistribution_factor:.1f}x) - "
                      f"selected makes absorbing volume from {len(excluded_makes)} discontinued makes")
        else:
            # No multi-select filter - use all makes
            make_velocity = all_make_velocity.copy()

        # Get current inventory and pipeline
        inventory = self.data_loader.load_current_inventory()
        on_order = self.data_loader.load_on_order()

        # Filter inventory
        inv_mask = pd.Series(True, index=inventory.index)
        if manufacturer:
            inv_mask &= inventory["Manufacturer"].str.upper() == manufacturer.upper()
        if make:
            inv_mask &= inventory["Make"].str.upper() == make.upper()
        filtered_inventory = inventory[inv_mask]

        if "Status Category" in filtered_inventory.columns:
            filtered_inventory = filtered_inventory[filtered_inventory["Status Category"] == "Sellable"]

        # Merge data sources at MAKE level
        make_data = make_velocity.copy()

        # Add margin data at make level
        make_margins = self.financial_analyzer.get_margin_by_make(manufacturer)
        if not make_margins.empty:
            make_data = pd.merge(
                make_data,
                make_margins[["Make", "Margin %", "ROI Score"]].drop_duplicates(),
                on="Make",
                how="left"
            )
        else:
            make_data["Margin %"] = 0.15
            make_data["ROI Score"] = 1.0

        make_data["Margin %"] = make_data["Margin %"].fillna(0.15)
        make_data["ROI Score"] = make_data["ROI Score"].fillna(1.0)

        # Add current inventory counts at MAKE level
        group_cols = ["Manufacturer", "Make"]
        if "Veh Type" in filtered_inventory.columns and "Veh Type" in make_data.columns:
            group_cols.append("Veh Type")

        inv_counts = filtered_inventory.groupby([c for c in group_cols if c in filtered_inventory.columns]).size().reset_index(name="Current Inventory")
        make_data = pd.merge(make_data, inv_counts, on=[c for c in group_cols if c in inv_counts.columns], how="left")
        make_data["Current Inventory"] = make_data["Current Inventory"].fillna(0).astype(int)

        # Add pipeline counts at MAKE level
        if not on_order.empty and "Make" in on_order.columns:
            on_order_mask = pd.Series(True, index=on_order.index)
            if manufacturer:
                on_order_mask &= on_order["Manufacturer"].str.upper() == manufacturer.upper()
            if make:
                on_order_mask &= on_order["Make"].str.upper() == make.upper()
            filtered_on_order = on_order[on_order_mask]

            pipe_counts = filtered_on_order.groupby([c for c in group_cols if c in filtered_on_order.columns]).size().reset_index(name="Pipeline")
            make_data = pd.merge(make_data, pipe_counts, on=[c for c in group_cols if c in pipe_counts.columns], how="left")
            make_data["Pipeline"] = make_data["Pipeline"].fillna(0).astype(int)
        else:
            make_data["Pipeline"] = 0

        # Calculate order priority score
        make_data["Total Position"] = make_data["Current Inventory"] + make_data["Pipeline"]

        avg_days = make_data["Avg Days to Sell"].mean() if "Avg Days to Sell" in make_data.columns else 60
        make_data["Order Priority"] = (
            make_data["Monthly Velocity"] *
            make_data["ROI Score"].fillna(1.0) *
            (1 / (make_data.get("Avg Days to Sell", pd.Series([60])).fillna(60).clip(lower=30) / avg_days))
        )

        make_data = make_data.sort_values("Order Priority", ascending=False)

        # Generate MAKE-level recommendations
        recommendations = []
        total_cost = 0
        total_units = 0

        for _, row in make_data.iterrows():
            velocity = row["Monthly Velocity"]

            if velocity < 0.1:
                continue

            # Apply redistribution factor for discontinued makes
            # This increases demand for selected makes to absorb discontinued volume
            adjusted_velocity = velocity * redistribution_factor

            # Calculate recommended qty based on forecast period
            forecast_demand = adjusted_velocity * forecast
            current_position = row["Total Position"]

            # Target: forecast period demand with buffer
            target_position = forecast_demand * (1 + self.config.safety_buffer_pct)
            gap = target_position - current_position

            recommended_qty = max(0, round(gap))

            if recommended_qty == 0:
                continue

            # Estimate cost (use average price as proxy)
            avg_price = row.get("Avg Price", 0)
            est_unit_cost = avg_price * 0.75 if avg_price > 0 else 30000
            est_total_cost = recommended_qty * est_unit_cost

            # Check budget constraint
            if budget and (total_cost + est_total_cost) > budget:
                available_budget = budget - total_cost
                recommended_qty = int(available_budget / est_unit_cost)
                est_total_cost = recommended_qty * est_unit_cost

            # Check unit constraint
            if target_units and (total_units + recommended_qty) > target_units:
                recommended_qty = target_units - total_units

            if recommended_qty <= 0:
                break

            recommendations.append({
                "manufacturer": row.get("Manufacturer", manufacturer or ""),
                "make": row["Make"],
                "veh_type": row.get("Veh Type", ""),
                "recommended_qty": recommended_qty,
                "current_inventory": int(row["Current Inventory"]),
                "pipeline": int(row["Pipeline"]),
                "total_position": int(row["Total Position"]),
                "monthly_velocity": round(velocity, 2),
                "adjusted_velocity": round(adjusted_velocity, 2),
                "redistribution_factor": round(redistribution_factor, 3),
                "forecast_demand": round(forecast_demand, 1),
                "avg_price": round(row.get("Avg Price", 0), 0),
                "avg_front_end": round(row.get("Avg Front-End", 0), 0) if "Avg Front-End" in row else 0,
                "margin_pct": round(row.get("Margin %", 0.15), 3),
                "priority_score": round(row["Order Priority"], 2),
                "estimated_unit_cost": round(est_unit_cost, 0),
                "estimated_total_cost": round(est_total_cost, 0),
                "urgency": self._classify_model_urgency(row["Current Inventory"], adjusted_velocity)
            })

            total_cost += est_total_cost
            total_units += recommended_qty

            if len(recommendations) >= self.config.max_models_in_report:
                break

        # Get inventory summary
        inventory_summary = self.pipeline_analyzer.get_inventory_position_summary(manufacturer or make)

        # Get market context
        market_share = self.market_analyzer.calculate_market_share(manufacturer)

        return {
            "status": "success",
            "manufacturer": manufacturer,
            "make": single_make,
            "makes_selected": selected_makes,
            "makes_excluded": excluded_makes,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "forecast_months": forecast,
                "start_month": start_month,
                "end_month": end_month,
                "budget_constraint": budget,
                "units_constraint": target_units,
                "vehicle_types": veh_types,
                "view_by": view_by
            },
            "redistribution": {
                "factor": round(redistribution_factor, 3),
                "excluded_count": len(excluded_makes),
                "excluded_makes": excluded_makes[:10] if excluded_makes else [],  # Limit for display
                "note": f"Volume from {len(excluded_makes)} discontinued/excluded makes redistributed to selected makes"
                        if excluded_makes else None
            },
            "summary": {
                "total_recommended_units": total_units,
                "total_estimated_cost": round(total_cost, 0),
                "budget_used": round(total_cost / budget, 3) if budget else None,
                "makes_recommended": len(recommendations),
                "avg_priority_score": round(
                    sum(r["priority_score"] for r in recommendations) / len(recommendations), 2
                ) if recommendations else 0
            },
            "inventory_position": inventory_summary,
            "market_context": market_share,
            "recommendations": recommendations,
            "by_urgency": self._group_by_urgency(recommendations),
            "by_veh_type": self._group_by_veh_type(recommendations)
        }

    def _classify_model_urgency(self, current_inventory: int, monthly_velocity: float) -> str:
        """Classify model urgency based on current position vs velocity."""
        if monthly_velocity <= 0:
            return "LOW"

        months_supply = current_inventory / monthly_velocity
        if months_supply < 1:
            return "CRITICAL"
        elif months_supply < 2:
            return "HIGH"
        elif months_supply < 3:
            return "MEDIUM"
        else:
            return "LOW"

    def _group_by_urgency(self, recommendations: List[Dict]) -> Dict:
        """Group recommendations by urgency level."""
        groups = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for rec in recommendations:
            urgency = rec.get("urgency", "LOW")
            if urgency in groups:
                groups[urgency].append(rec)
        return {
            k: {
                "count": len(v),
                "total_units": sum(r["recommended_qty"] for r in v),
                "total_cost": sum(r["estimated_total_cost"] for r in v)
            }
            for k, v in groups.items()
        }

    def _group_by_veh_type(self, recommendations: List[Dict]) -> Dict:
        """Group recommendations by vehicle type."""
        groups = {}
        for rec in recommendations:
            veh_type = rec.get("veh_type", "OTHER")
            if veh_type not in groups:
                groups[veh_type] = []
            groups[veh_type].append(rec)
        return {
            k: {
                "count": len(v),
                "total_units": sum(r["recommended_qty"] for r in v),
                "total_cost": sum(r["estimated_total_cost"] for r in v)
            }
            for k, v in groups.items()
        }

    # =========================================================================
    # ANALYSIS SUMMARIES
    # =========================================================================

    def get_manufacturer_summary(self, manufacturer: str) -> Dict:
        """Get comprehensive summary for a manufacturer."""
        inventory_position = self.pipeline_analyzer.get_inventory_position_summary(manufacturer)
        market_share = self.market_analyzer.calculate_market_share(manufacturer)
        holding_costs = self.financial_analyzer.get_total_holding_cost(manufacturer)
        margin_data = self.financial_analyzer.calculate_segment_margin(manufacturer=manufacturer)
        reorder_needs = self.pipeline_analyzer.identify_reorder_needs(manufacturer)

        return {
            "manufacturer": manufacturer,
            "inventory_position": inventory_position,
            "market_share": market_share,
            "holding_costs": holding_costs,
            "margin_performance": margin_data,
            "segments_needing_reorder": len(reorder_needs) if not reorder_needs.empty else 0,
            "total_reorder_units_needed": reorder_needs["Units Needed"].sum() if not reorder_needs.empty and "Units Needed" in reorder_needs.columns else 0
        }

    def list_manufacturers(self) -> List[Dict]:
        """List all manufacturers with summary stats."""
        retail = self.data_loader.load_retail_history()

        mfg_stats = (
            retail.groupby("Manufacturer")
            .agg({
                "Stock#": "count",
                "Deal FrontEnd": "sum" if "Deal FrontEnd" in retail.columns else "count"
            })
            .reset_index()
        )

        mfg_stats.columns = ["Manufacturer", "Units Sold", "Total Front-End"]
        mfg_stats = mfg_stats.sort_values("Units Sold", ascending=False)

        return mfg_stats.head(30).to_dict("records")

    def list_makes(self, manufacturer: str = None) -> List[Dict]:
        """
        List all makes with summary stats.

        Args:
            manufacturer: Optional filter by manufacturer

        Returns:
            List of dicts with make info: Manufacturer, Make, Veh Type, Units Sold, Total Front-End
        """
        retail = self.data_loader.load_retail_history()

        if manufacturer:
            retail = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]

        group_cols = ["Manufacturer", "Make"]
        if "Veh Type" in retail.columns:
            group_cols.append("Veh Type")

        agg_cols = {"Stock#": "count"}
        if "Deal FrontEnd" in retail.columns:
            agg_cols["Deal FrontEnd"] = "sum"

        make_stats = (
            retail.groupby(group_cols)
            .agg(agg_cols)
            .reset_index()
        )

        col_names = group_cols + ["Units Sold"]
        if "Deal FrontEnd" in agg_cols:
            col_names.append("Total Front-End")
        make_stats.columns = col_names

        make_stats = make_stats.sort_values("Units Sold", ascending=False)

        return make_stats.to_dict("records")

    def get_makes_for_manufacturer(self, manufacturer: str) -> List[str]:
        """Get list of makes for a specific manufacturer."""
        retail = self.data_loader.load_retail_history()
        filtered = retail[retail["Manufacturer"].str.upper() == manufacturer.upper()]
        if "Make" in filtered.columns:
            return sorted(filtered["Make"].dropna().unique().tolist())
        return []
