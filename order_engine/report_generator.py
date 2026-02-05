"""
Report Generator Module
Outputs analysis to Excel workbooks with multiple tabs.

Enhanced with:
- On Order / Pipeline Analysis tab
- Year-over-Year Comparison tab
- Market Share Analysis tab
- Recommendation Transparency tab
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

from .config import Config, default_config
from .data_loader import DataLoader
from .pipeline_analyzer import PipelineAnalyzer
from .market_analyzer import MarketAnalyzer
from .yoy_analyzer import YoYAnalyzer
from .transparency import TransparencyGenerator
from .monthly_planner import MonthlyPlanner
from .brand_licensing import BrandLicensingAnalyzer


class ReportGenerator:
    """Generate Excel reports from analysis results."""

    def __init__(self, config: Config = None):
        self.config = config or default_config
        self.output_path = self.config.output_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize analyzers for enhanced tabs
        self.data_loader = DataLoader(config=self.config)
        self.pipeline_analyzer = PipelineAnalyzer(data_loader=self.data_loader, config=self.config)
        self.market_analyzer = MarketAnalyzer(data_loader=self.data_loader, config=self.config)
        self.yoy_analyzer = YoYAnalyzer(data_loader=self.data_loader, config=self.config)
        self.transparency = TransparencyGenerator(data_loader=self.data_loader, config=self.config)
        self.monthly_planner = MonthlyPlanner(data_loader=self.data_loader, config=self.config)
        self.brand_licensing = BrandLicensingAnalyzer(data_loader=self.data_loader, config=self.config)

    def generate_order_plan_report(
        self,
        order_plan: Dict[str, Any],
        filename: str = None
    ) -> str:
        """
        Generate a comprehensive order plan report.

        Args:
            order_plan: Output from RecommendationEngine.generate_order_plan()
            filename: Optional output filename (auto-generated if not provided)

        Returns:
            Path to generated Excel file
        """
        if order_plan.get("status") == "error":
            raise ValueError(order_plan.get("message", "Order plan generation failed"))

        if not filename:
            manufacturer = order_plan.get("manufacturer", "Unknown")
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            period = order_plan.get("parameters", {}).get("forecast_months", 3)
            filename = f"{manufacturer}_OrderPlan_{period}mo_{date_str}.xlsx"

        output_file = self.output_path / filename

        manufacturer = order_plan.get("manufacturer")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Core tabs
            self._write_summary_tab(writer, order_plan)
            self._write_recommendations_tab(writer, order_plan)
            self._write_monthly_distribution_tab(writer, order_plan)  # NEW Phase 2
            self._write_inventory_position_tab(writer, order_plan)
            self._write_urgency_breakdown_tab(writer, order_plan)

            # Enhanced Phase 1 tabs
            self._write_on_order_tab(writer, manufacturer)
            self._write_yoy_tab(writer, manufacturer)
            self._write_market_share_tab(writer, manufacturer)
            self._write_transparency_tab(writer, order_plan)

            # Brand Licensing tab (Phase 3)
            self._write_brand_licensing_tab(writer, order_plan)

            # Zone and Location-level monthly views
            forecast_months = order_plan.get("parameters", {}).get("forecast_months", 3)
            self._write_monthly_by_zone_tab(writer, manufacturer, forecast_months)
            self._write_monthly_by_location_tab(writer, manufacturer, forecast_months)

            # Config tab last
            self._write_parameters_tab(writer, order_plan)

        return str(output_file)

    def _write_summary_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Executive Summary tab."""
        rows = []

        rows.append(["FTRV ORDER RECOMMENDATION REPORT", ""])
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        rows.append(["Manufacturer", order_plan.get("manufacturer", "N/A")])
        rows.append(["", ""])

        # Parameters
        params = order_plan.get("parameters", {})
        rows.append(["PARAMETERS", ""])

        # Show calendar months or rolling period
        if params.get("start_month") and params.get("end_month"):
            rows.append(["Period", f"{params['start_month']} to {params['end_month']}"])
        rows.append(["Forecast Period", f"{params.get('forecast_months', 3)} months"])

        if params.get("budget_constraint"):
            rows.append(["Budget Constraint", f"${params['budget_constraint']:,.0f}"])
        if params.get("units_constraint"):
            rows.append(["Units Constraint", params["units_constraint"]])
        if params.get("vehicle_types"):
            rows.append(["Vehicle Types", ", ".join(params["vehicle_types"])])
        rows.append(["", ""])

        # Make Selection / Redistribution info
        redistribution = order_plan.get("redistribution", {})
        if redistribution.get("excluded_count", 0) > 0:
            rows.append(["MAKE SELECTION", ""])
            rows.append(["Makes Selected", len(order_plan.get("makes_selected", []))])
            rows.append(["Makes Excluded (Discontinued)", redistribution["excluded_count"]])
            rows.append(["Volume Redistribution Factor", f"{redistribution.get('factor', 1.0):.2f}x"])
            if redistribution.get("excluded_makes"):
                excluded_list = ", ".join(redistribution["excluded_makes"][:5])
                if len(redistribution["excluded_makes"]) > 5:
                    excluded_list += f"... (+{len(redistribution['excluded_makes'])-5} more)"
                rows.append(["Excluded Makes", excluded_list])
            rows.append(["Note", "Volume from excluded makes redistributed to selected makes"])
            rows.append(["", ""])

        # Summary
        summary = order_plan.get("summary", {})
        rows.append(["ORDER SUMMARY", ""])
        rows.append(["Total Recommended Units", summary.get("total_recommended_units", 0)])
        rows.append(["Total Estimated Cost", f"${summary.get('total_estimated_cost', 0):,.0f}"])
        rows.append(["Makes Included", summary.get("makes_recommended", summary.get("models_recommended", 0))])
        if summary.get("budget_used"):
            rows.append(["Budget Utilized", f"{summary['budget_used']:.1%}"])
        rows.append(["", ""])

        # Inventory Position
        inv_pos = order_plan.get("inventory_position", {})
        rows.append(["CURRENT INVENTORY POSITION", ""])
        rows.append(["Total Inventory", inv_pos.get("total_inventory", 0)])
        rows.append(["Days Supply", f"{inv_pos.get('days_supply', 0):.0f} days"])
        rows.append(["Monthly Sales Rate", f"{inv_pos.get('monthly_sales_rate', 0):.1f} units/mo"])
        rows.append(["Inventory Health", inv_pos.get("inventory_health", "N/A")])
        rows.append(["Average Age", f"{inv_pos.get('avg_age_days', 0):.0f} days"])
        rows.append(["Units 90+ Days Old", inv_pos.get("aging_90_plus_units", 0)])
        rows.append(["", ""])

        # Market Context
        market = order_plan.get("market_context", {})
        if market and not market.get("error"):
            rows.append(["MARKET CONTEXT", ""])
            rows.append(["FTRV Sales (LTM)", market.get("ftrv_sales", 0)])
            rows.append(["Total Market", market.get("total_market", 0)])
            rows.append(["Market Share", market.get("market_share_pct", "N/A")])
            rows.append(["", ""])

        # Urgency Breakdown
        by_urgency = order_plan.get("by_urgency", {})
        rows.append(["URGENCY BREAKDOWN", "Units", "Est. Cost"])
        for urgency in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if urgency in by_urgency:
                data = by_urgency[urgency]
                rows.append([urgency, data.get("total_units", 0), f"${data.get('total_cost', 0):,.0f}"])

        df = pd.DataFrame(rows, columns=["Item", "Value", "Extra"])
        df.to_excel(writer, sheet_name="Summary", index=False)

    def _write_recommendations_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Order Recommendations tab at MAKE level (Type → Manufacturer → Make hierarchy)."""
        recs = order_plan.get("recommendations", [])

        if not recs:
            df = pd.DataFrame([["No recommendations generated"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Recommendations", index=False)
            return

        df = pd.DataFrame(recs)

        # Select and order columns - MAKE level hierarchy (not model/floorplan)
        # Order: Type → Manufacturer → Make
        columns = [
            "veh_type", "manufacturer", "make", "recommended_qty", "urgency",
            "current_inventory", "pipeline", "total_position",
            "monthly_velocity", "forecast_demand",
            "avg_price", "avg_front_end", "margin_pct",
            "estimated_unit_cost", "estimated_total_cost", "priority_score"
        ]
        columns = [c for c in columns if c in df.columns]
        df = df[columns]

        # Sort by Type → Manufacturer → Make hierarchy
        sort_cols = [c for c in ["veh_type", "manufacturer", "make"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        # Rename columns for readability
        rename_map = {
            "veh_type": "Veh Type",
            "manufacturer": "Manufacturer",
            "make": "Make",
            "recommended_qty": "Recommended Qty",
            "urgency": "Urgency",
            "current_inventory": "Current Inv",
            "pipeline": "On Order",
            "total_position": "Total Position",
            "monthly_velocity": "Monthly Velocity",
            "forecast_demand": "Forecast Demand",
            "avg_price": "Avg Price",
            "avg_front_end": "Avg Front-End",
            "margin_pct": "Margin %",
            "estimated_unit_cost": "Est. Unit Cost",
            "estimated_total_cost": "Est. Total Cost",
            "priority_score": "Priority Score"
        }
        df = df.rename(columns=rename_map)

        # Format currency columns
        currency_cols = ["Avg Price", "Avg Front-End", "Est. Unit Cost", "Est. Total Cost"]
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "")

        # Format percentage columns
        if "Margin %" in df.columns:
            df["Margin %"] = df["Margin %"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")

        df.to_excel(writer, sheet_name="Recommendations", index=False)

    def _write_inventory_position_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Inventory Position tab."""
        inv_pos = order_plan.get("inventory_position", {})

        rows = []
        rows.append(["INVENTORY POSITION DETAILS", ""])
        rows.append(["", ""])

        for key, value in inv_pos.items():
            if isinstance(value, float):
                if "pct" in key.lower():
                    rows.append([key.replace("_", " ").title(), f"{value:.1%}"])
                elif value > 1000:
                    rows.append([key.replace("_", " ").title(), f"${value:,.0f}"])
                else:
                    rows.append([key.replace("_", " ").title(), f"{value:.1f}"])
            else:
                rows.append([key.replace("_", " ").title(), value])

        df = pd.DataFrame(rows, columns=["Metric", "Value"])
        df.to_excel(writer, sheet_name="Inventory Position", index=False)

    def _write_urgency_breakdown_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Urgency Breakdown tab."""
        by_urgency = order_plan.get("by_urgency", {})

        rows = []
        rows.append(["Urgency Level", "Make Count", "Total Units", "Total Est. Cost"])

        for urgency in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if urgency in by_urgency:
                data = by_urgency[urgency]
                rows.append([
                    urgency,
                    data.get("count", 0),
                    data.get("total_units", 0),
                    f"${data.get('total_cost', 0):,.0f}"
                ])

        # Add vehicle type breakdown
        rows.append(["", "", "", ""])
        rows.append(["VEHICLE TYPE BREAKDOWN", "", "", ""])
        rows.append(["Vehicle Type", "Make Count", "Total Units", "Total Est. Cost"])

        by_veh_type = order_plan.get("by_veh_type", {})
        for veh_type, data in sorted(by_veh_type.items()):
            rows.append([
                veh_type,
                data.get("count", 0),
                data.get("total_units", 0),
                f"${data.get('total_cost', 0):,.0f}"
            ])

        df = pd.DataFrame(rows, columns=["Category", "Count", "Units", "Cost"])
        df.to_excel(writer, sheet_name="Breakdown", index=False)

    def _write_parameters_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Parameters tab for reproducibility."""
        params = order_plan.get("parameters", {})

        rows = []
        rows.append(["REPORT PARAMETERS", ""])
        rows.append(["Generated At", order_plan.get("generated_at", "")])
        rows.append(["Manufacturer", order_plan.get("manufacturer", "")])
        rows.append(["", ""])
        rows.append(["CONFIGURATION", ""])
        rows.append(["Forecast Period (months)", params.get("forecast_months", 3)])
        rows.append(["Budget Constraint", f"${params['budget_constraint']:,.0f}" if params.get("budget_constraint") else "None"])
        rows.append(["Units Constraint", params.get("units_constraint") or "None"])
        rows.append(["Vehicle Types Filter", ", ".join(params.get("vehicle_types") or []) or "All"])
        rows.append(["", ""])
        rows.append(["DATA SOURCES", ""])
        rows.append(["Data Hub Path", str(self.config.data_hub_path)])
        rows.append(["Inventory File", self.config.inventory_file])
        rows.append(["References File", self.config.references_file])
        rows.append(["SSI Data File", self.config.ssi_data_file])
        rows.append(["", ""])
        rows.append(["THRESHOLDS", ""])
        rows.append(["Target Days Supply", self.config.target_days_supply])
        rows.append(["Min Days Supply", self.config.min_days_supply])
        rows.append(["Max Days Supply", self.config.max_days_supply])
        rows.append(["Lookback Months", self.config.lookback_months])
        rows.append(["Safety Buffer %", f"{self.config.safety_buffer_pct:.0%}"])
        rows.append(["Floorplan Rate", f"{self.config.floorplan_rate:.1%}"])
        rows.append(["Total Lead Time (days)", self.config.total_lead_time])

        df = pd.DataFrame(rows, columns=["Parameter", "Value"])
        df.to_excel(writer, sheet_name="Parameters", index=False)

    # =========================================================================
    # MONTHLY DISTRIBUTION (Phase 2)
    # =========================================================================

    def _write_monthly_distribution_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Monthly Distribution tab showing recommendations by month."""
        manufacturer = order_plan.get("manufacturer")
        params = order_plan.get("parameters", {})
        num_months = params.get("forecast_months", 3)

        # Generate monthly plan
        monthly_plan = self.monthly_planner.generate_monthly_plan(
            manufacturer=manufacturer,
            num_months=num_months
        )

        # Convert to DataFrame
        df = self.monthly_planner.to_dataframe(monthly_plan)

        if df.empty:
            rows = [["No monthly distribution data available"]]
            pd.DataFrame(rows, columns=["Message"]).to_excel(
                writer, sheet_name="Monthly Distribution", index=False
            )
            return

        # Write summary header
        rows = []
        rows.append(["MONTHLY ORDER DISTRIBUTION", ""])
        rows.append(["Manufacturer", manufacturer])
        rows.append(["Base Monthly Velocity", f"{monthly_plan.get('base_monthly_velocity', 0):.1f} units/mo"])
        rows.append(["Current Inventory", monthly_plan.get("current_inventory", 0)])
        rows.append(["Pipeline Units", monthly_plan.get("total_pipeline", 0)])
        rows.append(["", ""])

        # Totals
        totals = monthly_plan.get("totals", {})
        rows.append(["TOTALS", ""])
        rows.append(["Total Projected Demand", f"{totals.get('total_projected_demand', 0):.0f} units"])
        rows.append(["Total Recommended Order", f"{totals.get('total_recommended_order', 0)} units"])
        rows.append(["Total Pipeline Arriving", f"{totals.get('total_pipeline_arriving', 0)} units"])
        rows.append(["Avg Monthly Order", f"{totals.get('avg_monthly_order', 0):.0f} units"])
        rows.append(["", ""])
        rows.append(["MONTHLY BREAKDOWN", ""])

        # Write header section
        header_df = pd.DataFrame(rows, columns=["Item", "Value"])
        header_df.to_excel(writer, sheet_name="Monthly Distribution", index=False)

        # Write monthly data below header
        df.to_excel(writer, sheet_name="Monthly Distribution", index=False, startrow=len(rows) + 1)

        # Write segment breakdowns on separate sheets
        self._write_segment_breakdown(writer, manufacturer, num_months, "veh_type", "Monthly by Type")
        self._write_segment_breakdown(writer, manufacturer, num_months, "make", "Monthly by Make")

    def _write_segment_breakdown(
        self,
        writer: pd.ExcelWriter,
        manufacturer: str,
        num_months: int,
        segment_by: str,
        sheet_name: str
    ):
        """Write segment-level monthly breakdown."""
        try:
            segment_plan = self.monthly_planner.generate_monthly_plan_by_segment(
                manufacturer=manufacturer,
                segment_by=segment_by,
                num_months=num_months
            )

            segment_title = {
                "veh_type": "VEHICLE TYPE",
                "make": "MAKE/BRAND",
                "zone": "ZONE"
            }.get(segment_by, segment_by.upper())

            rows = []
            rows.append([f"MONTHLY DISTRIBUTION BY {segment_title}", ""])
            rows.append(["", ""])

            segments = segment_plan.get("segments", {})

            # Summary table header
            rows.append(["Segment", "Projected Demand", "Recommended Order", "Current Inv", "Pipeline"])

            for segment, plan in sorted(segments.items()):
                if isinstance(plan, dict) and "totals" in plan:
                    totals = plan.get("totals", {})
                    rows.append([
                        segment,
                        f"{totals.get('total_projected_demand', 0):.0f}",
                        totals.get('total_recommended_order', 0),
                        plan.get('current_inventory', 0),
                        plan.get('total_pipeline', 0)
                    ])

            rows.append(["", "", "", "", ""])

            # Detailed monthly breakdown for top segments
            rows.append(["DETAILED MONTHLY BREAKDOWN (Top Segments)", ""])
            count = 0
            for segment, plan in sorted(segments.items(), key=lambda x: x[1].get('totals', {}).get('total_projected_demand', 0) if isinstance(x[1], dict) else 0, reverse=True):
                if count >= 5:  # Top 5 segments
                    break
                if not isinstance(plan, dict) or "monthly_buckets" not in plan:
                    continue

                rows.append(["", ""])
                rows.append([f"=== {segment} ===", ""])
                rows.append(["Month", "Demand", "Starting Inv", "Pipeline", "Gap", "Order"])

                for bucket in plan.get("monthly_buckets", []):
                    rows.append([
                        bucket.get("month", ""),
                        f"{bucket.get('projected_demand', 0):.0f}",
                        bucket.get("starting_inventory", 0),
                        bucket.get("pipeline_arriving", 0),
                        bucket.get("gap", 0),
                        bucket.get("recommended_order", 0)
                    ])
                count += 1

            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        except Exception:
            pass  # Skip if segment breakdown fails

    def _write_monthly_by_zone_tab(
        self,
        writer: pd.ExcelWriter,
        manufacturer: str,
        num_months: int
    ):
        """Write Monthly by Zone tab showing zone-level monthly breakdown."""
        try:
            zone_plan = self.monthly_planner.generate_monthly_plan_by_segment(
                manufacturer=manufacturer,
                segment_by="zone",
                num_months=num_months
            )

            rows = []
            rows.append(["MONTHLY ORDER DISTRIBUTION BY ZONE", ""])
            rows.append(["Manufacturer", manufacturer])
            rows.append(["Planning Horizon", f"{num_months} months"])
            rows.append(["", ""])

            segments = zone_plan.get("segments", {})

            # Summary table
            rows.append(["ZONE SUMMARY", "", "", "", "", ""])
            rows.append(["Zone", "Locations", "Current Inv", "Pipeline", "Proj. Demand", "Recommended Order"])

            for zone, plan in sorted(segments.items()):
                if isinstance(plan, dict) and "totals" in plan:
                    totals = plan.get("totals", {})
                    rows.append([
                        zone,
                        plan.get("location_count", len(plan.get("locations", []))),
                        plan.get("current_inventory", 0),
                        plan.get("total_pipeline", 0),
                        f"{totals.get('total_projected_demand', 0):.0f}",
                        totals.get('total_recommended_order', 0)
                    ])

            rows.append(["", "", "", "", "", ""])

            # Detailed monthly breakdown for each zone
            rows.append(["DETAILED MONTHLY BREAKDOWN BY ZONE", ""])
            for zone, plan in sorted(segments.items()):
                if not isinstance(plan, dict) or "monthly_buckets" not in plan:
                    continue

                rows.append(["", ""])
                rows.append([f"=== {zone} ===", "", "", "", "", ""])
                rows.append([
                    f"Base Velocity: {plan.get('base_monthly_velocity', 0):.1f}/mo",
                    f"Current Inv: {plan.get('current_inventory', 0)}",
                    f"Pipeline: {plan.get('total_pipeline', 0)}",
                    "", "", ""
                ])
                rows.append(["Month", "Seasonality", "Demand", "Starting Inv", "Pipeline In", "Recommended"])

                for bucket in plan.get("monthly_buckets", []):
                    rows.append([
                        bucket.get("month", ""),
                        bucket.get("seasonality_index", 100),
                        f"{bucket.get('projected_demand', 0):.0f}",
                        bucket.get("starting_inventory", 0),
                        bucket.get("pipeline_arriving", 0),
                        bucket.get("recommended_order", 0)
                    ])

            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name="Monthly by Zone", index=False, header=False)

        except Exception as e:
            rows = [["Error generating zone monthly breakdown", str(e)]]
            df = pd.DataFrame(rows, columns=["Error", "Details"])
            df.to_excel(writer, sheet_name="Monthly by Zone", index=False)

    def _write_monthly_by_location_tab(
        self,
        writer: pd.ExcelWriter,
        manufacturer: str,
        num_months: int
    ):
        """Write Monthly by Location tab showing location-level monthly breakdown."""
        try:
            location_plan = self.monthly_planner.generate_monthly_plan_by_location(
                manufacturer=manufacturer,
                num_months=num_months
            )

            rows = []
            rows.append(["MONTHLY ORDER DISTRIBUTION BY LOCATION", ""])
            rows.append(["Manufacturer", manufacturer])
            rows.append(["Planning Horizon", f"{num_months} months"])
            rows.append(["Total Locations", location_plan.get("location_count", 0)])
            rows.append(["", ""])

            # Aggregate totals
            agg = location_plan.get("aggregate_totals", {})
            rows.append(["AGGREGATE TOTALS", ""])
            rows.append(["Total Current Inventory", agg.get("total_current_inventory", 0)])
            rows.append(["Total Pipeline", agg.get("total_pipeline", 0)])
            rows.append(["Total Projected Demand", f"{agg.get('total_projected_demand', 0):.0f}"])
            rows.append(["Total Recommended Order", agg.get("total_recommended_order", 0)])
            rows.append(["", ""])

            location_plans = location_plan.get("location_plans", {})

            # Location summary table
            rows.append(["LOCATION SUMMARY", "", "", "", "", "", ""])
            rows.append(["PC", "Name", "Zone", "Current Inv", "Pipeline", "Proj. Demand", "Recommended"])

            for pc, plan in sorted(location_plans.items(), key=lambda x: x[1].get("totals", {}).get("total_recommended_order", 0), reverse=True):
                if isinstance(plan, dict) and "totals" in plan:
                    totals = plan.get("totals", {})
                    rows.append([
                        pc,
                        plan.get("location_name", "")[:20],
                        plan.get("zone", ""),
                        plan.get("current_inventory", 0),
                        plan.get("total_pipeline", 0),
                        f"{totals.get('total_projected_demand', 0):.0f}",
                        totals.get('total_recommended_order', 0)
                    ])

            if rows:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name="Monthly by Location", index=False, header=False)

            # Write detailed monthly breakdown on a separate sheet for top locations
            detail_rows = []
            detail_rows.append(["MONTHLY DETAIL BY LOCATION (Top 20 by Volume)", ""])
            detail_rows.append(["", ""])

            # Sort by recommended order and take top 20
            sorted_locs = sorted(
                location_plans.items(),
                key=lambda x: x[1].get("totals", {}).get("total_recommended_order", 0),
                reverse=True
            )[:20]

            for pc, plan in sorted_locs:
                if not isinstance(plan, dict) or "monthly_buckets" not in plan:
                    continue

                detail_rows.append([f"=== {pc} - {plan.get('location_name', '')} ({plan.get('zone', '')}) ===", ""])
                detail_rows.append([
                    f"Velocity: {plan.get('base_monthly_velocity', 0):.1f}/mo",
                    f"Current: {plan.get('current_inventory', 0)}",
                    f"Pipeline: {plan.get('total_pipeline', 0)}",
                    "", ""
                ])
                detail_rows.append(["Month", "Demand", "Starting Inv", "Pipeline In", "Recommended"])

                for bucket in plan.get("monthly_buckets", []):
                    detail_rows.append([
                        bucket.get("month", ""),
                        f"{bucket.get('projected_demand', 0):.1f}",
                        bucket.get("starting_inventory", 0),
                        bucket.get("pipeline_arriving", 0),
                        bucket.get("recommended_order", 0)
                    ])
                detail_rows.append(["", "", "", "", ""])

            if detail_rows:
                df_detail = pd.DataFrame(detail_rows)
                df_detail.to_excel(writer, sheet_name="Location Monthly Detail", index=False, header=False)

        except Exception as e:
            rows = [["Error generating location monthly breakdown", str(e)]]
            df = pd.DataFrame(rows, columns=["Error", "Details"])
            df.to_excel(writer, sheet_name="Monthly by Location", index=False)

    # =========================================================================
    # ENHANCED TABS (Phase 1)
    # =========================================================================

    def _write_on_order_tab(self, writer: pd.ExcelWriter, manufacturer: str):
        """Write On Order / Pipeline Analysis tab."""
        rows = []
        rows.append(["ON ORDER / PIPELINE ANALYSIS", ""])
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        rows.append(["", ""])

        # Summary metrics
        summary = self.pipeline_analyzer.get_on_order_summary(manufacturer)
        rows.append(["PIPELINE SUMMARY", ""])
        rows.append(["Total On Order/Pipeline", summary.get("total_on_order", 0)])
        rows.append(["Average Age (days)", summary.get("avg_age", 0)])
        rows.append(["Total Value", f"${summary.get('total_value', 0):,.0f}"])
        rows.append(["", ""])

        # PO and Location Status - KEY DISTINCTION
        rows.append(["PO & LOCATION STATUS", ""])
        rows.append(["Needs PO Assigned", summary.get("needs_po", 0), "(ORDERED status - no PO yet)"])
        rows.append(["Has PO Issued", summary.get("has_po", 0), "(PURCHASED, PO ISSUED)"])
        rows.append(["Needs Location Assignment", summary.get("needs_location", 0), "(YARD/PDI POOL)"])
        rows.append(["In Transit", summary.get("in_transit", 0), "(DISPATCHED, SHIPPED, TRANSFER)"])
        rows.append(["", "", ""])

        # By status
        rows.append(["BY STATUS", "Units", "Avg Age", "Total Cost"])
        by_status = summary.get("by_status", {})
        for status, data in sorted(by_status.items(), key=lambda x: x[1].get("count", 0), reverse=True):
            rows.append([
                status,
                data.get("count", 0),
                f"{data.get('avg_age', 0):.0f} days",
                f"${data.get('total_cost', 0):,.0f}"
            ])
        rows.append(["", "", "", ""])

        # Estimated arrivals
        rows.append(["ESTIMATED ARRIVAL WINDOWS", ""])
        arrivals = summary.get("estimated_arrivals", {})
        for month, units in arrivals.items():
            rows.append([month, f"{units} units"])
        rows.append(["", ""])

        # Pipeline flow health
        rows.append(["PIPELINE FLOW HEALTH (30 days)", ""])
        flow = self.pipeline_analyzer.get_pipeline_flow_health(manufacturer, days=30)
        rows.append(["Orders Placed", flow.get("orders_placed", 0)])
        rows.append(["Units Received", flow.get("units_received", 0)])
        rows.append(["Flow Ratio", flow.get("flow_ratio", 0)])
        rows.append(["Health Status", flow.get("health_status", "N/A").upper()])
        rows.append(["Interpretation", flow.get("interpretation", "")])

        df = pd.DataFrame(rows, columns=["Item", "Value", "Extra", "Notes"])
        df.to_excel(writer, sheet_name="On Order", index=False)

    def _write_yoy_tab(self, writer: pd.ExcelWriter, manufacturer: str):
        """Write Year-over-Year Comparison tab."""
        rows = []
        rows.append(["YEAR-OVER-YEAR COMPARISON", ""])
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        rows.append(["", ""])

        # YTD comparison (apples-to-apples)
        yoy = self.yoy_analyzer.compare_yoy(manufacturer=manufacturer, adjust_for_locations=True, use_ytd=True)

        if "error" not in yoy:
            rows.append([f"COMPARISON: {yoy.get('comparison_type', 'YTD')}", ""])
            rows.append(["", ""])
            rows.append([f"{yoy.get('last_year', '')} Units", yoy.get("last_year_units", 0)])
            rows.append([f"{yoy.get('current_year', '')} Units", yoy.get("current_year_units", 0)])
            rows.append(["Unit Change", yoy.get("units_change", 0)])
            rows.append(["Raw Change %", yoy.get("units_change_pct_formatted", "")])
            rows.append(["", ""])

            if "current_year_locations" in yoy:
                rows.append(["LOCATION-ADJUSTED COMPARISON", ""])
                rows.append([f"{yoy.get('last_year', '')} Active Locations", yoy.get("last_year_locations", 0)])
                rows.append([f"{yoy.get('current_year', '')} Active Locations", yoy.get("current_year_locations", 0)])
                rows.append(["Location Growth", yoy.get("location_growth", 0)])
                rows.append([f"{yoy.get('last_year', '')} Sales Per Location", f"{yoy.get('last_year_per_location', 0):.1f}"])
                rows.append([f"{yoy.get('current_year', '')} Sales Per Location", f"{yoy.get('current_year_per_location', 0):.1f}"])
                rows.append(["Location-Adjusted Change %", yoy.get("adjusted_change_pct_formatted", "")])
                rows.append(["", ""])
                rows.append(["Interpretation", yoy.get("interpretation", "")])
        else:
            rows.append(["Error", yoy.get("error", "Unknown error")])
        rows.append(["", ""])

        df_summary = pd.DataFrame(rows, columns=["Metric", "Value"])
        df_summary.to_excel(writer, sheet_name="YoY Comparison", index=False)

        # Quarterly comparison on separate sheet if enough data
        quarterly = self.yoy_analyzer.compare_yoy_by_quarter(manufacturer=manufacturer)
        if not quarterly.empty:
            quarterly.to_excel(writer, sheet_name="YoY by Quarter", index=False)

        # Monthly comparison
        monthly = self.yoy_analyzer.compare_yoy_by_month(manufacturer=manufacturer)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="YoY by Month", index=False)

    def _write_market_share_tab(self, writer: pd.ExcelWriter, manufacturer: str):
        """Write Market Share Analysis tab."""
        rows = []
        rows.append(["MARKET SHARE ANALYSIS (SSI DATA)", ""])
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        rows.append(["", ""])

        # Overall market share
        share = self.market_analyzer.calculate_market_share(
            manufacturer=manufacturer,
            use_ftrv_btas=True
        )

        if "error" not in share:
            rows.append(["FTRV MARKET POSITION", ""])
            rows.append(["FTRV Sales (LTM)", share.get("ftrv_sales", 0)])
            rows.append(["Total Market (FTRV Territory)", share.get("total_market", 0)])
            rows.append(["Market Share", share.get("market_share_pct", "N/A")])
            rows.append(["Data Source", share.get("data_source", "N/A")])
            rows.append(["BTAs Analyzed", share.get("btas_used", 0)])
            rows.append(["Dealers Matched", share.get("dealers_matched", 0)])
        else:
            rows.append(["Error", share.get("error", "Unknown error")])
        rows.append(["", ""])

        # Competitive landscape
        rows.append(["TOP MANUFACTURERS IN MARKET", ""])
        landscape = self.market_analyzer.get_competitive_landscape()
        if not landscape.empty:
            for _, row in landscape.head(10).iterrows():
                mfg = row.get("Manufacturer", "")
                units = row.get("Units", 0)
                mkt_share = row.get("Market Share", 0)
                rows.append([mfg, f"{units:,} units", f"{mkt_share:.1%}"])

        df = pd.DataFrame(rows, columns=["Item", "Value", "Extra"])
        df.to_excel(writer, sheet_name="Market Share", index=False)

        # BTA breakdown on separate sheet
        bta_share = self.market_analyzer.get_market_share_by_bta(manufacturer=manufacturer)
        if not bta_share.empty:
            bta_share.to_excel(writer, sheet_name="Share by BTA", index=False)

    def _write_transparency_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Recommendation Transparency / Methodology tab."""
        rows = []
        rows.append(["RECOMMENDATION METHODOLOGY", ""])
        rows.append(["", ""])

        rows.append(["HOW RECOMMENDATIONS ARE CALCULATED:", ""])
        rows.append(["", ""])

        rows.append(["1. DEMAND VELOCITY", ""])
        rows.append([f"   • Analyzed last {self.config.lookback_months} months of retail sales", ""])
        rows.append(["   • Calculated monthly sales rate for each model", ""])
        rows.append(["   • Identified trends (increasing/stable/decreasing)", ""])
        rows.append(["", ""])

        rows.append(["2. SEASONALITY ADJUSTMENT", ""])
        rows.append(["   • Applied seasonal indices based on RV industry patterns", ""])
        rows.append(["   • Jan-Feb: Peak show season (+20-30%)", ""])
        rows.append(["   • Jul-Aug: Summer slowdown (-15-20%)", ""])
        rows.append(["", ""])

        rows.append(["3. INVENTORY POSITION", ""])
        rows.append(["   • Counted current sellable inventory", ""])
        rows.append(["   • Added pipeline/on-order units", ""])
        rows.append([f"   • Target: {self.config.target_days_supply} days supply", ""])
        rows.append(["", ""])

        rows.append(["4. GAP CALCULATION", ""])
        rows.append(["   • Gap = Target Position - Current Position", ""])
        rows.append([f"   • Added {self.config.safety_buffer_pct:.0%} safety buffer", ""])
        rows.append([f"   • Accounted for {self.config.total_lead_time}-day lead time", ""])
        rows.append(["", ""])

        rows.append(["5. URGENCY CLASSIFICATION", ""])
        rows.append([f"   • CRITICAL: Days supply < {self.config.critical_days_supply}", ""])
        rows.append([f"   • HIGH: Days supply {self.config.critical_days_supply}-{self.config.min_days_supply}", ""])
        rows.append([f"   • MEDIUM: Days supply {self.config.min_days_supply}-{self.config.target_days_supply}", ""])
        rows.append([f"   • LOW: Days supply > {self.config.target_days_supply}", ""])
        rows.append(["", ""])

        rows.append(["FORMULA:", ""])
        rows.append(["Recommended Qty = (Forecast Demand × Seasonality) - Current Position + Safety Buffer + Lead Time Buffer", ""])
        rows.append(["", ""])

        rows.append(["CONFIDENCE LEVELS:", ""])
        rows.append(["   • HIGH: 50+ sales in last 12 months", ""])
        rows.append(["   • MEDIUM: 20-49 sales in last 12 months", ""])
        rows.append(["   • LOW: 5-19 sales in last 12 months", ""])
        rows.append(["   • VERY LOW: <5 sales - use caution", ""])

        df = pd.DataFrame(rows, columns=["Description", "Notes"])
        df.to_excel(writer, sheet_name="Methodology", index=False)

    def _write_brand_licensing_tab(self, writer: pd.ExcelWriter, order_plan: Dict):
        """Write Brand Licensing tab showing licensing status by make and location."""
        manufacturer = order_plan.get("manufacturer")
        rows = []
        rows.append(["BRAND LICENSING STATUS", ""])
        rows.append(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
        rows.append(["", ""])

        try:
            # Get licensing report
            report = self.brand_licensing.generate_licensing_report(manufacturer)

            if "error" in report:
                rows.append(["Error", report.get("error")])
                df = pd.DataFrame(rows, columns=["Item", "Value"])
                df.to_excel(writer, sheet_name="Brand Licensing", index=False)
                return

            # Summary
            rows.append(["SUMMARY", ""])
            rows.append(["Total Makes", report.get("total_makes", 0)])
            rows.append(["Locations Analyzed", report.get("locations_analyzed", 0)])
            rows.append(["Fully Licensed Makes", len(report.get("fully_licensed_makes", []))])
            rows.append(["Limited Availability Makes", len(report.get("limited_availability_makes", []))])
            rows.append(["", ""])

            # Limited availability makes - important for recommendations
            limited = report.get("limited_availability_makes", [])
            if limited:
                rows.append(["MAKES WITH LIMITED LICENSING", ""])
                rows.append(["Make", "Manufacturer", "Licensed Locations", "Total Locations"])
                for item in limited[:20]:  # Top 20
                    rows.append([
                        item.get("make", ""),
                        item.get("manufacturer", ""),
                        item.get("licensed_locations", 0),
                        item.get("total_locations", 0)
                    ])
                rows.append(["", "", "", ""])

            # Check recommendations for licensing issues
            recs = order_plan.get("recommendations", [])
            if recs:
                # Add licensing status to recommendations
                recs_with_licensing = self.brand_licensing.check_recommendations_licensing(recs)

                rows.append(["RECOMMENDATION LICENSING ALERTS", ""])
                alerts = [r for r in recs_with_licensing if r.get("licensing_warning")]
                if alerts:
                    rows.append(["Make", "Manufacturer", "Warning"])
                    for rec in alerts[:15]:
                        rows.append([
                            rec.get("make", ""),
                            rec.get("manufacturer", ""),
                            rec.get("licensing_warning", "")
                        ])
                else:
                    rows.append(["No licensing alerts for recommended makes", ""])

            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="Brand Licensing", index=False, header=False)

            # Also write full licensing matrix on separate sheet
            matrix = self.brand_licensing.get_licensing_matrix(manufacturer=manufacturer)
            if not matrix.empty:
                matrix.to_excel(writer, sheet_name="Licensing Matrix", index=False)

        except Exception as e:
            rows.append(["Error loading licensing data", str(e)])
            df = pd.DataFrame(rows, columns=["Item", "Value"])
            df.to_excel(writer, sheet_name="Brand Licensing", index=False)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def generate_quick_summary(self, order_plan: Dict) -> str:
        """Generate a quick text summary of the order plan."""
        if order_plan.get("status") == "error":
            return f"ERROR: {order_plan.get('message', 'Unknown error')}"

        summary = order_plan.get("summary", {})
        inv_pos = order_plan.get("inventory_position", {})
        params = order_plan.get("parameters", {})
        redistribution = order_plan.get("redistribution", {})

        # Build period string
        if params.get("start_month") and params.get("end_month"):
            period_str = f"{params['start_month']} to {params['end_month']} ({params.get('forecast_months', 3)} months)"
        else:
            period_str = f"{params.get('forecast_months', 3)} months (rolling)"

        # Build makes info
        makes_selected = order_plan.get("makes_selected")
        if makes_selected:
            makes_str = f"{len(makes_selected)} selected"
        else:
            makes_str = "All Makes"

        text = f"""
FTRV ORDER RECOMMENDATION - QUICK SUMMARY
==========================================
Manufacturer: {order_plan.get('manufacturer', 'N/A')}
Makes: {makes_str}
Forecast Period: {period_str}
Generated: {order_plan.get('generated_at', '')}
"""

        # Add redistribution note if applicable
        if redistribution.get("excluded_count", 0) > 0:
            text += f"""
DISCONTINUED MAKES HANDLED:
- {redistribution['excluded_count']} makes excluded (discontinued/not ordered)
- Volume redistribution factor: {redistribution.get('factor', 1.0):.2f}x
- Selected makes absorb discontinued volume
"""

        text += f"""
ORDER SUMMARY:
- Total Recommended Units: {summary.get('total_recommended_units', 0):,}
- Total Estimated Cost: ${summary.get('total_estimated_cost', 0):,.0f}
- Makes Included: {summary.get('makes_recommended', summary.get('models_recommended', 0))}

CURRENT POSITION:
- Inventory: {inv_pos.get('total_inventory', 0):,} units
- Days Supply: {inv_pos.get('days_supply', 0):.0f} days
- Monthly Velocity: {inv_pos.get('monthly_sales_rate', 0):.1f} units/mo
- Health Score: {inv_pos.get('inventory_health', 'N/A')}

TOP 10 MAKE RECOMMENDATIONS (Type | Manufacturer | Make):
"""
        recs = order_plan.get("recommendations", [])[:10]
        for i, rec in enumerate(recs, 1):
            veh_type = rec.get('veh_type', '')
            mfg = rec.get('manufacturer', '')
            make = rec.get('make', '')
            text += f"  {i}. {veh_type:3} | {mfg[:15]:15} | {make[:15]:15} Qty: {rec['recommended_qty']:3d}  ({rec['urgency']})\n"

        return text

    # =========================================================================
    # CAPACITY-BASED REPORT
    # =========================================================================

    def generate_capacity_report(self, capacity_data: Dict) -> str:
        """
        Generate Excel report for capacity-based analysis.

        Args:
            capacity_data: Output from CapacityPlanner.generate_capacity_report()

        Returns:
            Path to generated Excel file
        """
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Capacity_Analysis_{date_str}.xlsx"
        output_file = self.output_path / filename

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary tabs
            self._write_capacity_summary_tab(writer, capacity_data)
            self._write_inventory_flow_tab(writer, capacity_data)  # NEW: Flow projection
            self._write_capacity_by_type_tab(writer, capacity_data)
            self._write_capacity_by_zone_tab(writer, capacity_data)
            self._write_location_gaps_tab(writer, capacity_data)

            # Unassigned pipeline
            self._write_unassigned_pipeline_tab(writer, capacity_data)

            # VIN-level assignment recommendations (NEW)
            self._write_vin_gap_fill_tab(writer, capacity_data)
            self._write_vin_overflow_tab(writer, capacity_data)

            # Summary-level assignments (kept for quick reference)
            self._write_assignment_recommendations_tab(writer, capacity_data)
            self._write_fill_recommendations_tab(writer, capacity_data)

            # Unit-level detail tabs (NEW)
            self._write_inventory_detail_tab(writer, capacity_data)
            self._write_on_order_detail_tab(writer, capacity_data)

        return str(output_file)

    def _write_capacity_summary_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Capacity Summary tab."""
        rows = []
        rows.append(["FTRV CAPACITY-BASED INVENTORY ANALYSIS", ""])
        rows.append(["Generated", data.get("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M"))])
        rows.append(["", ""])

        summary = data.get("summary", {})
        rows.append(["COMPANY-WIDE SUMMARY", ""])
        rows.append(["Total Lot Capacity (NEW)", summary.get("total_lot_capacity", 0)])
        rows.append(["Current Sellable Inventory", summary.get("total_current_inventory", 0)])
        rows.append(["Pipeline/On Order", summary.get("total_pipeline", 0)])
        rows.append(["In Service (will return)", summary.get("total_in_service", 0)])
        rows.append(["", ""])

        available = summary.get("total_current_inventory", 0) + summary.get("total_pipeline", 0)
        fill_pct = available / summary.get("total_lot_capacity", 1) * 100 if summary.get("total_lot_capacity", 0) > 0 else 0
        rows.append(["Total Available (Current + Pipeline)", available])
        rows.append(["Fill Rate", f"{fill_pct:.1f}%"])
        rows.append(["TOTAL GAP TO FILL", summary.get("total_gap", 0)])
        rows.append(["", ""])

        # Unassigned pipeline summary
        unassigned = data.get("unassigned_pipeline", {})
        rows.append(["UNASSIGNED PIPELINE", ""])
        rows.append(["Units at PDI/YARD/CORP awaiting assignment", unassigned.get("total_unassigned", 0)])
        rows.append(["", ""])

        # Key insight
        total_gaps = sum(g.get("gap", 0) for g in data.get("location_gaps", []))
        unassigned_total = unassigned.get("total_unassigned", 0)
        rows.append(["KEY INSIGHT", ""])
        if unassigned_total >= total_gaps:
            rows.append(["Status", f"SUFFICIENT - {unassigned_total:,} unassigned units can fill {total_gaps:,} gaps"])
        else:
            rows.append(["Status", f"SHORTFALL - Need {total_gaps - unassigned_total:,} more units beyond pipeline"])

        df = pd.DataFrame(rows, columns=["Item", "Value"])
        df.to_excel(writer, sheet_name="Summary", index=False)

    def _write_capacity_by_type_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Capacity by Vehicle Type tab."""
        by_type = data.get("by_type", {})

        rows = []
        for veh_type, type_data in sorted(by_type.items()):
            rows.append({
                "Veh Type": veh_type,
                "Lot Capacity": type_data.get("capacity", 0),
                "Current Inventory": type_data.get("current", 0),
                "Pipeline": type_data.get("pipeline", 0),
                "In Service": type_data.get("in_service", 0),
                "Total Available": type_data.get("current", 0) + type_data.get("pipeline", 0),
                "Gap": type_data.get("gap", 0),
                "Fill %": f"{type_data.get('fill_pct', 0):.1f}%"
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="By Type", index=False)

    def _write_capacity_by_zone_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Capacity by Zone tab."""
        by_zone = data.get("by_zone", {})

        rows = []
        for zone, zone_data in sorted(by_zone.items(), key=lambda x: x[1].get("gap", 0), reverse=True):
            available = zone_data.get("current", 0) + zone_data.get("pipeline", 0)
            capacity = zone_data.get("capacity", 0)
            fill_pct = available / capacity * 100 if capacity > 0 else 0

            rows.append({
                "Zone": zone,
                "Lot Capacity": capacity,
                "Current Inventory": zone_data.get("current", 0),
                "Pipeline": zone_data.get("pipeline", 0),
                "In Service": zone_data.get("in_service", 0),
                "Total Available": available,
                "Gap": zone_data.get("gap", 0),
                "Fill %": f"{fill_pct:.1f}%"
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="By Zone", index=False)

    def _write_location_gaps_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Location Gaps tab."""
        gaps = data.get("location_gaps", [])

        if not gaps:
            df = pd.DataFrame([["No location gaps"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Location Gaps", index=False)
            return

        rows = []
        for gap in gaps:
            brands = ", ".join(gap.get("licensed_brands", [])[:10])
            if len(gap.get("licensed_brands", [])) > 10:
                brands += f"... (+{len(gap['licensed_brands'])-10} more)"

            rows.append({
                "Location": gap.get("location", ""),
                "Zone": gap.get("zone", ""),
                "Veh Type": gap.get("veh_type", ""),
                "Lot Capacity": gap.get("capacity", 0),
                "Current Inventory": gap.get("current", 0),
                "Pipeline": gap.get("pipeline", 0),
                "In Service": gap.get("in_service", 0),
                "Gap": gap.get("gap", 0),
                "Licensed Brands": brands
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Location Gaps", index=False)

    def _write_unassigned_pipeline_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Unassigned Pipeline tab."""
        unassigned = data.get("unassigned_pipeline", {})

        rows = []
        rows.append(["UNASSIGNED PIPELINE ANALYSIS", ""])
        rows.append(["Total Unassigned Units", unassigned.get("total_unassigned", 0)])
        rows.append(["", ""])

        # By Type
        rows.append(["BY VEHICLE TYPE", ""])
        for vt, count in sorted(unassigned.get("by_type", {}).items(), key=lambda x: x[1], reverse=True):
            rows.append([vt, count])
        rows.append(["", ""])

        # By Status
        rows.append(["BY STATUS", ""])
        for status, count in sorted(unassigned.get("by_status", {}).items(), key=lambda x: x[1], reverse=True):
            rows.append([status, count])
        rows.append(["", ""])

        # By Manufacturer
        rows.append(["BY MANUFACTURER", ""])
        for mfg, count in sorted(unassigned.get("by_manufacturer", {}).items(), key=lambda x: x[1], reverse=True):
            rows.append([mfg, count])
        rows.append(["", ""])

        # By Make (top 30)
        rows.append(["BY MAKE (Top 30)", ""])
        for make, count in list(unassigned.get("by_make", {}).items())[:30]:
            rows.append([make, count])

        df = pd.DataFrame(rows, columns=["Category", "Units"])
        df.to_excel(writer, sheet_name="Unassigned Pipeline", index=False)

    def _write_assignment_recommendations_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Assignment Recommendations tab."""
        assignments = data.get("assignment_recommendations", [])

        if not assignments:
            df = pd.DataFrame([["No assignment recommendations"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Assignments", index=False)
            return

        # Detailed assignments
        rows = []
        for rec in assignments[:500]:  # Limit to 500 for Excel
            rows.append({
                "Stock#": rec.get("stock_num", ""),
                "Veh Type": rec.get("veh_type", ""),
                "Manufacturer": rec.get("manufacturer", ""),
                "Make": rec.get("make", ""),
                "Current Status": rec.get("current_status", ""),
                "Assign To Location": rec.get("recommended_location", ""),
                "Zone": rec.get("recommended_zone", ""),
                "Location Gap": rec.get("location_gap", 0),
                "Reason": rec.get("reason", "")
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Assignments", index=False)

        # Summary by location
        by_loc = data.get("assignments_by_location", {})
        if by_loc:
            summary_rows = []
            for loc, loc_data in sorted(by_loc.items(), key=lambda x: x[1]["total_units"], reverse=True):
                types_str = ", ".join([f"{t}:{c}" for t, c in sorted(loc_data["by_type"].items(), key=lambda x: x[1], reverse=True)])
                makes_str = ", ".join([f"{m}" for m, c in sorted(loc_data["by_make"].items(), key=lambda x: x[1], reverse=True)[:5]])

                summary_rows.append({
                    "Location": loc,
                    "Zone": loc_data.get("zone", ""),
                    "Total Units": loc_data.get("total_units", 0),
                    "By Type": types_str,
                    "Top Makes": makes_str
                })

            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name="Assignment Summary", index=False)

    def _write_fill_recommendations_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Fill Recommendations tab (zone/type level)."""
        fill_recs = data.get("fill_recommendations", [])

        if not fill_recs:
            df = pd.DataFrame([["No fill recommendations"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Fill Recommendations", index=False)
            return

        rows = []
        for rec in fill_recs:
            rows.append({
                "Zone": rec.get("zone", ""),
                "Veh Type": rec.get("veh_type", ""),
                "Units Needed": rec.get("total_units_needed", 0),
                "Locations with Gaps": rec.get("locations_with_gaps", 0),
                "Location List": ", ".join(rec.get("location_list", [])),
                "Available Brands": ", ".join(rec.get("available_brands", [])[:15]) +
                    (f"... (+{len(rec.get('available_brands', []))-15} more)" if len(rec.get("available_brands", [])) > 15 else ""),
                "Recommendation": rec.get("recommendation", "")
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="Fill Recommendations", index=False)

    def _write_inventory_flow_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write Inventory Flow Projection tab with inflow/outflow visualization."""
        flow = data.get("inventory_flow", {})

        if not flow:
            df = pd.DataFrame([["No flow projection available"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Inventory Flow", index=False)
            return

        rows = []
        rows.append(["INVENTORY FLOW PROJECTION", ""])
        rows.append(["Generated", flow.get("generated_at", "")])
        rows.append(["Projection Period", f"{flow.get('projection_months', 6)} months"])
        rows.append(["", ""])

        # Starting position
        start = flow.get("starting_position", {})
        rows.append(["STARTING POSITION", ""])
        rows.append(["Current Sellable Inventory", start.get("current_inventory", 0)])
        rows.append(["Current Pipeline", start.get("current_pipeline", 0)])
        rows.append(["Total Lot Capacity", start.get("total_capacity", 0)])
        rows.append(["Monthly Sales Velocity", start.get("monthly_velocity", 0)])
        rows.append(["", ""])

        # Velocity by type
        rows.append(["MONTHLY VELOCITY BY TYPE", ""])
        for vt, vel in flow.get("velocity_by_type", {}).items():
            rows.append([vt, f"{vel:.1f} units/mo"])
        rows.append(["", ""])

        # Summary
        summary = flow.get("summary", {})
        rows.append(["PROJECTION SUMMARY", ""])
        rows.append(["Total Projected Sales", summary.get("total_projected_sales", 0)])
        rows.append(["Total Pipeline Arrivals", summary.get("total_pipeline_arrivals", 0)])
        rows.append(["Total Reorder Recommended", summary.get("total_reorder_recommended", 0)])
        rows.append(["Ending Inventory", summary.get("ending_inventory", 0)])
        rows.append(["Ending Fill Rate", f"{summary.get('ending_fill_rate', 0):.1f}%"])
        rows.append(["", ""])

        # Write header section
        header_df = pd.DataFrame(rows, columns=["Item", "Value"])
        header_df.to_excel(writer, sheet_name="Inventory Flow", index=False)

        # Write monthly projections as a table below
        projections = flow.get("monthly_projections", [])
        if projections:
            proj_df = pd.DataFrame(projections)
            # Reorder columns for readability
            col_order = [
                "month", "starting_inventory", "projected_sales", "pipeline_arriving",
                "ending_inventory", "pipeline_remaining", "capacity", "fill_rate",
                "gap", "days_supply", "reorder_needed", "seasonality_factor"
            ]
            col_order = [c for c in col_order if c in proj_df.columns]
            proj_df = proj_df[col_order]

            # Rename columns
            rename_map = {
                "month": "Month",
                "starting_inventory": "Starting Inv",
                "projected_sales": "Proj. Sales",
                "pipeline_arriving": "Pipeline In",
                "ending_inventory": "Ending Inv",
                "pipeline_remaining": "Pipeline Left",
                "capacity": "Capacity",
                "fill_rate": "Fill %",
                "gap": "Gap",
                "days_supply": "Days Supply",
                "reorder_needed": "Reorder Qty",
                "seasonality_factor": "Season Idx"
            }
            proj_df = proj_df.rename(columns=rename_map)

            proj_df.to_excel(writer, sheet_name="Flow Projection", index=False)

    def _write_vin_gap_fill_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write VIN-level gap fill assignments."""
        assignments = data.get("vin_assignments_gap_fill", [])

        if not assignments:
            df = pd.DataFrame([["No VIN-level gap fill assignments"]], columns=["Message"])
            df.to_excel(writer, sheet_name="VIN Gap Fill", index=False)
            return

        rows = []
        for rec in assignments:
            rows.append({
                "Stock#": rec.get("stock_num", ""),
                "VIN": rec.get("vin", ""),
                "Year": rec.get("year", ""),
                "Manufacturer": rec.get("manufacturer", ""),
                "Make": rec.get("make", ""),
                "Model": rec.get("model", ""),
                "Veh Type": rec.get("veh_type", ""),
                "Current Status": rec.get("current_status", ""),
                "Retail Price": rec.get("retail_price", 0),
                "Total Cost": rec.get("total_cost", 0),
                "ASSIGN TO": rec.get("assigned_location", ""),
                "Zone": rec.get("assigned_zone", ""),
                "Gap Being Filled": rec.get("fills_gap_of", 0),
                "Gap After": rec.get("remaining_gap_after", 0),
                "Reason": rec.get("reason", "")
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="VIN Gap Fill", index=False)

    def _write_vin_overflow_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write VIN-level overflow/remaining yard allocation."""
        overflow = data.get("vin_assignments_overflow", [])

        if not overflow:
            df = pd.DataFrame([["No overflow units - all units assigned to gaps"]], columns=["Message"])
            df.to_excel(writer, sheet_name="VIN Overflow", index=False)
            return

        rows = []
        rows_header = [["REMAINING YARD UNITS - SUGGESTED ALLOCATION", ""]]
        rows_header.append(["These units could not fill specific gaps (licensing or type mismatch)", ""])
        rows_header.append(["Suggested allocations based on capacity and velocity", ""])
        rows_header.append(["Total Overflow Units", len(overflow)])
        rows_header.append(["", ""])

        header_df = pd.DataFrame(rows_header, columns=["Info", "Value"])
        header_df.to_excel(writer, sheet_name="VIN Overflow", index=False)

        for rec in overflow:
            rows.append({
                "Stock#": rec.get("stock_num", ""),
                "VIN": rec.get("vin", ""),
                "Year": rec.get("year", ""),
                "Manufacturer": rec.get("manufacturer", ""),
                "Make": rec.get("make", ""),
                "Model": rec.get("model", ""),
                "Veh Type": rec.get("veh_type", ""),
                "Current Status": rec.get("current_status", ""),
                "Retail Price": rec.get("retail_price", 0),
                "Total Cost": rec.get("total_cost", 0),
                "SUGGESTED LOCATION": rec.get("suggested_location", "TBD"),
                "Zone": rec.get("suggested_zone", ""),
                "Reason": rec.get("reason", "")
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="VIN Overflow", index=False, startrow=6)

    def _write_inventory_detail_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write current inventory detail with VIN/Model."""
        inv_detail = data.get("inventory_detail")

        if inv_detail is None or (isinstance(inv_detail, pd.DataFrame) and inv_detail.empty):
            df = pd.DataFrame([["No inventory detail available"]], columns=["Message"])
            df.to_excel(writer, sheet_name="Inventory Detail", index=False)
            return

        # Sort by location then type
        sort_cols = [c for c in ["PC", "Veh Type", "Make", "Model"] if c in inv_detail.columns]
        if sort_cols:
            inv_detail = inv_detail.sort_values(sort_cols)

        inv_detail.to_excel(writer, sheet_name="Inventory Detail", index=False)

    def _write_on_order_detail_tab(self, writer: pd.ExcelWriter, data: Dict):
        """Write on-order detail with VIN/Model."""
        oo_detail = data.get("on_order_detail")

        if oo_detail is None or (isinstance(oo_detail, pd.DataFrame) and oo_detail.empty):
            df = pd.DataFrame([["No on-order detail available"]], columns=["Message"])
            df.to_excel(writer, sheet_name="On Order Detail", index=False)
            return

        # Sort by location then type
        sort_cols = [c for c in ["PC", "Veh Type", "Make", "Model"] if c in oo_detail.columns]
        if sort_cols:
            oo_detail = oo_detail.sort_values(sort_cols)

        oo_detail.to_excel(writer, sheet_name="On Order Detail", index=False)
