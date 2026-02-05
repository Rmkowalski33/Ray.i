"""
Recommendation Transparency Module
Provides detailed explanations for how each order recommendation is calculated.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .config import Config, default_config
from .data_loader import DataLoader


@dataclass
class RecommendationExplanation:
    """Detailed breakdown of a single recommendation."""

    # Identification
    make: str
    model: str
    veh_type: str

    # Recommendation
    recommended_qty: int
    urgency: str
    confidence: str

    # Current Position
    current_inventory: int
    pipeline_units: int
    total_position: int

    # Demand Analysis
    ltm_sales: int
    monthly_velocity: float
    avg_days_to_sell: float
    trend: str
    trend_pct: float

    # Seasonality Adjustments
    seasonality_factors: Dict[str, float] = field(default_factory=dict)
    forecast_by_month: Dict[str, float] = field(default_factory=dict)
    total_forecast_demand: float = 0

    # Gap Calculation
    target_days_supply: int = 90
    target_position: int = 0
    position_gap: int = 0
    safety_buffer: float = 0
    lead_time_buffer: int = 0

    # Financial Impact
    avg_unit_cost: float = 0
    estimated_total_cost: float = 0
    avg_front_end: float = 0
    margin_pct: float = 0
    est_holding_cost: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "make": self.make,
            "model": self.model,
            "veh_type": self.veh_type,
            "recommended_qty": self.recommended_qty,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "current_position": {
                "current_inventory": self.current_inventory,
                "pipeline_units": self.pipeline_units,
                "total_position": self.total_position
            },
            "demand_analysis": {
                "ltm_sales": self.ltm_sales,
                "monthly_velocity": self.monthly_velocity,
                "avg_days_to_sell": self.avg_days_to_sell,
                "trend": self.trend,
                "trend_pct": self.trend_pct
            },
            "seasonality": {
                "factors": self.seasonality_factors,
                "forecast_by_month": self.forecast_by_month,
                "total_forecast_demand": self.total_forecast_demand
            },
            "gap_calculation": {
                "target_days_supply": self.target_days_supply,
                "target_position": self.target_position,
                "position_gap": self.position_gap,
                "safety_buffer": self.safety_buffer,
                "lead_time_buffer": self.lead_time_buffer
            },
            "financial_impact": {
                "avg_unit_cost": self.avg_unit_cost,
                "estimated_total_cost": self.estimated_total_cost,
                "avg_front_end": self.avg_front_end,
                "margin_pct": self.margin_pct,
                "est_holding_cost": self.est_holding_cost
            }
        }

    def to_text(self) -> str:
        """Generate human-readable explanation."""
        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"MODEL: {self.make} {self.model}")
        lines.append(f"Recommended Qty: {self.recommended_qty} units")
        lines.append(f"Urgency: {self.urgency} | Confidence: {self.confidence}")
        lines.append("")
        lines.append("WHY THIS RECOMMENDATION:")
        lines.append("-" * 40)

        # Current Position
        lines.append("")
        lines.append("CURRENT POSITION:")
        lines.append(f"  • Current Inventory: {self.current_inventory} units")
        lines.append(f"  • On Order (Pipeline): {self.pipeline_units} units")
        lines.append(f"  • Total Position: {self.total_position} units")

        # Demand Analysis
        lines.append("")
        lines.append("DEMAND ANALYSIS:")
        lines.append(f"  • Last 12 Months Sales: {self.ltm_sales} units ({self.monthly_velocity:.1f}/month)")
        lines.append(f"  • Avg Days to Sell: {self.avg_days_to_sell:.0f} days")
        lines.append(f"  • Trend: {self.trend.upper()} ({self.trend_pct:+.1%} YoY)")

        # Seasonality
        if self.forecast_by_month:
            lines.append("")
            lines.append("SEASONALITY ADJUSTMENTS:")
            for month, forecast in self.forecast_by_month.items():
                factor = self.seasonality_factors.get(month, 1.0)
                lines.append(f"  • {month}: {factor:.0%} index → {forecast:.1f} expected units")
            lines.append(f"  • Total Forecast ({len(self.forecast_by_month)} months): {self.total_forecast_demand:.1f} units")

        # Gap Calculation
        lines.append("")
        lines.append("GAP CALCULATION:")
        lines.append(f"  • Target Days Supply: {self.target_days_supply} days")
        lines.append(f"  • Target Position: {self.target_position} units")
        lines.append(f"  • Current Position: {self.total_position} units")
        lines.append(f"  • Gap: {self.position_gap} units")
        lines.append(f"  • Safety Buffer ({default_config.safety_buffer_pct:.0%}): {self.safety_buffer:.1f} units")
        if self.lead_time_buffer > 0:
            lines.append(f"  • Lead Time Buffer: {self.lead_time_buffer} units")
        lines.append(f"  → RECOMMENDED: {self.recommended_qty} units")

        # Financial Impact
        lines.append("")
        lines.append("FINANCIAL IMPACT:")
        lines.append(f"  • Est. Unit Cost: ${self.avg_unit_cost:,.0f}")
        lines.append(f"  • Est. Total Cost: ${self.estimated_total_cost:,.0f}")
        if self.avg_front_end > 0:
            lines.append(f"  • Avg Front-End: ${self.avg_front_end:,.0f}/unit")
            lines.append(f"  • Margin: {self.margin_pct:.1%}")
        if self.est_holding_cost > 0:
            lines.append(f"  • Est. Holding Cost (at {self.avg_days_to_sell:.0f} days): ${self.est_holding_cost:,.0f}")

        lines.append("")
        lines.append(f"{'='*60}")

        return "\n".join(lines)


class TransparencyGenerator:
    """Generate detailed explanations for order recommendations."""

    def __init__(self, data_loader: DataLoader = None, config: Config = None):
        self.config = config or default_config
        self.data_loader = data_loader or DataLoader(config=self.config)

    def explain_recommendation(
        self,
        make: str,
        model: str,
        manufacturer: str,
        forecast_months: int = None
    ) -> RecommendationExplanation:
        """
        Generate detailed explanation for a single model recommendation.

        Args:
            make: Make name (e.g., "IMPRESSION")
            model: Model name (e.g., "360MYR")
            manufacturer: Manufacturer name
            forecast_months: Forecast period in months

        Returns:
            RecommendationExplanation with full breakdown
        """
        forecast_months = forecast_months or self.config.forecast_months
        inventory = self.data_loader.load_current_inventory()
        retail = self.data_loader.load_retail_history()
        on_order = self.data_loader.load_on_order()

        # Filter to this model
        make_upper = make.upper()
        model_upper = model.upper()
        mfg_upper = manufacturer.upper()

        inv_mask = (
            (inventory["Manufacturer"].str.upper() == mfg_upper) &
            (inventory["Make"].str.upper() == make_upper) &
            (inventory["Model"].str.upper() == model_upper)
        )
        if "Status Category" in inventory.columns:
            inv_mask &= inventory["Status Category"] == "Sellable"

        model_inventory = inventory[inv_mask]

        # Retail history
        retail_mask = (
            (retail["Manufacturer"].str.upper() == mfg_upper) &
            (retail["Make"].str.upper() == make_upper) &
            (retail["Model"].str.upper() == model_upper)
        )
        cutoff_date = datetime.now() - timedelta(days=365)
        if "Sold Date" in retail.columns:
            retail_mask &= retail["Sold Date"] >= cutoff_date

        model_retail = retail[retail_mask]

        # On order
        pipe_mask = pd.Series(False, index=on_order.index) if not on_order.empty else pd.Series(dtype=bool)
        if not on_order.empty:
            if "Manufacturer" in on_order.columns:
                pipe_mask = on_order["Manufacturer"].str.upper() == mfg_upper
            if "Make" in on_order.columns:
                pipe_mask &= on_order["Make"].str.upper() == make_upper
            if "Model" in on_order.columns:
                pipe_mask &= on_order["Model"].str.upper() == model_upper

        model_pipeline = on_order[pipe_mask] if not on_order.empty else pd.DataFrame()

        # Get vehicle type
        veh_type = "Unknown"
        if "Veh Type" in model_inventory.columns and len(model_inventory) > 0:
            veh_type = model_inventory["Veh Type"].iloc[0]
        elif "Veh Type" in model_retail.columns and len(model_retail) > 0:
            veh_type = model_retail["Veh Type"].iloc[0]

        # === Current Position ===
        current_inventory = len(model_inventory)
        pipeline_units = len(model_pipeline)
        total_position = current_inventory + pipeline_units

        # === Demand Analysis ===
        ltm_sales = len(model_retail)
        monthly_velocity = ltm_sales / 12 if ltm_sales > 0 else 0

        avg_days_to_sell = 0
        if "Age" in model_retail.columns and len(model_retail) > 0:
            avg_days_to_sell = model_retail["Age"].mean()

        # Trend calculation
        trend = "stable"
        trend_pct = 0.0
        if "Sold Date" in model_retail.columns and len(model_retail) >= 4:
            model_retail_sorted = model_retail.sort_values("Sold Date")
            mid = len(model_retail_sorted) // 2
            first_half = len(model_retail_sorted.iloc[:mid])
            second_half = len(model_retail_sorted.iloc[mid:])
            if first_half > 0:
                trend_pct = (second_half - first_half) / first_half
                if trend_pct > 0.1:
                    trend = "increasing"
                elif trend_pct < -0.1:
                    trend = "decreasing"

        # === Seasonality ===
        now = datetime.now()
        seasonality_factors = {}
        forecast_by_month = {}
        total_forecast = 0

        for i in range(forecast_months):
            future_month = now + timedelta(days=30 * i)
            month_num = future_month.month
            month_name = future_month.strftime("%b %Y")

            factor = self.config.get_seasonality_factor(month_num)
            seasonality_factors[month_name] = factor

            monthly_forecast = monthly_velocity * factor
            forecast_by_month[month_name] = monthly_forecast
            total_forecast += monthly_forecast

        # === Gap Calculation ===
        target_days_supply = self.config.target_days_supply

        # Target position = forecast demand for target days supply
        if monthly_velocity > 0:
            target_position = int((target_days_supply / 30) * monthly_velocity * 1.1)  # Include buffer
        else:
            target_position = 0

        position_gap = max(0, target_position - total_position)
        safety_buffer = position_gap * self.config.safety_buffer_pct

        # Lead time consideration - need to order enough to cover lead time period
        lead_time_days = self.config.total_lead_time
        lead_time_buffer = 0
        if monthly_velocity > 0:
            lead_time_buffer = int((lead_time_days / 30) * monthly_velocity * 0.5)

        recommended_qty = max(0, int(position_gap + safety_buffer + lead_time_buffer))

        # === Urgency ===
        days_supply = (total_position / (monthly_velocity / 30)) if monthly_velocity > 0 else 999
        if days_supply < self.config.critical_days_supply:
            urgency = "CRITICAL"
        elif days_supply < self.config.min_days_supply:
            urgency = "HIGH"
        elif days_supply < self.config.target_days_supply:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # === Confidence ===
        if ltm_sales >= 50:
            confidence = "HIGH"
        elif ltm_sales >= 20:
            confidence = "MEDIUM"
        elif ltm_sales >= 5:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"

        # === Financial Impact ===
        avg_unit_cost = 0
        if "Total Cost" in model_retail.columns and len(model_retail) > 0:
            avg_unit_cost = model_retail["Total Cost"].mean()
        elif "Total Cost" in model_inventory.columns and len(model_inventory) > 0:
            avg_unit_cost = model_inventory["Total Cost"].mean()

        estimated_total_cost = avg_unit_cost * recommended_qty

        avg_front_end = 0
        if "Deal FrontEnd" in model_retail.columns and len(model_retail) > 0:
            avg_front_end = model_retail["Deal FrontEnd"].mean()

        margin_pct = 0
        if "Retail Price" in model_retail.columns and "Total Cost" in model_retail.columns:
            valid_retail = model_retail[(model_retail["Retail Price"] > 0) & (model_retail["Total Cost"] > 0)]
            if len(valid_retail) > 0:
                margin_pct = ((valid_retail["Retail Price"] - valid_retail["Total Cost"]) / valid_retail["Retail Price"]).mean()

        est_holding_cost = 0
        if avg_unit_cost > 0 and avg_days_to_sell > 0:
            est_holding_cost = self.config.calculate_holding_cost(avg_unit_cost * recommended_qty, avg_days_to_sell)

        return RecommendationExplanation(
            make=make,
            model=model,
            veh_type=veh_type,
            recommended_qty=recommended_qty,
            urgency=urgency,
            confidence=confidence,
            current_inventory=current_inventory,
            pipeline_units=pipeline_units,
            total_position=total_position,
            ltm_sales=ltm_sales,
            monthly_velocity=round(monthly_velocity, 2),
            avg_days_to_sell=round(avg_days_to_sell, 1),
            trend=trend,
            trend_pct=round(trend_pct, 3),
            seasonality_factors=seasonality_factors,
            forecast_by_month=forecast_by_month,
            total_forecast_demand=round(total_forecast, 1),
            target_days_supply=target_days_supply,
            target_position=target_position,
            position_gap=position_gap,
            safety_buffer=round(safety_buffer, 1),
            lead_time_buffer=lead_time_buffer,
            avg_unit_cost=round(avg_unit_cost, 2),
            estimated_total_cost=round(estimated_total_cost, 2),
            avg_front_end=round(avg_front_end, 2),
            margin_pct=round(margin_pct, 4),
            est_holding_cost=round(est_holding_cost, 2)
        )

    def explain_all_recommendations(
        self,
        recommendations: List[Dict],
        manufacturer: str,
        forecast_months: int = None
    ) -> List[RecommendationExplanation]:
        """
        Generate explanations for all recommendations.

        Args:
            recommendations: List of recommendation dicts from RecommendationEngine
            manufacturer: Manufacturer name
            forecast_months: Forecast period

        Returns:
            List of RecommendationExplanation objects
        """
        explanations = []
        for rec in recommendations:
            make = rec.get("make", "")
            model = rec.get("model", "")
            if make and model:
                explanation = self.explain_recommendation(
                    make=make,
                    model=model,
                    manufacturer=manufacturer,
                    forecast_months=forecast_months
                )
                explanations.append(explanation)
        return explanations

    def generate_summary_explanation(
        self,
        order_plan: Dict
    ) -> str:
        """
        Generate a summary explanation of the overall order plan.

        Args:
            order_plan: Output from RecommendationEngine.generate_order_plan()

        Returns:
            Human-readable summary text
        """
        if order_plan.get("status") == "error":
            return f"Error: {order_plan.get('message', 'Unknown error')}"

        lines = []
        lines.append("=" * 70)
        lines.append("ORDER RECOMMENDATION METHODOLOGY")
        lines.append("=" * 70)
        lines.append("")

        # Explain the approach
        lines.append("HOW RECOMMENDATIONS ARE CALCULATED:")
        lines.append("-" * 40)
        lines.append("")
        lines.append("1. DEMAND VELOCITY")
        lines.append(f"   • Analyzed last {self.config.lookback_months} months of retail sales")
        lines.append("   • Calculated monthly sales rate for each model")
        lines.append("   • Identified trends (increasing/stable/decreasing)")
        lines.append("")

        lines.append("2. SEASONALITY ADJUSTMENT")
        lines.append("   • Applied seasonal indices based on RV industry patterns")
        lines.append("   • Jan-Feb: Peak show season (+20-30%)")
        lines.append("   • Jul-Aug: Summer slowdown (-15-20%)")
        lines.append("   • Adjusted forecast demand for each month")
        lines.append("")

        lines.append("3. INVENTORY POSITION")
        lines.append("   • Counted current sellable inventory")
        lines.append("   • Added pipeline/on-order units")
        lines.append(f"   • Target: {self.config.target_days_supply} days supply")
        lines.append("")

        lines.append("4. GAP CALCULATION")
        lines.append("   • Gap = Target Position - Current Position")
        lines.append(f"   • Added {self.config.safety_buffer_pct:.0%} safety buffer")
        lines.append(f"   • Accounted for {self.config.total_lead_time}-day lead time")
        lines.append("")

        lines.append("5. PRIORITIZATION")
        lines.append("   • CRITICAL: Days supply < 45")
        lines.append("   • HIGH: Days supply 45-60")
        lines.append("   • MEDIUM: Days supply 60-90")
        lines.append("   • LOW: Days supply > 90")
        lines.append("")

        # Summary stats
        summary = order_plan.get("summary", {})
        params = order_plan.get("parameters", {})

        lines.append("=" * 70)
        lines.append("THIS ORDER PLAN:")
        lines.append("-" * 40)
        lines.append(f"Manufacturer: {order_plan.get('manufacturer', 'N/A')}")
        lines.append(f"Forecast Period: {params.get('forecast_months', 3)} months")
        lines.append(f"Total Recommended Units: {summary.get('total_recommended_units', 0):,}")
        lines.append(f"Total Estimated Cost: ${summary.get('total_estimated_cost', 0):,.0f}")
        lines.append(f"Models Included: {summary.get('models_recommended', 0)}")
        lines.append("")

        # Urgency breakdown
        by_urgency = order_plan.get("by_urgency", {})
        if by_urgency:
            lines.append("URGENCY BREAKDOWN:")
            for urgency in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if urgency in by_urgency:
                    data = by_urgency[urgency]
                    lines.append(f"  • {urgency}: {data.get('total_units', 0)} units (${data.get('total_cost', 0):,.0f})")
        lines.append("")

        # Data quality note
        inv_pos = order_plan.get("inventory_position", {})
        lines.append("DATA QUALITY:")
        sample_size = inv_pos.get("monthly_sales_rate", 0) * 12
        if sample_size >= 100:
            lines.append("  • HIGH confidence - substantial sales history")
        elif sample_size >= 50:
            lines.append("  • MEDIUM confidence - adequate sales history")
        else:
            lines.append("  • LOW confidence - limited sales data, use caution")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
