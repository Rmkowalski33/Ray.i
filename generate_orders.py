#!/usr/bin/env python3
"""
FTRV Order Recommendation Generator
====================================

Generates intelligent order recommendations for RV manufacturers
based on demand analysis, inventory position, and market data.

Usage:
    # GUI Mode (default)
    python generate_orders.py

    # CLI Mode - Generate order plan
    python generate_orders.py --cli --manufacturer "FOREST RIVER"
    python generate_orders.py --cli --manufacturer "KEYSTONE" --period 6
    python generate_orders.py --cli --manufacturer "GULF STREAM" --budget 500000

    # List available manufacturers
    python generate_orders.py --cli --list-manufacturers

    # List zones
    python generate_orders.py --cli --list-zones

Examples:
    # 3-month order plan for Forest River
    python generate_orders.py --cli -m "FOREST RIVER"

    # 6-month plan for Travel Trailers only
    python generate_orders.py --cli -m "KEYSTONE" --period 6 --veh-type TT

    # Budget-constrained order
    python generate_orders.py --cli -m "GULF STREAM" --budget 1000000

    # Location-specific
    python generate_orders.py --cli -m "WINNEBAGO" --locations AMA,DAL,HOU
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from order_engine.config import Config, default_config
from order_engine.data_loader import DataLoader
from order_engine.recommendation_engine import RecommendationEngine
from order_engine.report_generator import ReportGenerator


def list_manufacturers():
    """List manufacturers with sales history."""
    print("\nLoading data...")
    engine = RecommendationEngine()
    manufacturers = engine.list_manufacturers()

    print("\nManufacturers with sales history:")
    print("=" * 60)
    print(f"{'Manufacturer':<35} {'Units Sold':>12} {'Total Front-End':>15}")
    print("-" * 60)

    for mfg in manufacturers:
        front_end = mfg.get('Total Front-End', 0)
        front_end_str = f"${front_end:,.0f}" if front_end > 0 else "N/A"
        print(f"{mfg['Manufacturer']:<35} {mfg['Units Sold']:>12,} {front_end_str:>15}")

    print(f"\nTotal: {len(manufacturers)} manufacturers shown")


def list_zones():
    """List available zones."""
    print("\nLoading zones...")
    data_loader = DataLoader()
    zones = data_loader.get_zones()

    print("\nAvailable Zones:")
    print("=" * 40)
    for zone in zones:
        locations = data_loader.get_locations_in_zone(zone)
        print(f"  {zone}: {len(locations)} locations")


def list_veh_types():
    """List available vehicle types."""
    print("\nLoading vehicle types...")
    data_loader = DataLoader()
    veh_types = data_loader.get_veh_types()

    print("\nAvailable Vehicle Types:")
    print("=" * 30)
    for vt in veh_types:
        print(f"  {vt}")


def parse_month(month_str: str):
    """Parse month string to datetime."""
    if not month_str:
        return None

    from dateutil import parser as date_parser
    try:
        # Try parsing "Mar 2026" or "2026-03" or "March 2026"
        dt = date_parser.parse(month_str)
        return dt.replace(day=1)
    except Exception:
        return None


def generate_order_plan(
    manufacturer: str,
    make: str = None,
    period: int = None,
    budget: float = None,
    units: int = None,
    veh_types: list = None,
    locations: list = None,
    zones: list = None,
    start_month: str = None,
    end_month: str = None
):
    """Generate an order plan for a manufacturer (at make level)."""
    from dateutil.relativedelta import relativedelta

    # Parse calendar months if provided
    start_dt = parse_month(start_month)
    end_dt = parse_month(end_month)

    # Calculate period from calendar months if both provided
    if start_dt and end_dt:
        months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
        period = max(1, months_diff)
        period_desc = f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')} ({period} months)"
    elif start_dt:
        period = period or 3
        period_desc = f"{start_dt.strftime('%b %Y')} + {period} months"
    else:
        period = period or 3
        period_desc = f"Next {period} months (rolling)"

    print(f"\n{'=' * 60}")
    print("FTRV ORDER RECOMMENDATION GENERATOR")
    print("(Make-Level Orders: Type → Manufacturer → Make)")
    print(f"{'=' * 60}")
    print(f"Manufacturer: {manufacturer}")
    if make:
        print(f"Make: {make}")
    print(f"Forecast Period: {period_desc}")
    if budget:
        print(f"Budget Constraint: ${budget:,.0f}")
    if units:
        print(f"Units Constraint: {units}")
    if veh_types:
        print(f"Vehicle Types: {', '.join(veh_types)}")
    if locations:
        print(f"Locations: {', '.join(locations)}")
    if zones:
        print(f"Zones: {', '.join(zones)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create config with filters
    config = default_config
    if locations:
        config = config.with_filters(locations=locations)
    if zones:
        config = config.with_filters(zones=zones)
    if period:
        config = config.with_filters(forecast_months=period)

    # Initialize components
    engine = RecommendationEngine(config=config)
    report_gen = ReportGenerator(config=config)

    try:
        # Generate order plan at MAKE level
        print("Analyzing demand and inventory position (make-level)...")
        order_plan = engine.generate_order_plan(
            manufacturer=manufacturer,
            make=make,
            budget=budget,
            target_units=units,
            forecast_months=period,
            veh_types=veh_types
        )

        if order_plan.get("status") == "error":
            print(f"\nError: {order_plan.get('message', 'Unknown error')}")
            return 1

        # Generate report
        print("Generating Excel report...")
        output_file = report_gen.generate_order_plan_report(order_plan)
        print(f"\nReport generated: {output_file}")

        # Print summary
        summary = order_plan.get("summary", {})
        inv_pos = order_plan.get("inventory_position", {})

        print("\n" + "-" * 60)
        print("ORDER PLAN SUMMARY")
        print("-" * 60)
        print(f"Total Recommended Units: {summary.get('total_recommended_units', 0):,}")
        print(f"Total Estimated Cost: ${summary.get('total_estimated_cost', 0):,.0f}")
        print(f"Models Recommended: {summary.get('models_recommended', 0)}")

        if summary.get("budget_used"):
            print(f"Budget Used: {summary['budget_used']:.1%}")

        print("\n" + "-" * 60)
        print("CURRENT INVENTORY POSITION")
        print("-" * 60)
        print(f"Total Inventory: {inv_pos.get('total_inventory', 0):,} units")
        print(f"Days Supply: {inv_pos.get('days_supply', 0):.0f} days")
        print(f"Monthly Velocity: {inv_pos.get('monthly_sales_rate', 0):.1f} units/mo")
        print(f"Inventory Health: {inv_pos.get('inventory_health', 'N/A')}")

        # Print urgency breakdown
        by_urgency = order_plan.get("by_urgency", {})
        print("\n" + "-" * 60)
        print("URGENCY BREAKDOWN")
        print("-" * 60)
        for urgency in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if urgency in by_urgency and by_urgency[urgency]["total_units"] > 0:
                data = by_urgency[urgency]
                print(f"  {urgency:10} {data['total_units']:5,} units  ${data['total_cost']:>12,.0f}")

        # Print top recommendations
        recs = order_plan.get("recommendations", [])
        if recs:
            print("\n" + "-" * 60)
            print("TOP 15 RECOMMENDATIONS")
            print("-" * 60)
            print(f"{'Make':<15} {'Model':<20} {'Qty':>5} {'Urgency':<10} {'Est Cost':>12}")
            print("-" * 60)
            for rec in recs[:15]:
                print(f"{rec['make']:<15} {rec['model']:<20} {rec['recommended_qty']:>5} "
                      f"{rec['urgency']:<10} ${rec['estimated_total_cost']:>10,.0f}")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"\nError during order plan generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_capacity_analysis(veh_types: list = None, zones: list = None):
    """
    Run capacity-based analysis.

    Top-down approach:
    1. What do we NEED company-wide?
    2. Where do we need it (zone/location)?
    3. What's available to fill it (licensed brands)?
    """
    from order_engine.capacity_planner import CapacityPlanner

    print(f"\n{'=' * 70}")
    print("FTRV CAPACITY-BASED INVENTORY ANALYSIS")
    print("(Top-Down: What do we need? -> Who can supply it?)")
    print(f"{'=' * 70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        planner = CapacityPlanner()
        report = planner.generate_capacity_report()

        # Summary
        summary = report.get("summary", {})
        print("COMPANY-WIDE SUMMARY")
        print("-" * 50)
        print(f"Total Lot Capacity (NEW):     {summary.get('total_lot_capacity', 0):,} units")
        print(f"Current Inventory:            {summary.get('total_current_inventory', 0):,} units")
        print(f"Pipeline/On Order:            {summary.get('total_pipeline', 0):,} units")
        print(f"In Service (will return):     {summary.get('total_in_service', 0):,} units")
        print(f"TOTAL GAP TO FILL:            {summary.get('total_gap', 0):,} units")
        print()

        # By Type
        print("NEEDS BY VEHICLE TYPE")
        print("-" * 50)
        print(f"{'Type':<10} {'Capacity':>10} {'Current':>10} {'Pipeline':>10} {'Gap':>10} {'Fill %':>10}")
        print("-" * 60)
        for veh_type, data in report.get("by_type", {}).items():
            if veh_types and veh_type not in veh_types:
                continue
            print(f"{veh_type:<10} {data['capacity']:>10,} {data['current']:>10,} "
                  f"{data['pipeline']:>10,} {data['gap']:>10,} {data['fill_pct']:>9.1f}%")
        print()

        # By Zone
        print("NEEDS BY ZONE")
        print("-" * 50)
        print(f"{'Zone':<20} {'Capacity':>10} {'Current':>10} {'Pipeline':>10} {'Gap':>10}")
        print("-" * 60)
        for zone, data in sorted(report.get("by_zone", {}).items(), key=lambda x: x[1].get("gap", 0), reverse=True):
            if zones and zone not in zones:
                continue
            print(f"{zone:<20} {data['capacity']:>10,} {data['current']:>10,} "
                  f"{data['pipeline']:>10,} {data['gap']:>10,}")
        print()

        # Top Location Gaps
        print("TOP 15 LOCATION GAPS (Largest Shortages)")
        print("-" * 70)
        print(f"{'Location':<10} {'Zone':<15} {'Type':<5} {'Gap':>8} {'Licensed Brands':<30}")
        print("-" * 70)
        for gap in report.get("location_gaps", [])[:15]:
            brands = ", ".join(gap.get("licensed_brands", [])[:5])
            if len(gap.get("licensed_brands", [])) > 5:
                brands += "..."
            print(f"{gap['location']:<10} {gap['zone']:<15} {gap['veh_type']:<5} "
                  f"{gap['gap']:>8,} {brands:<30}")
        print()

        # Fill Recommendations
        print("FILL RECOMMENDATIONS (by Zone & Type)")
        print("-" * 70)
        for rec in report.get("fill_recommendations", [])[:10]:
            print(f"\n{rec['recommendation']}")
            print(f"   Locations: {', '.join(rec['location_list'][:5])}" +
                  (f"... (+{len(rec['location_list'])-5} more)" if len(rec['location_list']) > 5 else ""))
            print(f"   Available brands: {', '.join(rec['available_brands'][:8])}" +
                  (f"... (+{len(rec['available_brands'])-8} more)" if len(rec['available_brands']) > 8 else ""))

        # Unassigned Pipeline Analysis
        print("\n" + "=" * 70)
        print("UNASSIGNED PIPELINE ANALYSIS")
        print("(Units at PDI/YARD/CORP awaiting location assignment)")
        print("-" * 70)
        unassigned = report.get("unassigned_pipeline", {})
        print(f"Total Unassigned Units: {unassigned.get('total_unassigned', 0):,}")
        print()

        if unassigned.get("by_type"):
            print("By Vehicle Type:")
            for vt, count in sorted(unassigned["by_type"].items(), key=lambda x: x[1], reverse=True):
                print(f"   {vt:<10} {count:>6,} units")
            print()

        if unassigned.get("by_status"):
            print("By Status:")
            for status, count in sorted(unassigned["by_status"].items(), key=lambda x: x[1], reverse=True)[:8]:
                print(f"   {status:<25} {count:>6,} units")
            print()

        if unassigned.get("by_manufacturer"):
            print("By Manufacturer (top 10):")
            for mfg, count in sorted(unassigned["by_manufacturer"].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {mfg:<25} {count:>6,} units")

        # Assignment Recommendations
        print("\n" + "=" * 70)
        print("LOCATION ASSIGNMENT RECOMMENDATIONS")
        print("(Where to send unassigned pipeline units)")
        print("-" * 70)
        assignments_by_loc = report.get("assignments_by_location", {})

        if assignments_by_loc:
            print(f"{'Location':<10} {'Zone':<18} {'Units':>8} {'Top Types':<20} {'Top Makes'}")
            print("-" * 90)
            for loc, data in sorted(assignments_by_loc.items(), key=lambda x: x[1]["total_units"], reverse=True)[:15]:
                top_types = ", ".join([f"{t}:{c}" for t, c in sorted(data["by_type"].items(), key=lambda x: x[1], reverse=True)[:3]])
                top_makes = ", ".join([f"{m[:12]}" for m, c in sorted(data["by_make"].items(), key=lambda x: x[1], reverse=True)[:3]])
                print(f"{loc:<10} {data['zone']:<18} {data['total_units']:>8,} {top_types:<20} {top_makes}")

            total_assigned = sum(d["total_units"] for d in assignments_by_loc.values())
            total_gaps = sum(g["gap"] for g in report.get("location_gaps", []))
            print("-" * 90)
            print(f"{'TOTAL':<10} {'':<18} {total_assigned:>8,} units can be assigned to fill {total_gaps:,} unit gaps")
        else:
            print("No assignment recommendations (no unassigned units or no gaps)")

        # Generate Excel report
        print("\n" + "=" * 70)
        print("Generating Excel report...")
        try:
            from order_engine.report_generator import ReportGenerator
            report_gen = ReportGenerator()
            output_file = report_gen.generate_capacity_report(report)
            print(f"Report saved to: {output_file}")
        except Exception as e:
            print(f"Could not generate Excel report: {e}")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"\nError during capacity analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_reconciliation(planning_months: int = 0, target_fill_pct: float = 0.90,
                       manufacturer: str = None):
    """
    Run reconciliation analysis.

    Bridges capacity-based needs with manufacturer allocations:
    1. Get total needs from capacity analysis (or projected needs in planning mode)
    2. Calculate fair share by manufacturer (based on velocity + licensing)
    3. Apply user adjustments from order_adjustments.yaml
    4. Generate final allocation recommendations

    When manufacturer is specified, capacity is filtered to only licensed locations.
    """
    from order_engine.reconciliation import ReconciliationEngine, ReconciliationReportGenerator

    print(f"\n{'=' * 70}")
    print("FTRV ORDER RECONCILIATION ANALYSIS")
    if manufacturer:
        print(f"MANUFACTURER FILTER: {manufacturer}")
        print("(Capacity filtered to licensed locations only)")
    if planning_months > 0:
        print(f"PLANNING MODE: {planning_months}-Month Forward Projection")
    print("(Allocating Capacity Needs Across Manufacturers)")
    print(f"{'=' * 70}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        engine = ReconciliationEngine()
        result = engine.generate_reconciliation(
            planning_months=planning_months,
            target_fill_pct=target_fill_pct,
            manufacturer=manufacturer
        )

        # Summary
        capacity = result.get("capacity_summary", {})
        projection = result.get("projection_details", {})

        # Show licensing filter info if manufacturer was specified
        if capacity.get("filtered_for_manufacturer"):
            print(f"LICENSING FILTER: {capacity['filtered_for_manufacturer']}")
            print(f"  Licensed locations included: {capacity.get('licensed_locations_count', 'N/A')}")
            print("-" * 50)
            print()

        if capacity.get("planning_mode"):
            print(f"\n{planning_months}-MONTH PLANNING PROJECTION")
            print("-" * 50)
            print(f"Target Fill Rate:      {capacity.get('target_fill_pct', 0.90):.0%}")
            print(f"Avg Seasonality:       {capacity.get('avg_seasonality', 1.0):.2f}x")
            print()
            print("CURRENT POSITION:")
            print(f"  Total Lot Capacity:    {capacity.get('total_lot_capacity', 0):,} units")
            print(f"  Current Inventory:     {capacity.get('total_current_inventory', 0):,} units")
            print(f"  Pipeline:              {capacity.get('total_pipeline', 0):,} units")
            print()
            print(f"PROJECTED OVER {planning_months} MONTHS:")
            print(f"  Expected Sales:        {capacity.get('total_projected_sales', 0):,} units")
            print(f"  Projected Need:        {capacity.get('total_projected_need', 0):,} units")
            print(f"  ORDER RECOMMENDATION:  {capacity.get('total_gap', 0):,} units")
            print()

            # Show by type
            by_type = result.get("needs_by_type", {})
            if by_type:
                print("PROJECTED NEEDS BY VEHICLE TYPE")
                print("-" * 70)
                print(f"{'Type':<8} {'Velocity':>10} {'Proj Sales':>12} {'End Inv':>10} {'Target':>10} {'NEED':>10}")
                print("-" * 70)
                for vt, data in by_type.items():
                    if isinstance(data, dict):
                        print(f"{vt:<8} {data.get('monthly_velocity', 0):>10.1f}/mo "
                              f"{data.get('projected_sales', 0):>10,} "
                              f"{data.get('ending_inventory_without_order', 0):>10,} "
                              f"{data.get('target_inventory', 0):>10,} "
                              f"{data.get('projected_need', 0):>10,}")
                print()
        else:
            print("\nCURRENT CAPACITY POSITION")
            print("-" * 50)
            print(f"Total Lot Capacity:    {capacity.get('total_lot_capacity', 0):,} units")
            print(f"Current Inventory:     {capacity.get('total_current_inventory', 0):,} units")
            print(f"Pipeline:              {capacity.get('total_pipeline', 0):,} units")
            print(f"TOTAL GAP TO FILL:     {capacity.get('total_gap', 0):,} units")
            print()
            if capacity.get('total_gap', 0) <= 0:
                print("NOTE: You are currently at or above capacity.")
                print("Use --planning-months 3 (or 6) to project future needs.")
                print()

        # Adjustments applied
        adjustments = result.get("adjustments_applied", [])
        if adjustments:
            print("STRATEGIC ADJUSTMENTS APPLIED")
            print("-" * 50)
            for adj in adjustments:
                print(f"  {adj['type'].upper()}: {adj['target']} -> {adj['adjustment']}")
                if adj.get("note"):
                    print(f"    Note: {adj['note']}")
            print()
        else:
            print("No strategic adjustments configured.")
            print("Edit order_adjustments.yaml to add adjustments.")
            print()

        # Final allocation
        print("MANUFACTURER ALLOCATION RECOMMENDATIONS")
        print("-" * 70)
        print(f"{'Manufacturer':<30} {'Units':>10} {'Share':>10} {'Base':>10} {'Adj':>10}")
        print("-" * 70)

        final = result.get("final_allocation", [])
        for alloc in final:
            adj_str = f"{alloc['adjustment_delta']:+,}" if alloc['adjustment_delta'] != 0 else "-"
            print(f"{alloc['manufacturer']:<30} {alloc['recommended_units']:>10,} "
                  f"{alloc['share_of_total']:>9.1%} {alloc['base_units']:>10,} {adj_str:>10}")

        total_units = sum(a['recommended_units'] for a in final)
        print("-" * 70)
        print(f"{'TOTAL':<30} {total_units:>10,}")
        print()

        # Comparison summary
        comparison = result.get("comparison", {})
        if comparison.get("manufacturers_increased"):
            print("INCREASED ALLOCATIONS:")
            for m in comparison["manufacturers_increased"]:
                print(f"  {m['manufacturer']}: {m['change']}")
        if comparison.get("manufacturers_decreased"):
            print("DECREASED ALLOCATIONS:")
            for m in comparison["manufacturers_decreased"]:
                print(f"  {m['manufacturer']}: {m['change']}")
        print()

        # Generate report
        print("Generating Excel report...")
        report_gen = ReconciliationReportGenerator()
        output_file = report_gen.generate_report(result)
        print(f"Report saved to: {output_file}")

        print()
        print("NEXT STEPS:")
        print("-" * 50)
        print("1. Review the allocation recommendations above")
        print("2. Edit order_adjustments.yaml to add/modify strategic adjustments")
        print("3. Run again to see updated allocations")
        print("4. Use --manufacturer to generate detailed orders for each manufacturer")
        print()
        print(f"Adjustments file: {engine.adjustments_file}")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"\nError during reconciliation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_gui():
    """Launch the GUI application."""
    try:
        from gui.app import OrderRecommendationApp
        import tkinter as tk

        root = tk.Tk()
        app = OrderRecommendationApp(root)
        root.mainloop()
    except ImportError as e:
        print(f"GUI not available: {e}")
        print("Please run with --cli flag for command-line mode.")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='FTRV Order Recommendation Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--cli', '-c',
        action='store_true',
        help='Run in CLI mode (default is GUI)'
    )

    parser.add_argument(
        '--manufacturer', '-m',
        help='Manufacturer/brand to generate order plan for'
    )

    parser.add_argument(
        '--period', '-p',
        type=int,
        default=3,
        help='Forecast period in months (default: 3). Use for rolling periods.'
    )

    parser.add_argument(
        '--start-month',
        help='Start month for calendar selection (e.g., "Mar 2026" or "2026-03")'
    )

    parser.add_argument(
        '--end-month',
        help='End month for calendar selection (e.g., "May 2026" or "2026-05")'
    )

    parser.add_argument(
        '--budget', '-b',
        type=float,
        help='Maximum budget constraint'
    )

    parser.add_argument(
        '--units', '-u',
        type=int,
        help='Maximum units constraint'
    )

    parser.add_argument(
        '--veh-type', '-v',
        action='append',
        dest='veh_types',
        help='Vehicle type filter (can specify multiple: -v TT -v FW)'
    )

    parser.add_argument(
        '--locations', '-l',
        help='Comma-separated list of location codes (e.g., AMA,DAL,HOU)'
    )

    parser.add_argument(
        '--zones', '-z',
        help='Comma-separated list of zones (e.g., "TX-NCENTRAL,TX-SOUTH")'
    )

    parser.add_argument(
        '--list-manufacturers',
        action='store_true',
        help='List available manufacturers'
    )

    parser.add_argument(
        '--list-zones',
        action='store_true',
        help='List available zones'
    )

    parser.add_argument(
        '--list-veh-types',
        action='store_true',
        help='List available vehicle types'
    )

    parser.add_argument(
        '--view-by',
        choices=['type', 'zone', 'make', 'model'],
        help='Generate segment-level views (monthly distribution by Type, Zone, Make, or Model)'
    )

    parser.add_argument(
        '--capacity',
        action='store_true',
        help='Run capacity-based analysis (top-down: what do we need, then who can supply it)'
    )

    parser.add_argument(
        '--reconcile',
        action='store_true',
        help='Run reconciliation analysis (allocate capacity needs across manufacturers with adjustments)'
    )

    parser.add_argument(
        '--planning-months',
        type=int,
        default=0,
        help='Project forward N months for planning mode (use with --reconcile). Shows what you will need to order.'
    )

    parser.add_argument(
        '--target-fill',
        type=float,
        default=0.90,
        help='Target fill percentage for planning mode (default: 0.90 = 90%%)'
    )

    parser.add_argument(
        '--make',
        help='Specific make/brand to order (use with --manufacturer)'
    )

    args = parser.parse_args()

    # Parse location and zone lists (used by multiple modes)
    locations = args.locations.split(',') if args.locations else None
    zones = args.zones.split(',') if args.zones else None

    # Capacity-based analysis mode (top-down approach)
    if args.capacity:
        return run_capacity_analysis(
            veh_types=args.veh_types,
            zones=zones
        )

    # Reconciliation mode (allocate needs across manufacturers)
    if args.reconcile:
        return run_reconciliation(
            planning_months=args.planning_months,
            target_fill_pct=args.target_fill,
            manufacturer=args.manufacturer
        )

    # CLI mode (manufacturer-centric)
    if args.cli or args.list_manufacturers or args.list_zones or args.list_veh_types or args.manufacturer:
        if args.list_manufacturers:
            list_manufacturers()
            return 0

        if args.list_zones:
            list_zones()
            return 0

        if args.list_veh_types:
            list_veh_types()
            return 0

        if not args.manufacturer:
            parser.print_help()
            print("\n\nExample usage:")
            print('  # Manufacturer-centric approach:')
            print('  python generate_orders.py --cli -m "FOREST RIVER"')
            print('  python generate_orders.py --cli -m "KEYSTONE" --period 6')
            print('  python generate_orders.py --cli -m "KEYSTONE" --make "ALPINE"')
            print()
            print('  # Capacity-based approach (top-down):')
            print('  python generate_orders.py --capacity')
            print('  python generate_orders.py --capacity -v TT -v FW')
            print()
            print('  # List options:')
            print('  python generate_orders.py --cli --list-manufacturers')
            return 1

        return generate_order_plan(
            manufacturer=args.manufacturer,
            make=args.make,
            period=args.period,
            budget=args.budget,
            units=args.units,
            veh_types=args.veh_types,
            locations=locations,
            zones=zones,
            start_month=args.start_month,
            end_month=args.end_month
        )

    # GUI mode (default)
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
