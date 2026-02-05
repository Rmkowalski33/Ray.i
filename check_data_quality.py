#!/usr/bin/env python
"""
FTRV Data Quality Checker
=========================
Analyzes inventory data for quality issues, anomalies, and inconsistencies.

Usage:
    python check_data_quality.py                    # Run all checks
    python check_data_quality.py --summary-only     # Just show summary counts
    python check_data_quality.py --check licensing  # Run specific check

Checks performed:
    - Licensing violations (units at unlicensed locations)
    - Text quality (potential typos, similar strings)
    - Whitespace issues (leading/trailing spaces, double spaces)
    - Relationship inconsistencies (manufacturer-make-model-type)
    - Model year issues (too old, future years, missing)
    - Pricing issues (missing, negative margin, outliers)
    - Missing required data
    - Duplicate detection (duplicate VINs, Stock #s)
    - Status anomalies (stuck units, unexpected statuses)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from order_engine.data_quality import DataQualityAnalyzer, DataQualityReportGenerator, DataCleaner
from order_engine.config import default_config


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def print_issue_summary(category: str, findings: dict):
    """Print summary for a category."""
    count = findings.get("issue_count", 0)
    status_icon = "X" if count > 0 else "OK"
    print(f"  [{status_icon}] {category}: {count} issues")

    # Show brief description if issues found
    if count > 0 and findings.get("description"):
        print(f"      -> {findings['description']}")


def print_detailed_findings(category: str, findings: dict, max_items: int = 10):
    """Print detailed findings for a category."""
    issues = findings.get("issues", findings.get("violations", []))

    if not issues:
        return

    print(f"\n{category}:")
    print("-" * 50)

    for i, issue in enumerate(issues[:max_items]):
        if isinstance(issue, dict):
            stock = issue.get("stock_num", issue.get("value", ""))
            issue_text = issue.get("issue", issue.get("suggestion", str(issue)))
            print(f"  {i+1}. [{stock}] {issue_text}")
        else:
            print(f"  {i+1}. {issue}")

    if len(issues) > max_items:
        print(f"  ... and {len(issues) - max_items} more")


def main():
    parser = argparse.ArgumentParser(
        description="FTRV Data Quality Checker - Find anomalies and inconsistencies in inventory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_data_quality.py                    # Full analysis with report
  python check_data_quality.py --summary-only     # Quick summary
  python check_data_quality.py --check licensing  # Specific check
  python check_data_quality.py --verbose          # Show detailed findings
        """
    )

    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Show only summary counts, skip detailed output"
    )

    parser.add_argument(
        "--check", "-c",
        type=str,
        choices=["licensing", "text", "whitespace", "relationships", "years", "pricing", "missing", "duplicates", "status"],
        help="Run only a specific check"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed findings in console output"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating Excel report"
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=10,
        help="Max items to show per category in verbose mode (default: 10)"
    )

    parser.add_argument(
        "--fix-list",
        action="store_true",
        help="Generate a DMS-friendly fix list for whitespace and typo corrections"
    )

    args = parser.parse_args()

    print_section("FTRV DATA QUALITY ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize analyzer
    analyzer = DataQualityAnalyzer(config=default_config)

    # Run analysis
    results = analyzer.run_full_analysis()

    # Print summary
    print_section("SUMMARY", "-")
    print(f"  Total Units Analyzed: {results.get('total_units_analyzed', 0):,}")
    print(f"  Total Issues Found:   {results.get('total_issues', 0):,}")
    print()

    categories = results.get("categories", {})

    # Group by severity
    high_priority = []
    medium_priority = []
    low_priority = []

    severity_map = {
        "Licensing Violations": "HIGH",
        "Pricing Issues": "HIGH",
        "Missing Required Data": "MEDIUM",
        "Relationship Inconsistencies": "MEDIUM",
        "Text Quality Issues": "MEDIUM",
        "Model Year Issues": "MEDIUM",
        "Duplicate Detection": "HIGH",
        "Status Anomalies": "MEDIUM",
        "Whitespace Issues": "LOW",
    }

    for cat, findings in categories.items():
        count = findings.get("issue_count", 0)
        if count > 0:
            severity = severity_map.get(cat, "LOW")
            if severity == "HIGH":
                high_priority.append((cat, findings))
            elif severity == "MEDIUM":
                medium_priority.append((cat, findings))
            else:
                low_priority.append((cat, findings))

    # Print by priority
    if high_priority:
        print("  HIGH PRIORITY (fix immediately):")
        for cat, findings in high_priority:
            print_issue_summary(cat, findings)

    if medium_priority:
        print("\n  MEDIUM PRIORITY (investigate soon):")
        for cat, findings in medium_priority:
            print_issue_summary(cat, findings)

    if low_priority:
        print("\n  LOW PRIORITY (cleanup when possible):")
        for cat, findings in low_priority:
            print_issue_summary(cat, findings)

    # Print all zero categories
    zero_cats = [cat for cat, f in categories.items() if f.get("issue_count", 0) == 0]
    if zero_cats:
        print("\n  PASSED (no issues):")
        for cat in zero_cats:
            print(f"    [OK] {cat}")

    # Verbose output
    if args.verbose and not args.summary_only:
        print_section("DETAILED FINDINGS", "-")

        for cat, findings in categories.items():
            if findings.get("issue_count", 0) > 0:
                print_detailed_findings(cat, findings, max_items=args.max_items)

    # Generate report
    if not args.no_report:
        print_section("GENERATING REPORT", "-")
        report_gen = DataQualityReportGenerator(config=default_config)
        report_path = report_gen.generate_report(results)
        print(f"  Report saved to: {report_path}")

    # Generate DMS fix list if requested
    if args.fix_list:
        print_section("GENERATING DMS FIX LIST", "-")
        cleaner = DataCleaner(config=default_config)
        fixes = cleaner.preview_fixes()

        if fixes["total_fixes"] > 0:
            # Generate the fix script file
            fix_file = cleaner.generate_fix_script()
            print(f"  Fix list saved to: {fix_file}")
            print(f"  Total fixes needed: {fixes['total_fixes']}")
            print(f"  Records affected: {fixes['total_records_affected']}")
            print()
            print("  WHITESPACE FIXES NEEDED:")
            for fix in fixes["whitespace_fixes"][:10]:
                print(f"    Stock# {fix['stock_num']}: {fix['field']}")
                print(f"      Change: \"{fix['original']}\" -> \"{fix['fixed']}\"")
            if len(fixes["whitespace_fixes"]) > 10:
                print(f"    ... and {len(fixes['whitespace_fixes']) - 10} more")
            print()
            print("  TYPO FIXES NEEDED:")
            for fix in fixes["typo_fixes"]:
                print(f"    Stock# {fix['stock_num']}: {fix['field']}")
                print(f"      Change: \"{fix['original']}\" -> \"{fix['fixed']}\"")
            if not fixes["typo_fixes"]:
                print("    (none)")
        else:
            print("  No auto-fixable issues found.")

    # Recommendations
    print_section("RECOMMENDATIONS")

    if not high_priority and not medium_priority:
        print("  Data quality looks good! No critical issues found.")
    else:
        if high_priority:
            print("  1. Address HIGH priority issues first:")
            for cat, findings in high_priority:
                count = findings.get("issue_count", 0)
                if "Licensing" in cat:
                    print(f"     - {count} units may need transfer or licensing review")
                elif "Pricing" in cat:
                    print(f"     - {count} pricing issues affect margin calculations")
                elif "Duplicate" in cat:
                    print(f"     - {count} potential duplicates need investigation")

        if medium_priority:
            print("\n  2. Review MEDIUM priority for data consistency:")
            for cat, findings in medium_priority:
                count = findings.get("issue_count", 0)
                print(f"     - {cat}: {count} issues")

        print("\n  3. Review the Excel report for full details on each issue.")

    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
