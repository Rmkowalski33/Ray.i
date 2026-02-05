# Order Recommendation Engine v2 - Enhancement Plan

## Overview
Transform the current recommendation tool into a comprehensive decision-support system with full transparency, supporting data, and monthly planning capabilities.

---

## 1. MONTHLY DISTRIBUTION VIEW

### Requirements
- Break recommendations into monthly buckets across the planning horizon
- Show units by month accounting for:
  - Lead time (manufacturing + transit + PDI)
  - Seasonality factors
  - Existing pipeline arrivals

### Implementation
```
Month    | Projected Demand | Current Inv | Pipeline Arriving | Gap | Recommended
---------|------------------|-------------|-------------------|-----|------------
Mar 2026 |              120 |         350 |                45 |  25 |          30
Apr 2026 |              115 |         325 |                30 |  40 |          45
May 2026 |              130 |         290 |                20 |  60 |          65
```

### Views Available
1. **By Type** (TT, FW, MH) - Monthly breakdown
2. **By Zone** (TX-NCENTRAL, EAST US, etc.) - Monthly breakdown
3. **By Brand/Make** - Monthly breakdown
4. **By Floorplan/Model** - Monthly breakdown

---

## 2. TIME PERIOD SELECTION

### Calendar Month Selection
- Pick start month (e.g., "March 2026")
- Pick end month (e.g., "May 2026")
- System calculates demand for those specific months

### Rolling Period Selection
- "Next 3 months", "Next 6 months", etc.
- Auto-calculates from current date

---

## 3. SUPPORTING DATA IN OUTPUT

### Tab: Current Inventory Position
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Units | 2,378 | - | - |
| Days Supply | 197 | 90 | OVERSTOCKED |
| Avg Age | 67 days | <60 | WARNING |
| Units 90+ Days | 412 | <10% | CRITICAL |
| Total Retail Value | $95.2M | - | - |
| Est. Holding Cost/Month | $126K | - | - |

### Tab: On Order Analysis
| Status | Units | Avg Age | Est. Arrival Window |
|--------|-------|---------|---------------------|
| PO ISSUED | 892 | 12 days | Mar-Apr 2026 |
| IN PRODUCTION | 356 | 28 days | Apr-May 2026 |
| IN TRANSIT | 200 | 45 days | Feb-Mar 2026 |

**Pipeline Flow Health:**
- Units ordered last 30 days: X
- Units received last 30 days: Y
- Flow ratio: Y/X (healthy = 0.8-1.2)

### Tab: Retail Performance (YoY)
| Period | 2024 | 2025 | YoY Change | Adj. for Locations |
|--------|------|------|------------|-------------------|
| Q1 | 2,100 | 2,850 | +35.7% | +14.2%* |
| Q2 | 2,400 | 3,150 | +31.3% | +10.8%* |
| ...

*Adjusted for location count: 2024 had 27 locations, 2025 had 35

### Tab: Brand Licensing Status
| Brand | Licensed Locations | Pending | Not Available |
|-------|-------------------|---------|---------------|
| IMPRESSION | 28 | 3 | 5 |
| SANDPIPER | 25 | 2 | 9 |
| ...

### Tab: Market Share (SSI)
| BTA Region | FTRV Sales | Total Market | Share | Trend |
|------------|------------|--------------|-------|-------|
| Dallas-Fort Worth | 1,245 | 8,900 | 14.0% | +2.1% |
| Houston | 890 | 7,200 | 12.4% | +0.8% |
| ...

---

## 4. RECOMMENDATION TRANSPARENCY

### For Each Model Recommendation, Show:
```
MODEL: IMPRESSION 360MYR
Recommended Qty: 15 units over 3 months (5/mo)

WHY THIS RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━
Current Position:
  • Current Inventory: 8 units
  • On Order: 3 units
  • Total Position: 11 units

Demand Analysis:
  • LTM Sales: 100 units (8.3/month)
  • Avg Days to Sell: 59 days
  • Trend: STABLE (+2% YoY)

Seasonality Adjustment:
  • Mar: 115 index → 9.5 expected
  • Apr: 105 index → 8.7 expected
  • May: 100 index → 8.3 expected
  • 3-Month Forecast: 26.5 units

Gap Calculation:
  • Target Days Supply: 90 days (3 months)
  • Target Position: 25 units
  • Current Position: 11 units
  • Gap: 14 units
  • Safety Buffer (10%): 1.4 units
  • RECOMMENDED: 15 units

Financial Impact:
  • Est. Cost: $750,000 ($50K/unit)
  • Avg Front-End: $4,450/unit
  • Margin %: 14.7%
  • Est. Holding Cost: $12,500 (at 90-day avg)

Confidence: HIGH (100 sales in LTM)
```

---

## 5. ORDER FORM HANDLING

### Option A: Manual Model Selection
- GUI with checkboxes for each model
- User enters quantities
- System validates against recommendation

### Option B: Excel Upload (Intelligent Parse)
- User uploads manufacturer Excel
- AI/heuristics detect: Model, Qty, Cost columns
- Preview for user approval
- Fall back to manual mapping if needed

### Option C: Quick Entry
- Text input: "IMPRESSION 360MYR: 15, SANDPIPER 3800RK: 10"
- Parsed and validated

---

## 6. FILE STRUCTURE CHANGES

```
Claude Toolkit/
├── generate_orders.py          # Main entry point
├── gui/
│   ├── app.py                  # Enhanced GUI
│   └── upload_wizard.py        # Order form upload
├── order_engine/
│   ├── config.py               # Settings
│   ├── data_loader.py          # Data loading
│   ├── market_analyzer.py      # SSI integration (FIXED)
│   ├── pipeline_analyzer.py    # On-order analysis
│   ├── financial_analyzer.py   # Margins, holding costs
│   ├── recommendation_engine.py # Core logic
│   ├── monthly_planner.py      # NEW: Monthly distribution
│   ├── transparency.py         # NEW: Explain recommendations
│   ├── yoy_analyzer.py         # NEW: YoY comparisons
│   └── report_generator.py     # Enhanced Excel output
└── output/
```

---

## 7. PRIORITY ORDER

### Phase 1 (Critical)
1. Fix SSI market share data loading
2. Add On Order analysis tab
3. Add YoY comparison (location-adjusted)
4. Improve recommendation transparency

### Phase 2 (High)
5. Monthly distribution view
6. Calendar/rolling month selection
7. Views by Type/Zone/Brand/Floorplan

### Phase 3 (Medium)
8. Order form upload/parsing
9. Brand licensing integration
10. Enhanced GUI with all features

---

## 8. KEY METRICS TO SURFACE

### Inventory Health Dashboard
- Days Supply by Type/Brand
- Age Distribution (pie chart data)
- Flow Rate (orders vs receipts)
- Holding Cost Accumulation

### Demand Signals
- Monthly velocity trends
- Seasonality patterns
- YoY growth (adjusted)
- Market share trajectory

### Financial Impact
- Total investment required
- Projected margin
- Projected holding costs
- ROI by segment

---

## Questions for User Before Implementation

1. For YoY adjustment: Should we use avg locations for the year, or month-by-month location count?
2. For pipeline flow health: What's your target flow ratio (receipts/orders)?
3. For market share: Which BTAs are most important to track?
