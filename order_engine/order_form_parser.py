"""
Order Form Parser Module
Intelligently parses manufacturer Excel order forms to extract model/qty/cost data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import re


@dataclass
class ColumnMapping:
    """Mapping of detected columns to standard fields."""
    model_col: str = None
    qty_col: str = None
    cost_col: str = None
    make_col: str = None
    msrp_col: str = None
    floorplan_col: str = None
    notes_col: str = None
    confidence: float = 0.0
    unmapped_cols: List[str] = field(default_factory=list)


@dataclass
class ParsedOrder:
    """A single parsed order line."""
    model: str
    qty: int
    unit_cost: float = 0
    make: str = None
    msrp: float = 0
    floorplan: str = None
    notes: str = None
    row_number: int = 0
    raw_data: Dict = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing an order form."""
    success: bool
    orders: List[ParsedOrder]
    column_mapping: ColumnMapping
    total_units: int = 0
    total_cost: float = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_file: str = ""
    sheet_name: str = ""


class OrderFormParser:
    """Parse manufacturer order forms with intelligent column detection."""

    # Common column name patterns
    MODEL_PATTERNS = [
        r'^model$', r'^model\s*(name|#|number)?$', r'^unit\s*model$',
        r'^floor\s*plan$', r'^floorplan$', r'^sku$', r'^item$',
        r'^product$', r'^description$'
    ]

    QTY_PATTERNS = [
        r'^qty$', r'^quantity$', r'^units?$', r'^count$', r'^order\s*qty$',
        r'^#\s*units$', r'^num(ber)?$', r'^amount$'
    ]

    COST_PATTERNS = [
        r'^cost$', r'^unit\s*cost$', r'^dealer\s*cost$', r'^invoice$',
        r'^wholesale$', r'^price$', r'^net\s*price$', r'^dealer\s*price$'
    ]

    MSRP_PATTERNS = [
        r'^msrp$', r'^retail$', r'^list\s*price$', r'^srp$',
        r'^suggested\s*retail$'
    ]

    MAKE_PATTERNS = [
        r'^make$', r'^brand$', r'^line$', r'^product\s*line$',
        r'^series$'
    ]

    def __init__(self):
        self.last_mapping = None

    def parse_file(
        self,
        file_path: str,
        sheet_name: str = None,
        column_mapping: ColumnMapping = None,
        header_row: int = None
    ) -> ParseResult:
        """
        Parse an order form file.

        Args:
            file_path: Path to Excel/CSV file
            sheet_name: Sheet name for Excel files (auto-detect if None)
            column_mapping: Pre-defined column mapping (auto-detect if None)
            header_row: Row number for headers (auto-detect if None)

        Returns:
            ParseResult with orders and metadata
        """
        path = Path(file_path)

        if not path.exists():
            return ParseResult(
                success=False,
                orders=[],
                column_mapping=ColumnMapping(),
                errors=[f"File not found: {file_path}"]
            )

        try:
            # Load file
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                sheet_name = "CSV"
            else:
                # Excel file - detect best sheet if not specified
                if sheet_name is None:
                    sheet_name = self._detect_best_sheet(file_path)
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

            # Clean up the dataframe
            df = self._clean_dataframe(df)

            if df.empty:
                return ParseResult(
                    success=False,
                    orders=[],
                    column_mapping=ColumnMapping(),
                    errors=["No data found in file"]
                )

            # Detect or use provided column mapping
            if column_mapping is None:
                column_mapping = self._detect_columns(df)

            # Validate we have minimum required columns
            if not column_mapping.model_col:
                return ParseResult(
                    success=False,
                    orders=[],
                    column_mapping=column_mapping,
                    errors=["Could not detect Model column"],
                    source_file=str(file_path),
                    sheet_name=sheet_name
                )

            if not column_mapping.qty_col:
                # Try to infer qty = 1 for each row
                column_mapping.qty_col = "_inferred_qty"
                df["_inferred_qty"] = 1

            # Parse orders
            orders, warnings = self._parse_orders(df, column_mapping)

            # Calculate totals
            total_units = sum(o.qty for o in orders)
            total_cost = sum(o.qty * o.unit_cost for o in orders)

            self.last_mapping = column_mapping

            return ParseResult(
                success=True,
                orders=orders,
                column_mapping=column_mapping,
                total_units=total_units,
                total_cost=total_cost,
                warnings=warnings,
                source_file=str(file_path),
                sheet_name=sheet_name
            )

        except Exception as e:
            return ParseResult(
                success=False,
                orders=[],
                column_mapping=ColumnMapping(),
                errors=[f"Error parsing file: {str(e)}"],
                source_file=str(file_path)
            )

    def _detect_best_sheet(self, file_path: str) -> str:
        """Detect the best sheet to use in an Excel file."""
        xl = pd.ExcelFile(file_path)
        sheets = xl.sheet_names

        if len(sheets) == 1:
            return sheets[0]

        # Look for sheets with order-related names
        order_keywords = ['order', 'unit', 'model', 'inventory', 'quote', 'proposal']
        for sheet in sheets:
            if any(kw in sheet.lower() for kw in order_keywords):
                return sheet

        # Return first sheet as default
        return sheets[0]

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up dataframe - remove empty rows, standardize columns."""
        # Drop completely empty rows
        df = df.dropna(how='all')

        # Drop completely empty columns
        df = df.dropna(axis=1, how='all')

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # Check if first row looks like headers (common in manufacturer forms)
        if len(df) > 0:
            first_row = df.iloc[0]
            # If first row values look like column names, use them
            if all(isinstance(v, str) for v in first_row.values if pd.notna(v)):
                potential_headers = first_row.values
                if any(self._matches_pattern(str(v), self.MODEL_PATTERNS) for v in potential_headers if pd.notna(v)):
                    df.columns = [str(v).strip() if pd.notna(v) else f"Col_{i}" for i, v in enumerate(potential_headers)]
                    df = df.iloc[1:].reset_index(drop=True)

        return df

    def _detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        """Auto-detect column mappings based on column names and content."""
        mapping = ColumnMapping()
        columns = list(df.columns)
        mapped = set()

        # Try to match each column type
        mapping.model_col = self._find_matching_column(columns, self.MODEL_PATTERNS, mapped)
        mapping.qty_col = self._find_matching_column(columns, self.QTY_PATTERNS, mapped)
        mapping.cost_col = self._find_matching_column(columns, self.COST_PATTERNS, mapped)
        mapping.msrp_col = self._find_matching_column(columns, self.MSRP_PATTERNS, mapped)
        mapping.make_col = self._find_matching_column(columns, self.MAKE_PATTERNS, mapped)

        # If model not found by name, look for column with model-like values
        if not mapping.model_col:
            mapping.model_col = self._detect_model_column_by_content(df, mapped)

        # If qty not found by name, look for numeric column with small integers
        if not mapping.qty_col:
            mapping.qty_col = self._detect_qty_column_by_content(df, mapped)

        # If cost not found, look for currency-like values
        if not mapping.cost_col:
            mapping.cost_col = self._detect_cost_column_by_content(df, mapped)

        # Track unmapped columns
        mapping.unmapped_cols = [c for c in columns if c not in mapped and not c.startswith('_')]

        # Calculate confidence
        matched_count = sum([
            1 if mapping.model_col else 0,
            1 if mapping.qty_col else 0,
            1 if mapping.cost_col else 0
        ])
        mapping.confidence = matched_count / 3.0

        return mapping

    def _find_matching_column(
        self,
        columns: List[str],
        patterns: List[str],
        mapped: set
    ) -> Optional[str]:
        """Find a column matching any of the patterns."""
        for col in columns:
            if col in mapped:
                continue
            col_lower = col.lower().strip()
            for pattern in patterns:
                if re.match(pattern, col_lower, re.IGNORECASE):
                    mapped.add(col)
                    return col
        return None

    def _matches_pattern(self, value: str, patterns: List[str]) -> bool:
        """Check if value matches any pattern."""
        value_lower = value.lower().strip()
        for pattern in patterns:
            if re.match(pattern, value_lower, re.IGNORECASE):
                return True
        return False

    def _detect_model_column_by_content(
        self,
        df: pd.DataFrame,
        mapped: set
    ) -> Optional[str]:
        """Detect model column by analyzing content (alphanumeric codes)."""
        for col in df.columns:
            if col in mapped:
                continue

            values = df[col].dropna().astype(str)
            if len(values) == 0:
                continue

            # Model numbers often have patterns like: 360MYR, 3800RK, 29QB
            model_pattern = r'^[A-Z0-9]{2,}[A-Z0-9\-/\s]*$'
            matches = values.str.upper().str.match(model_pattern).sum()
            if matches > len(values) * 0.5:  # >50% match
                mapped.add(col)
                return col

        return None

    def _detect_qty_column_by_content(
        self,
        df: pd.DataFrame,
        mapped: set
    ) -> Optional[str]:
        """Detect quantity column by analyzing content (small integers)."""
        for col in df.columns:
            if col in mapped:
                continue

            try:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) == 0:
                    continue

                # Qty is usually small positive integers
                if values.min() >= 0 and values.max() <= 100 and (values == values.astype(int)).all():
                    if values.mean() < 20:  # Average qty usually low
                        mapped.add(col)
                        return col
            except Exception:
                continue

        return None

    def _detect_cost_column_by_content(
        self,
        df: pd.DataFrame,
        mapped: set
    ) -> Optional[str]:
        """Detect cost column by analyzing content (currency values)."""
        for col in df.columns:
            if col in mapped:
                continue

            values = df[col].dropna()
            if len(values) == 0:
                continue

            # Try to parse as currency
            try:
                # Remove currency symbols and commas
                cleaned = values.astype(str).str.replace(r'[$,]', '', regex=True)
                numeric = pd.to_numeric(cleaned, errors='coerce').dropna()

                if len(numeric) > 0:
                    # Cost is usually in thousands range for RVs
                    if numeric.min() > 1000 and numeric.max() < 500000:
                        mapped.add(col)
                        return col
            except Exception:
                continue

        return None

    def _parse_orders(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping
    ) -> Tuple[List[ParsedOrder], List[str]]:
        """Parse orders from dataframe using column mapping."""
        orders = []
        warnings = []

        for idx, row in df.iterrows():
            try:
                # Get model
                model = str(row.get(mapping.model_col, "")).strip()
                if not model or model.lower() in ['nan', 'none', '']:
                    continue

                # Get quantity
                qty_raw = row.get(mapping.qty_col, 1)
                try:
                    qty = int(float(str(qty_raw).replace(',', '')))
                except (ValueError, TypeError):
                    qty = 1
                    warnings.append(f"Row {idx + 1}: Could not parse qty '{qty_raw}', using 1")

                if qty <= 0:
                    continue

                # Get cost
                cost = 0
                if mapping.cost_col and mapping.cost_col in row.index:
                    cost_raw = row.get(mapping.cost_col, 0)
                    try:
                        cost = float(str(cost_raw).replace('$', '').replace(',', ''))
                    except (ValueError, TypeError):
                        cost = 0

                # Get optional fields
                make = None
                if mapping.make_col and mapping.make_col in row.index:
                    make = str(row.get(mapping.make_col, "")).strip()
                    if make.lower() in ['nan', 'none', '']:
                        make = None

                msrp = 0
                if mapping.msrp_col and mapping.msrp_col in row.index:
                    try:
                        msrp = float(str(row.get(mapping.msrp_col, 0)).replace('$', '').replace(',', ''))
                    except (ValueError, TypeError):
                        msrp = 0

                order = ParsedOrder(
                    model=model.upper(),
                    qty=qty,
                    unit_cost=cost,
                    make=make.upper() if make else None,
                    msrp=msrp,
                    row_number=idx + 1,
                    raw_data=row.to_dict()
                )
                orders.append(order)

            except Exception as e:
                warnings.append(f"Row {idx + 1}: Error parsing - {str(e)}")

        return orders, warnings

    def validate_against_recommendations(
        self,
        parsed_orders: List[ParsedOrder],
        recommendations: List[Dict]
    ) -> Dict:
        """
        Validate parsed orders against system recommendations.

        Returns dict with validation results and comparisons.
        """
        # Build lookup from recommendations
        rec_by_model = {}
        for rec in recommendations:
            model = str(rec.get('model', '')).upper().strip()
            if model:
                rec_by_model[model] = rec

        matched = []
        unmatched = []
        over_recommended = []
        under_recommended = []

        for order in parsed_orders:
            model = order.model.upper().strip()

            if model in rec_by_model:
                rec = rec_by_model[model]
                rec_qty = rec.get('recommended_qty', 0)

                match_info = {
                    'order': order,
                    'recommendation': rec,
                    'order_qty': order.qty,
                    'recommended_qty': rec_qty,
                    'difference': order.qty - rec_qty
                }

                matched.append(match_info)

                if order.qty > rec_qty:
                    over_recommended.append(match_info)
                elif order.qty < rec_qty:
                    under_recommended.append(match_info)
            else:
                unmatched.append(order)

        return {
            'matched_count': len(matched),
            'unmatched_count': len(unmatched),
            'matched': matched,
            'unmatched': unmatched,
            'over_recommended': over_recommended,
            'under_recommended': under_recommended,
            'total_order_units': sum(o.qty for o in parsed_orders),
            'total_recommended_units': sum(r.get('recommended_qty', 0) for r in recommendations)
        }

    def to_dataframe(self, orders: List[ParsedOrder]) -> pd.DataFrame:
        """Convert parsed orders to DataFrame."""
        if not orders:
            return pd.DataFrame()

        data = []
        for order in orders:
            data.append({
                'Model': order.model,
                'Make': order.make or '',
                'Qty': order.qty,
                'Unit Cost': order.unit_cost,
                'Total Cost': order.qty * order.unit_cost,
                'MSRP': order.msrp,
                'Row #': order.row_number
            })

        return pd.DataFrame(data)

    def suggest_column_mapping(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get column suggestions for manual mapping.

        Returns dict with field names and list of candidate columns.
        """
        columns = list(df.columns)

        suggestions = {
            'model': [],
            'qty': [],
            'cost': [],
            'make': [],
            'msrp': []
        }

        for col in columns:
            col_lower = col.lower()

            # Score each column for each field type
            if any(re.match(p, col_lower, re.I) for p in self.MODEL_PATTERNS):
                suggestions['model'].insert(0, col)
            elif 'model' in col_lower or 'floor' in col_lower or 'sku' in col_lower:
                suggestions['model'].append(col)

            if any(re.match(p, col_lower, re.I) for p in self.QTY_PATTERNS):
                suggestions['qty'].insert(0, col)
            elif 'qty' in col_lower or 'unit' in col_lower or 'count' in col_lower:
                suggestions['qty'].append(col)

            if any(re.match(p, col_lower, re.I) for p in self.COST_PATTERNS):
                suggestions['cost'].insert(0, col)
            elif 'cost' in col_lower or 'price' in col_lower or 'invoice' in col_lower:
                suggestions['cost'].append(col)

            if any(re.match(p, col_lower, re.I) for p in self.MAKE_PATTERNS):
                suggestions['make'].insert(0, col)
            elif 'make' in col_lower or 'brand' in col_lower or 'line' in col_lower:
                suggestions['make'].append(col)

            if any(re.match(p, col_lower, re.I) for p in self.MSRP_PATTERNS):
                suggestions['msrp'].insert(0, col)
            elif 'msrp' in col_lower or 'retail' in col_lower:
                suggestions['msrp'].append(col)

        return suggestions
