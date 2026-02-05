"""
Configuration for the FTRV Order Recommendation Engine.

CONFIGURATION OPTIONS:
----------------------
1. EASY WAY (Recommended): Edit settings.yaml in the Claude Toolkit folder
   - Human-readable YAML format
   - No Python knowledge required
   - Just edit values and save

2. PROGRAMMATIC WAY: Modify Config dataclass or use with_filters()
   - For advanced users or automation
   - Full type safety with dataclasses

All paths, thresholds, and parameters are configurable via either method.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml

# =============================================================================
# DEFAULT PATHS
# =============================================================================

TOOLKIT_PATH = Path(r"C:\Users\Raymond Kowalski\OneDrive\Desktop\Claude Toolkit")
DATA_HUB_PATH = Path(r"C:\Users\Raymond Kowalski\OneDrive\Desktop\RV Reporting\Data Hub")
OUTPUT_PATH = TOOLKIT_PATH / "output"
SETTINGS_FILE = TOOLKIT_PATH / "settings.yaml"


@dataclass
class Config:
    """Configuration settings for the order recommendation system."""

    # =========================================================================
    # FILE PATHS
    # =========================================================================
    data_hub_path: Path = DATA_HUB_PATH
    output_path: Path = OUTPUT_PATH

    # Data files (relative to data_hub_path)
    inventory_file: str = "Inventory_Extract.xlsx"
    references_file: str = "FTRV Index & References.xlsx"
    ssi_data_file: str = "SSI/SSI Full Data.csv"
    ssi_fact_tables_file: str = "SSI/FTRV SSI Fact Tables.xlsx"

    # Sheet names
    inventory_sheet: str = "Inventory Records"
    retail_sheet: str = "Retail Units"
    on_order_sheet: str = "On Order"
    locations_sheet: str = "Locations"
    status_sheet: str = "Status - Normalized"
    product_sheet: str = "Product"

    # =========================================================================
    # FORECAST PARAMETERS
    # =========================================================================
    forecast_months: int = 3          # Default order period (3, 4, 6, 12 months)
    lookback_months: int = 12         # Months of history for demand analysis

    # =========================================================================
    # INVENTORY TARGET METRICS
    # =========================================================================
    target_days_supply: int = 90      # Ideal days of sellable inventory
    min_days_supply: int = 60         # Reorder trigger threshold
    max_days_supply: int = 150        # Overstocked threshold
    critical_days_supply: int = 45    # Urgent reorder threshold

    # =========================================================================
    # LEAD TIME ASSUMPTIONS (in days)
    # =========================================================================
    mfg_build_days: int = 35          # 4-6 weeks manufacturing
    transit_to_pdi_days: int = 14     # Shipping to PDI center
    pdi_processing_days: int = 7      # Inspection time at PDI
    transit_to_store_days: int = 14   # Final delivery to stores

    @property
    def total_lead_time(self) -> int:
        """Total days from order to on-lot availability."""
        return (self.mfg_build_days +
                self.transit_to_pdi_days +
                self.pdi_processing_days +
                self.transit_to_store_days)

    # =========================================================================
    # FINANCIAL PARAMETERS
    # =========================================================================
    floorplan_rate: float = 0.08      # Annual floorplan interest rate (8%)
    target_turn_rate: float = 4.0     # Target annual inventory turns
    min_acceptable_margin: float = 0.05  # Minimum front-end gross margin (5%)

    @property
    def daily_holding_cost_rate(self) -> float:
        """Daily floorplan interest cost as a decimal."""
        return self.floorplan_rate / 365

    # =========================================================================
    # SEASONALITY INDEX (100 = average month)
    # Based on RV industry patterns
    # =========================================================================
    seasonality_index: Dict[int, int] = field(default_factory=lambda: {
        1: 120,   # January - RV show season begins
        2: 130,   # February - Peak show season
        3: 115,   # March - Spring buying starts
        4: 105,   # April
        5: 100,   # May - Average
        6: 95,    # June
        7: 85,    # July - Summer slowdown
        8: 80,    # August - Lowest demand
        9: 90,    # September - Fall uptick
        10: 95,   # October
        11: 85,   # November
        12: 100   # December - Year-end deals
    })

    # =========================================================================
    # AGE BUCKETS FOR INVENTORY ANALYSIS
    # =========================================================================
    age_buckets: Dict[str, tuple] = field(default_factory=lambda: {
        "Fresh (0-30)": (0, 30),
        "Good (31-60)": (31, 60),
        "Aging (61-90)": (61, 90),
        "Stale (91-120)": (91, 120),
        "Problem (121-180)": (121, 180),
        "Critical (181+)": (181, 9999)
    })

    # =========================================================================
    # VEHICLE TYPE MAPPINGS
    # =========================================================================
    veh_type_names: Dict[str, str] = field(default_factory=lambda: {
        "TT": "Travel Trailer",
        "FW": "Fifth Wheel",
        "TH": "Toy Hauler",
        "MH": "Motorhome",
        "CT/EXP": "Camping Trailer",
    })

    # =========================================================================
    # STATUS CLASSIFICATIONS
    # =========================================================================
    sellable_statuses: List[str] = field(default_factory=lambda: [
        "RVASAP", "READY FOR SALE", "SHOWROOM", "PRE PDI",
        "HAIL DAMAGE", "AS IS", "RESTOCK"
    ])

    constrained_statuses: List[str] = field(default_factory=lambda: [
        "SALE PENDING", "AWAITING TITLE", "AWAITING PARTS",
        "IN SERVICE", "TRANSFER", "DISPATCHED"
    ])

    dead_statuses: List[str] = field(default_factory=lambda: [
        "WHOLESALE - USED", "TRADE IN", "BUYBACK",
        "INSURANCE CLAIM", "STORM DAMAGE", "FACTORY REVIEW"
    ])

    # =========================================================================
    # FILTER OPTIONS (for runtime filtering)
    # =========================================================================
    filter_manufacturer: Optional[str] = None
    filter_make: Optional[str] = None
    filter_veh_types: Optional[List[str]] = None
    filter_locations: Optional[List[str]] = None
    filter_zones: Optional[List[str]] = None
    filter_price_groups: Optional[List[str]] = None

    # =========================================================================
    # RECOMMENDATION PARAMETERS
    # =========================================================================
    min_sample_size: int = 5          # Minimum sales to calculate velocity
    safety_buffer_pct: float = 0.10   # 10% safety stock buffer
    max_models_in_report: int = 100   # Limit models in output

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_full_path(self, relative_path: str) -> Path:
        """Get full path for a file relative to data_hub_path."""
        return self.data_hub_path / relative_path

    def get_seasonality_factor(self, month: int) -> float:
        """Get seasonality factor for a given month (1-12)."""
        return self.seasonality_index.get(month, 100) / 100.0

    def get_age_bucket(self, age_days: int) -> str:
        """Classify age into bucket."""
        for bucket_name, (min_age, max_age) in self.age_buckets.items():
            if min_age <= age_days <= max_age:
                return bucket_name
        return "Critical (181+)"

    def calculate_holding_cost(self, cost: float, days: int) -> float:
        """Calculate floorplan interest cost for holding a unit."""
        return cost * self.daily_holding_cost_rate * days

    def with_filters(
        self,
        manufacturer: str = None,
        make: str = None,
        veh_types: List[str] = None,
        locations: List[str] = None,
        zones: List[str] = None,
        forecast_months: int = None
    ) -> "Config":
        """Return a new config with filters applied."""
        from dataclasses import replace
        new_config = replace(self)
        if manufacturer:
            new_config.filter_manufacturer = manufacturer
        if make:
            new_config.filter_make = make
        if veh_types:
            new_config.filter_veh_types = veh_types
        if locations:
            new_config.filter_locations = locations
        if zones:
            new_config.filter_zones = zones
        if forecast_months:
            new_config.forecast_months = forecast_months
        return new_config


# =============================================================================
# SETTINGS LOADER
# =============================================================================

def load_settings_from_yaml(yaml_path: Path = None) -> dict:
    """
    Load settings from YAML file.

    Args:
        yaml_path: Path to settings.yaml (uses default if not provided)

    Returns:
        Dictionary of settings, or empty dict if file not found
    """
    yaml_path = yaml_path or SETTINGS_FILE
    if not yaml_path.exists():
        return {}

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load settings from {yaml_path}: {e}")
        return {}


def config_from_yaml(yaml_path: Path = None) -> "Config":
    """
    Create a Config object from settings.yaml.

    This is the recommended way to get a configured Config object.
    It reads settings.yaml and applies all values to a new Config.

    Args:
        yaml_path: Path to settings.yaml (uses default if not provided)

    Returns:
        Config object with settings applied
    """
    settings = load_settings_from_yaml(yaml_path)

    if not settings:
        return Config()

    # Extract nested settings
    paths = settings.get('paths', {})
    files = settings.get('files', {})
    sheets = settings.get('sheets', {})
    forecast = settings.get('forecast', {})
    targets = settings.get('inventory_targets', {})
    lead_time = settings.get('lead_time', {})
    financial = settings.get('financial', {})
    seasonality = settings.get('seasonality', {})
    age_buckets_raw = settings.get('age_buckets', {})
    status_cats = settings.get('status_categories', {})
    veh_types = settings.get('vehicle_types', {})
    recs = settings.get('recommendations', {})
    filters = settings.get('default_filters', {})

    # Build seasonality index
    month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    seasonality_index = {}
    for i, month_name in enumerate(month_names, 1):
        seasonality_index[i] = seasonality.get(month_name, 100)

    # Build age buckets
    age_bucket_map = {
        'fresh': 'Fresh (0-30)',
        'good': 'Good (31-60)',
        'aging': 'Aging (61-90)',
        'stale': 'Stale (91-120)',
        'problem': 'Problem (121-180)',
        'critical': 'Critical (181+)'
    }
    age_buckets = {}
    for key, label in age_bucket_map.items():
        if key in age_buckets_raw:
            bounds = age_buckets_raw[key]
            age_buckets[label] = (bounds[0], bounds[1])

    # Create config with settings
    config = Config(
        # Paths
        data_hub_path=Path(paths.get('data_hub', DATA_HUB_PATH)),
        output_path=Path(paths.get('output', OUTPUT_PATH)),

        # Files
        inventory_file=files.get('inventory', 'Inventory_Extract.xlsx'),
        references_file=files.get('references', 'FTRV Index & References.xlsx'),
        ssi_data_file=files.get('ssi_data', 'SSI/SSI Full Data.csv'),
        ssi_fact_tables_file=files.get('ssi_fact_tables', 'SSI/FTRV SSI Fact Tables.xlsx'),

        # Sheets
        inventory_sheet=sheets.get('inventory_records', 'Inventory Records'),
        retail_sheet=sheets.get('retail_units', 'Retail Units'),
        on_order_sheet=sheets.get('on_order', 'On Order'),
        locations_sheet=sheets.get('locations', 'Locations'),
        status_sheet=sheets.get('status_normalized', 'Status - Normalized'),
        product_sheet=sheets.get('product', 'Product'),

        # Forecast
        forecast_months=forecast.get('default_months', 3),
        lookback_months=forecast.get('lookback_months', 12),

        # Targets
        target_days_supply=targets.get('target_days_supply', 90),
        min_days_supply=targets.get('min_days_supply', 60),
        max_days_supply=targets.get('max_days_supply', 150),
        critical_days_supply=targets.get('critical_days_supply', 45),

        # Lead time
        mfg_build_days=lead_time.get('manufacturing_days', 35),
        transit_to_pdi_days=lead_time.get('transit_to_pdi_days', 14),
        pdi_processing_days=lead_time.get('pdi_processing_days', 7),
        transit_to_store_days=lead_time.get('transit_to_store_days', 14),

        # Financial
        floorplan_rate=financial.get('floorplan_rate', 0.08),
        target_turn_rate=financial.get('target_turn_rate', 4.0),
        min_acceptable_margin=financial.get('min_acceptable_margin', 0.05),

        # Seasonality
        seasonality_index=seasonality_index if seasonality_index else None,

        # Age buckets
        age_buckets=age_buckets if age_buckets else None,

        # Status categories
        sellable_statuses=status_cats.get('sellable') if status_cats.get('sellable') else None,
        constrained_statuses=status_cats.get('constrained') if status_cats.get('constrained') else None,
        dead_statuses=status_cats.get('dead') if status_cats.get('dead') else None,

        # Vehicle types
        veh_type_names=veh_types if veh_types else None,

        # Recommendations
        min_sample_size=recs.get('min_sample_size', 5),
        safety_buffer_pct=recs.get('safety_buffer_pct', 0.10),
        max_models_in_report=recs.get('max_models_in_report', 100),

        # Default filters
        filter_manufacturer=filters.get('manufacturer'),
        filter_make=filters.get('make'),
        filter_veh_types=filters.get('vehicle_types'),
        filter_locations=filters.get('locations'),
        filter_zones=filters.get('zones'),
        filter_price_groups=filters.get('price_groups'),
    )

    return config


# Create default configuration instance
# First try to load from settings.yaml, fall back to defaults
try:
    default_config = config_from_yaml()
except Exception:
    default_config = Config()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_price_group(price: float) -> str:
    """Classify price into standard price groups."""
    if price <= 0:
        return "Unknown"
    elif price < 10000:
        return "Under $10K"
    elif price < 20000:
        return "$10K-$20K"
    elif price < 30000:
        return "$20K-$30K"
    elif price < 40000:
        return "$30K-$40K"
    elif price < 50000:
        return "$40K-$50K"
    elif price < 75000:
        return "$50K-$75K"
    elif price < 100000:
        return "$75K-$100K"
    else:
        return "$100K+"


def print_current_settings():
    """Print current configuration settings for debugging."""
    config = default_config
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION SETTINGS")
    print("="*60)
    print(f"\nData Hub Path: {config.data_hub_path}")
    print(f"Output Path: {config.output_path}")
    print(f"\nForecast Months: {config.forecast_months}")
    print(f"Lookback Months: {config.lookback_months}")
    print(f"\nTarget Days Supply: {config.target_days_supply}")
    print(f"Min Days Supply: {config.min_days_supply}")
    print(f"Max Days Supply: {config.max_days_supply}")
    print(f"\nTotal Lead Time: {config.total_lead_time} days")
    print(f"  - Manufacturing: {config.mfg_build_days}")
    print(f"  - Transit to PDI: {config.transit_to_pdi_days}")
    print(f"  - PDI Processing: {config.pdi_processing_days}")
    print(f"  - Transit to Store: {config.transit_to_store_days}")
    print(f"\nFloorplan Rate: {config.floorplan_rate:.1%}")
    print(f"Safety Buffer: {config.safety_buffer_pct:.0%}")
    print("="*60 + "\n")


def reload_settings():
    """Reload settings from YAML file."""
    global default_config
    default_config = config_from_yaml()
    return default_config
