"""
FTRV Order Recommendation Engine
================================

Generates intelligent order recommendations for RV manufacturers
based on demand analysis, inventory position, and market data.

Configuration:
- Edit settings.yaml in the Claude Toolkit folder for easy configuration
- Or modify config.py for programmatic control
"""

from .config import Config, default_config, config_from_yaml, reload_settings, print_current_settings
from .data_loader import DataLoader
from .market_analyzer import MarketAnalyzer
from .pipeline_analyzer import PipelineAnalyzer
from .financial_analyzer import FinancialAnalyzer
from .recommendation_engine import RecommendationEngine
from .report_generator import ReportGenerator
from .yoy_analyzer import YoYAnalyzer
from .transparency import TransparencyGenerator, RecommendationExplanation
from .monthly_planner import MonthlyPlanner
from .order_form_parser import OrderFormParser, ParseResult, ColumnMapping
from .brand_licensing import BrandLicensingAnalyzer
from .capacity_planner import CapacityPlanner, LocationCapacity, TypeNeed
from .data_quality import DataQualityAnalyzer, DataQualityReportGenerator, DataCleaner
from .exceptions_config import ExceptionsManager, ERROR_CODES, get_error_description
from .reconciliation import ReconciliationEngine, ReconciliationReportGenerator

__version__ = "1.7.0"
__all__ = [
    "Config",
    "default_config",
    "config_from_yaml",
    "reload_settings",
    "print_current_settings",
    "DataLoader",
    "MarketAnalyzer",
    "PipelineAnalyzer",
    "FinancialAnalyzer",
    "RecommendationEngine",
    "ReportGenerator",
    "YoYAnalyzer",
    "TransparencyGenerator",
    "RecommendationExplanation",
    "MonthlyPlanner",
    "OrderFormParser",
    "ParseResult",
    "ColumnMapping",
    "BrandLicensingAnalyzer",
    "CapacityPlanner",
    "LocationCapacity",
    "TypeNeed",
    "DataQualityAnalyzer",
    "DataQualityReportGenerator",
    "DataCleaner",
    "ExceptionsManager",
    "ERROR_CODES",
    "get_error_description",
    "ReconciliationEngine",
    "ReconciliationReportGenerator",
]
