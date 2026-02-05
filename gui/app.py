"""
GUI Application for FTRV Order Recommendation Generator
Phase 4 Enhanced Version - Tabbed Interface with Capacity, Reconciliation, and YoY Dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from pathlib import Path
import sys
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from order_engine.config import Config, default_config
from order_engine.data_loader import DataLoader
from order_engine.recommendation_engine import RecommendationEngine
from order_engine.report_generator import ReportGenerator


class OrderRecommendationApp:
    """GUI Application for generating order recommendations."""

    def __init__(self, root):
        self.root = root
        self.root.title("FTRV Order Recommendation Generator")
        self.root.geometry("1000x850")
        self.root.minsize(900, 750)

        # Data
        self.data_loader = None
        self.manufacturers = []
        self.makes = {}  # Dict: manufacturer -> list of makes
        self.zones = []
        self.veh_types = []

        # Build UI
        self._create_widgets()
        self._load_reference_data()

    def _create_widgets(self):
        """Create all UI widgets using tabbed interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="FTRV Order Recommendation Generator",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Create Notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Create tabs
        self.order_tab = ttk.Frame(self.notebook)
        self.capacity_tab = ttk.Frame(self.notebook)
        self.reconciliation_tab = ttk.Frame(self.notebook)
        self.yoy_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.order_tab, text="Order Generator")
        self.notebook.add(self.capacity_tab, text="Capacity Analysis")
        self.notebook.add(self.reconciliation_tab, text="Reconciliation")
        self.notebook.add(self.yoy_tab, text="YoY Dashboard")

        # Build each tab
        self._create_order_tab()
        self._create_capacity_tab()
        self._create_reconciliation_tab()
        self._create_yoy_tab()

        # Status bar at bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

    # =========================================================================
    # ORDER GENERATOR TAB (existing functionality)
    # =========================================================================

    def _create_order_tab(self):
        """Create the Order Generator tab (existing functionality)."""
        # Create scrollable frame
        canvas = tk.Canvas(self.order_tab)
        scrollbar = ttk.Scrollbar(self.order_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        subtitle = ttk.Label(
            main_frame,
            text="Make-Level Orders: Type -> Manufacturer -> Make",
            font=("Helvetica", 10, "italic")
        )
        subtitle.pack(pady=(0, 10))

        # ================================================================
        # MANUFACTURER SELECTION
        # ================================================================
        mfg_frame = ttk.LabelFrame(main_frame, text="1. Select Manufacturer", padding="10")
        mfg_frame.pack(fill=tk.X, pady=5)

        mfg_inner = ttk.Frame(mfg_frame)
        mfg_inner.pack(fill=tk.X)
        ttk.Label(mfg_inner, text="Manufacturer:", width=15).pack(side=tk.LEFT)
        self.mfg_var = tk.StringVar()
        self.mfg_combo = ttk.Combobox(mfg_inner, textvariable=self.mfg_var, width=40, state="readonly")
        self.mfg_combo.pack(side=tk.LEFT, padx=5)
        self.mfg_combo.bind("<<ComboboxSelected>>", self._on_manufacturer_changed)

        # ================================================================
        # MAKE SELECTION (MULTI-SELECT)
        # ================================================================
        make_frame = ttk.LabelFrame(main_frame, text="2. Select Makes to Order (Multi-Select)", padding="10")
        make_frame.pack(fill=tk.X, pady=5)

        make_help = ttk.Label(
            make_frame,
            text="Select which makes you want to order. Discontinued makes (not selected) will have their volume redistributed to selected makes.",
            wraplength=700,
            foreground="gray"
        )
        make_help.pack(anchor=tk.W, pady=(0, 5))

        make_controls = ttk.Frame(make_frame)
        make_controls.pack(fill=tk.X, pady=2)

        ttk.Button(make_controls, text="Select All", command=self._select_all_makes).pack(side=tk.LEFT, padx=2)
        ttk.Button(make_controls, text="Clear All", command=self._clear_all_makes).pack(side=tk.LEFT, padx=2)

        self.make_count_var = tk.StringVar(value="0 makes selected")
        ttk.Label(make_controls, textvariable=self.make_count_var, foreground="blue").pack(side=tk.LEFT, padx=20)

        make_list_frame = ttk.Frame(make_frame)
        make_list_frame.pack(fill=tk.X, pady=5)

        self.make_listbox = tk.Listbox(
            make_list_frame,
            selectmode=tk.MULTIPLE,
            height=6,
            exportselection=False
        )
        self.make_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.make_listbox.bind("<<ListboxSelect>>", self._on_makes_changed)

        make_scroll = ttk.Scrollbar(make_list_frame, command=self.make_listbox.yview)
        make_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.make_listbox.config(yscrollcommand=make_scroll.set)

        # ================================================================
        # FORECAST PERIOD (Calendar Month Pickers)
        # ================================================================
        period_frame = ttk.LabelFrame(main_frame, text="3. Forecast Period", padding="10")
        period_frame.pack(fill=tk.X, pady=5)

        period_type_frame = ttk.Frame(period_frame)
        period_type_frame.pack(fill=tk.X, pady=2)

        self.period_type_var = tk.StringVar(value="rolling")

        ttk.Radiobutton(
            period_type_frame,
            text="Rolling Period:",
            variable=self.period_type_var,
            value="rolling",
            command=self._on_period_type_changed
        ).pack(side=tk.LEFT)

        self.period_var = tk.StringVar(value="3")
        self.period_combo = ttk.Combobox(
            period_type_frame,
            textvariable=self.period_var,
            values=["3", "4", "6", "9", "12"],
            width=5,
            state="readonly"
        )
        self.period_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(period_type_frame, text="months from today").pack(side=tk.LEFT)

        calendar_frame = ttk.Frame(period_frame)
        calendar_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(
            calendar_frame,
            text="Calendar Months:",
            variable=self.period_type_var,
            value="calendar",
            command=self._on_period_type_changed
        ).pack(side=tk.LEFT)

        ttk.Label(calendar_frame, text="From:").pack(side=tk.LEFT, padx=(10, 2))
        self.start_month_var = tk.StringVar()
        self.start_month_combo = ttk.Combobox(
            calendar_frame,
            textvariable=self.start_month_var,
            width=12,
            state="disabled"
        )
        self.start_month_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(calendar_frame, text="To:").pack(side=tk.LEFT, padx=(10, 2))
        self.end_month_var = tk.StringVar()
        self.end_month_combo = ttk.Combobox(
            calendar_frame,
            textvariable=self.end_month_var,
            width=12,
            state="disabled"
        )
        self.end_month_combo.pack(side=tk.LEFT, padx=2)

        self._populate_month_combos()

        # ================================================================
        # VEHICLE TYPES
        # ================================================================
        veh_frame = ttk.LabelFrame(main_frame, text="4. Vehicle Types", padding="10")
        veh_frame.pack(fill=tk.X, pady=5)

        veh_help = ttk.Label(
            veh_frame,
            text="Select vehicle types to include. Leave all unchecked for all types.",
            foreground="gray"
        )
        veh_help.pack(anchor=tk.W, pady=(0, 5))

        self.veh_type_vars = {}
        self.veh_check_frame = ttk.Frame(veh_frame)
        self.veh_check_frame.pack(anchor=tk.W)

        # ================================================================
        # SEGMENT VIEW OPTIONS
        # ================================================================
        view_frame = ttk.LabelFrame(main_frame, text="5. Report View Options", padding="10")
        view_frame.pack(fill=tk.X, pady=5)

        view_inner = ttk.Frame(view_frame)
        view_inner.pack(fill=tk.X)

        ttk.Label(view_inner, text="Group by:").pack(side=tk.LEFT)
        self.view_by_var = tk.StringVar(value="make")
        for view in [("Make", "make"), ("Type", "type"), ("Zone", "zone"), ("Model", "model")]:
            ttk.Radiobutton(
                view_inner,
                text=view[0],
                variable=self.view_by_var,
                value=view[1]
            ).pack(side=tk.LEFT, padx=10)

        # ================================================================
        # CONSTRAINTS
        # ================================================================
        const_frame = ttk.LabelFrame(main_frame, text="6. Constraints (Optional)", padding="10")
        const_frame.pack(fill=tk.X, pady=5)

        budget_frame = ttk.Frame(const_frame)
        budget_frame.pack(fill=tk.X, pady=2)
        ttk.Label(budget_frame, text="Max Budget:", width=15).pack(side=tk.LEFT)
        self.budget_var = tk.StringVar()
        ttk.Entry(budget_frame, textvariable=self.budget_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(budget_frame, text="(e.g., 500000)").pack(side=tk.LEFT)

        units_frame = ttk.Frame(const_frame)
        units_frame.pack(fill=tk.X, pady=2)
        ttk.Label(units_frame, text="Max Units:", width=15).pack(side=tk.LEFT)
        self.units_var = tk.StringVar()
        ttk.Entry(units_frame, textvariable=self.units_var, width=20).pack(side=tk.LEFT, padx=5)

        # ================================================================
        # FILTERS
        # ================================================================
        filter_frame = ttk.LabelFrame(main_frame, text="7. Location Filters (Optional)", padding="10")
        filter_frame.pack(fill=tk.X, pady=5)

        zone_frame = ttk.Frame(filter_frame)
        zone_frame.pack(fill=tk.X, pady=2)
        ttk.Label(zone_frame, text="Zone:", width=15).pack(side=tk.LEFT)
        self.zones_var = tk.StringVar()
        self.zones_combo = ttk.Combobox(zone_frame, textvariable=self.zones_var, width=40)
        self.zones_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(zone_frame, text="(leave blank for all)").pack(side=tk.LEFT)

        # ================================================================
        # BUTTONS
        # ================================================================
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.generate_btn = ttk.Button(
            btn_frame,
            text="Generate Order Plan",
            command=self._generate_orders,
            style="Accent.TButton"
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Upload Order Form",
            command=self._upload_order_form
        ).pack(side=tk.LEFT, padx=5)

        self.refresh_btn = ttk.Button(
            btn_frame,
            text="Refresh Data",
            command=self._refresh_data
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Open Output Folder",
            command=self._open_output_folder
        ).pack(side=tk.LEFT, padx=5)

        # ================================================================
        # RESULTS
        # ================================================================
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.results_text = tk.Text(results_frame, height=12, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        results_scroll = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scroll.set)

    # =========================================================================
    # CAPACITY ANALYSIS TAB
    # =========================================================================

    def _create_capacity_tab(self):
        """Create the Capacity Analysis tab."""
        # Main content frame
        main_frame = ttk.Frame(self.capacity_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        subtitle = ttk.Label(
            main_frame,
            text="Top-Down Analysis: What do we need? -> Where? -> Who can supply it?",
            font=("Helvetica", 10, "italic")
        )
        subtitle.pack(pady=(0, 10))

        # ================================================================
        # FILTERS
        # ================================================================
        filter_frame = ttk.LabelFrame(main_frame, text="Filters", padding="10")
        filter_frame.pack(fill=tk.X, pady=5)

        # Vehicle Types row
        veh_row = ttk.Frame(filter_frame)
        veh_row.pack(fill=tk.X, pady=2)
        ttk.Label(veh_row, text="Vehicle Types:").pack(side=tk.LEFT)

        self.cap_veh_type_vars = {}
        for vt in ["TT", "FW", "MH", "TH"]:
            var = tk.BooleanVar(value=True)
            self.cap_veh_type_vars[vt] = var
            ttk.Checkbutton(veh_row, text=vt, variable=var).pack(side=tk.LEFT, padx=10)

        # Zone row
        zone_row = ttk.Frame(filter_frame)
        zone_row.pack(fill=tk.X, pady=2)
        ttk.Label(zone_row, text="Zone:", width=12).pack(side=tk.LEFT)
        self.cap_zone_var = tk.StringVar(value="All Zones")
        self.cap_zone_combo = ttk.Combobox(zone_row, textvariable=self.cap_zone_var, width=30, state="readonly")
        self.cap_zone_combo.pack(side=tk.LEFT, padx=5)

        # ================================================================
        # SUMMARY SECTION
        # ================================================================
        summary_frame = ttk.LabelFrame(main_frame, text="Company-Wide Summary", padding="10")
        summary_frame.pack(fill=tk.X, pady=5)

        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack(fill=tk.X)

        # Row 1
        ttk.Label(summary_grid, text="Lot Capacity:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.cap_lot_capacity_var = tk.StringVar(value="--")
        ttk.Label(summary_grid, textvariable=self.cap_lot_capacity_var, font=("Helvetica", 10, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(summary_grid, text="Current Inventory:").grid(row=0, column=2, sticky=tk.W, padx=15)
        self.cap_current_inv_var = tk.StringVar(value="--")
        ttk.Label(summary_grid, textvariable=self.cap_current_inv_var, font=("Helvetica", 10, "bold")).grid(row=0, column=3, sticky=tk.W, padx=5)

        # Row 2
        ttk.Label(summary_grid, text="Pipeline:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.cap_pipeline_var = tk.StringVar(value="--")
        ttk.Label(summary_grid, textvariable=self.cap_pipeline_var, font=("Helvetica", 10, "bold")).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(summary_grid, text="Total Gap:").grid(row=1, column=2, sticky=tk.W, padx=15)
        self.cap_gap_var = tk.StringVar(value="--")
        ttk.Label(summary_grid, textvariable=self.cap_gap_var, font=("Helvetica", 10, "bold"), foreground="red").grid(row=1, column=3, sticky=tk.W, padx=5)

        # ================================================================
        # RESULTS TREEVIEWS
        # ================================================================
        results_pane = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        results_pane.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left pane: Gap by Zone
        zone_frame = ttk.LabelFrame(results_pane, text="Gap by Zone", padding="5")
        results_pane.add(zone_frame, weight=1)

        self.cap_zone_tree = ttk.Treeview(
            zone_frame,
            columns=("zone", "capacity", "current", "pipeline", "gap", "fill"),
            show="headings",
            height=8
        )
        self.cap_zone_tree.heading("zone", text="Zone")
        self.cap_zone_tree.heading("capacity", text="Capacity")
        self.cap_zone_tree.heading("current", text="Current")
        self.cap_zone_tree.heading("pipeline", text="Pipeline")
        self.cap_zone_tree.heading("gap", text="Gap")
        self.cap_zone_tree.heading("fill", text="Fill %")

        for col in ("zone", "capacity", "current", "pipeline", "gap", "fill"):
            self.cap_zone_tree.column(col, width=70, anchor=tk.CENTER)
        self.cap_zone_tree.column("zone", width=100, anchor=tk.W)

        zone_scroll = ttk.Scrollbar(zone_frame, orient=tk.VERTICAL, command=self.cap_zone_tree.yview)
        self.cap_zone_tree.configure(yscrollcommand=zone_scroll.set)
        self.cap_zone_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        zone_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Right pane: Gap by Type
        type_frame = ttk.LabelFrame(results_pane, text="Gap by Vehicle Type", padding="5")
        results_pane.add(type_frame, weight=1)

        self.cap_type_tree = ttk.Treeview(
            type_frame,
            columns=("type", "capacity", "current", "pipeline", "gap", "fill"),
            show="headings",
            height=8
        )
        self.cap_type_tree.heading("type", text="Type")
        self.cap_type_tree.heading("capacity", text="Capacity")
        self.cap_type_tree.heading("current", text="Current")
        self.cap_type_tree.heading("pipeline", text="Pipeline")
        self.cap_type_tree.heading("gap", text="Gap")
        self.cap_type_tree.heading("fill", text="Fill %")

        for col in ("type", "capacity", "current", "pipeline", "gap", "fill"):
            self.cap_type_tree.column(col, width=70, anchor=tk.CENTER)
        self.cap_type_tree.column("type", width=60, anchor=tk.W)

        type_scroll = ttk.Scrollbar(type_frame, orient=tk.VERTICAL, command=self.cap_type_tree.yview)
        self.cap_type_tree.configure(yscrollcommand=type_scroll.set)
        self.cap_type_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        type_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ================================================================
        # TOP LOCATION GAPS
        # ================================================================
        loc_gaps_frame = ttk.LabelFrame(main_frame, text="Top Location Gaps", padding="5")
        loc_gaps_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.cap_location_text = tk.Text(loc_gaps_frame, height=6, wrap=tk.WORD)
        self.cap_location_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        loc_scroll = ttk.Scrollbar(loc_gaps_frame, command=self.cap_location_text.yview)
        loc_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.cap_location_text.config(yscrollcommand=loc_scroll.set)

        # ================================================================
        # BUTTONS
        # ================================================================
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="Run Capacity Analysis",
            command=self._run_capacity_analysis
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Generate Excel Report",
            command=self._generate_capacity_excel
        ).pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # RECONCILIATION TAB
    # =========================================================================

    def _create_reconciliation_tab(self):
        """Create the Reconciliation tab."""
        # Main content frame
        main_frame = ttk.Frame(self.reconciliation_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        subtitle = ttk.Label(
            main_frame,
            text="Allocate capacity needs across manufacturers with strategic adjustments",
            font=("Helvetica", 10, "italic")
        )
        subtitle.pack(pady=(0, 10))

        # ================================================================
        # PLANNING MODE
        # ================================================================
        mode_frame = ttk.LabelFrame(main_frame, text="Planning Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=5)

        mode_row = ttk.Frame(mode_frame)
        mode_row.pack(fill=tk.X)

        self.recon_mode_var = tk.StringVar(value="current")

        ttk.Radiobutton(
            mode_row,
            text="Current State (fill current gaps)",
            variable=self.recon_mode_var,
            value="current",
            command=self._on_recon_mode_changed
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            mode_row,
            text="Projected (plan future orders)",
            variable=self.recon_mode_var,
            value="projected",
            command=self._on_recon_mode_changed
        ).pack(side=tk.LEFT, padx=20)

        # ================================================================
        # PLANNING PARAMETERS (enabled when Projected selected)
        # ================================================================
        self.recon_params_frame = ttk.LabelFrame(main_frame, text="Planning Parameters", padding="10")
        self.recon_params_frame.pack(fill=tk.X, pady=5)

        params_row1 = ttk.Frame(self.recon_params_frame)
        params_row1.pack(fill=tk.X, pady=2)

        ttk.Label(params_row1, text="Planning Months:", width=15).pack(side=tk.LEFT)
        self.recon_months_var = tk.StringVar(value="3")
        self.recon_months_combo = ttk.Combobox(
            params_row1,
            textvariable=self.recon_months_var,
            values=["3", "6", "12"],
            width=10,
            state="disabled"
        )
        self.recon_months_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(params_row1, text="Target Fill %:", width=15).pack(side=tk.LEFT, padx=(20, 0))
        self.recon_fill_var = tk.IntVar(value=90)
        self.recon_fill_scale = ttk.Scale(
            params_row1,
            from_=60,
            to=100,
            variable=self.recon_fill_var,
            orient=tk.HORIZONTAL,
            length=150,
            state="disabled"
        )
        self.recon_fill_scale.pack(side=tk.LEFT, padx=5)
        self.recon_fill_label = ttk.Label(params_row1, text="90%")
        self.recon_fill_label.pack(side=tk.LEFT)
        self.recon_fill_var.trace_add("write", self._update_fill_label)

        params_row2 = ttk.Frame(self.recon_params_frame)
        params_row2.pack(fill=tk.X, pady=2)

        ttk.Label(params_row2, text="Manufacturer Filter:", width=18).pack(side=tk.LEFT)
        self.recon_mfg_var = tk.StringVar(value="All Manufacturers")
        self.recon_mfg_combo = ttk.Combobox(
            params_row2,
            textvariable=self.recon_mfg_var,
            width=30,
            state="readonly"
        )
        self.recon_mfg_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(params_row2, text="(filters to licensed locations)", foreground="gray").pack(side=tk.LEFT, padx=5)

        # ================================================================
        # ADJUSTMENTS PREVIEW
        # ================================================================
        adj_frame = ttk.LabelFrame(main_frame, text="Strategic Adjustments (from order_adjustments.yaml)", padding="10")
        adj_frame.pack(fill=tk.X, pady=5)

        self.recon_adj_tree = ttk.Treeview(
            adj_frame,
            columns=("type", "target", "adjustment", "note"),
            show="headings",
            height=4
        )
        self.recon_adj_tree.heading("type", text="Type")
        self.recon_adj_tree.heading("target", text="Target")
        self.recon_adj_tree.heading("adjustment", text="Adjustment")
        self.recon_adj_tree.heading("note", text="Note")

        self.recon_adj_tree.column("type", width=100)
        self.recon_adj_tree.column("target", width=150)
        self.recon_adj_tree.column("adjustment", width=80)
        self.recon_adj_tree.column("note", width=300)

        self.recon_adj_tree.pack(fill=tk.X, side=tk.TOP)

        adj_btn_frame = ttk.Frame(adj_frame)
        adj_btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            adj_btn_frame,
            text="Edit Adjustments",
            command=self._edit_adjustments
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            adj_btn_frame,
            text="Reload Adjustments",
            command=self._reload_adjustments
        ).pack(side=tk.LEFT, padx=5)

        # ================================================================
        # RESULTS SUMMARY
        # ================================================================
        results_frame = ttk.LabelFrame(main_frame, text="Results Summary", padding="10")
        results_frame.pack(fill=tk.X, pady=5)

        results_row = ttk.Frame(results_frame)
        results_row.pack(fill=tk.X)

        ttk.Label(results_row, text="Total Gap:").pack(side=tk.LEFT, padx=5)
        self.recon_total_gap_var = tk.StringVar(value="--")
        ttk.Label(results_row, textvariable=self.recon_total_gap_var, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)

        ttk.Label(results_row, text="Total Allocated:").pack(side=tk.LEFT, padx=15)
        self.recon_total_alloc_var = tk.StringVar(value="--")
        ttk.Label(results_row, textvariable=self.recon_total_alloc_var, font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)

        # ================================================================
        # ALLOCATION TREEVIEW
        # ================================================================
        alloc_frame = ttk.LabelFrame(main_frame, text="Manufacturer Allocations", padding="5")
        alloc_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.recon_alloc_tree = ttk.Treeview(
            alloc_frame,
            columns=("mfg", "base", "adjusted", "share", "delta"),
            show="headings",
            height=10
        )
        self.recon_alloc_tree.heading("mfg", text="Manufacturer")
        self.recon_alloc_tree.heading("base", text="Base Units")
        self.recon_alloc_tree.heading("adjusted", text="Adjusted Units")
        self.recon_alloc_tree.heading("share", text="Share %")
        self.recon_alloc_tree.heading("delta", text="Delta")

        self.recon_alloc_tree.column("mfg", width=200, anchor=tk.W)
        self.recon_alloc_tree.column("base", width=100, anchor=tk.CENTER)
        self.recon_alloc_tree.column("adjusted", width=100, anchor=tk.CENTER)
        self.recon_alloc_tree.column("share", width=80, anchor=tk.CENTER)
        self.recon_alloc_tree.column("delta", width=80, anchor=tk.CENTER)

        alloc_scroll = ttk.Scrollbar(alloc_frame, orient=tk.VERTICAL, command=self.recon_alloc_tree.yview)
        self.recon_alloc_tree.configure(yscrollcommand=alloc_scroll.set)
        self.recon_alloc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        alloc_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ================================================================
        # BUTTONS
        # ================================================================
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="Run Reconciliation",
            command=self._run_reconciliation
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Generate Excel Report",
            command=self._generate_reconciliation_excel
        ).pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # YOY DASHBOARD TAB
    # =========================================================================

    def _create_yoy_tab(self):
        """Create the YoY Dashboard tab."""
        # Main content frame
        main_frame = ttk.Frame(self.yoy_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        subtitle = ttk.Label(
            main_frame,
            text="Year-over-Year Performance Analysis with Location Adjustments",
            font=("Helvetica", 10, "italic")
        )
        subtitle.pack(pady=(0, 10))

        # ================================================================
        # FILTERS
        # ================================================================
        filter_frame = ttk.LabelFrame(main_frame, text="Filters", padding="10")
        filter_frame.pack(fill=tk.X, pady=5)

        filter_row = ttk.Frame(filter_frame)
        filter_row.pack(fill=tk.X)

        ttk.Label(filter_row, text="Manufacturer:").pack(side=tk.LEFT, padx=5)
        self.yoy_mfg_var = tk.StringVar(value="All Manufacturers")
        self.yoy_mfg_combo = ttk.Combobox(
            filter_row,
            textvariable=self.yoy_mfg_var,
            width=25,
            state="readonly"
        )
        self.yoy_mfg_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_row, text="Vehicle Type:").pack(side=tk.LEFT, padx=(20, 5))
        self.yoy_veh_var = tk.StringVar(value="All Types")
        self.yoy_veh_combo = ttk.Combobox(
            filter_row,
            textvariable=self.yoy_veh_var,
            width=15,
            state="readonly"
        )
        self.yoy_veh_combo.pack(side=tk.LEFT, padx=5)

        self.yoy_adj_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            filter_row,
            text="Location-adjusted",
            variable=self.yoy_adj_var
        ).pack(side=tk.LEFT, padx=20)

        # ================================================================
        # YTD COMPARISON SUMMARY
        # ================================================================
        ytd_frame = ttk.LabelFrame(main_frame, text="YTD Comparison", padding="10")
        ytd_frame.pack(fill=tk.X, pady=5)

        ytd_grid = ttk.Frame(ytd_frame)
        ytd_grid.pack(fill=tk.X)

        # Row headers
        ttk.Label(ytd_grid, text="").grid(row=0, column=0, padx=5)
        ttk.Label(ytd_grid, text="Last Year", font=("Helvetica", 9, "bold")).grid(row=0, column=1, padx=10)
        ttk.Label(ytd_grid, text="This Year", font=("Helvetica", 9, "bold")).grid(row=0, column=2, padx=10)
        ttk.Label(ytd_grid, text="Change", font=("Helvetica", 9, "bold")).grid(row=0, column=3, padx=10)
        ttk.Label(ytd_grid, text="Adj. Change", font=("Helvetica", 9, "bold")).grid(row=0, column=4, padx=10)

        # Units row
        ttk.Label(ytd_grid, text="Units:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.yoy_ly_units_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_ly_units_var).grid(row=1, column=1, padx=10)
        self.yoy_cy_units_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_cy_units_var).grid(row=1, column=2, padx=10)
        self.yoy_change_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_change_var).grid(row=1, column=3, padx=10)
        self.yoy_adj_change_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_adj_change_var).grid(row=1, column=4, padx=10)

        # Locations row
        ttk.Label(ytd_grid, text="Locations:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.yoy_ly_locs_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_ly_locs_var).grid(row=2, column=1, padx=10)
        self.yoy_cy_locs_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_cy_locs_var).grid(row=2, column=2, padx=10)
        self.yoy_loc_change_var = tk.StringVar(value="--")
        ttk.Label(ytd_grid, textvariable=self.yoy_loc_change_var).grid(row=2, column=3, padx=10)

        # Interpretation
        self.yoy_interp_var = tk.StringVar(value="")
        ttk.Label(ytd_frame, textvariable=self.yoy_interp_var, foreground="blue", wraplength=600).pack(anchor=tk.W, pady=(10, 0))

        # ================================================================
        # QUARTERLY COMPARISON
        # ================================================================
        quarterly_frame = ttk.LabelFrame(main_frame, text="Quarterly Comparison", padding="5")
        quarterly_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.yoy_quarterly_tree = ttk.Treeview(
            quarterly_frame,
            columns=("quarter", "last_year", "this_year", "change", "adj_change"),
            show="headings",
            height=5
        )
        self.yoy_quarterly_tree.heading("quarter", text="Quarter")
        self.yoy_quarterly_tree.heading("last_year", text="Last Year")
        self.yoy_quarterly_tree.heading("this_year", text="This Year")
        self.yoy_quarterly_tree.heading("change", text="Change %")
        self.yoy_quarterly_tree.heading("adj_change", text="Adj. Change %")

        for col in self.yoy_quarterly_tree["columns"]:
            self.yoy_quarterly_tree.column(col, width=100, anchor=tk.CENTER)

        self.yoy_quarterly_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        q_scroll = ttk.Scrollbar(quarterly_frame, orient=tk.VERTICAL, command=self.yoy_quarterly_tree.yview)
        self.yoy_quarterly_tree.configure(yscrollcommand=q_scroll.set)
        q_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ================================================================
        # MONTHLY COMPARISON
        # ================================================================
        monthly_frame = ttk.LabelFrame(main_frame, text="Monthly Comparison", padding="5")
        monthly_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.yoy_monthly_tree = ttk.Treeview(
            monthly_frame,
            columns=("month", "last_year", "this_year", "change"),
            show="headings",
            height=6
        )
        self.yoy_monthly_tree.heading("month", text="Month")
        self.yoy_monthly_tree.heading("last_year", text="Last Year")
        self.yoy_monthly_tree.heading("this_year", text="This Year")
        self.yoy_monthly_tree.heading("change", text="Change %")

        for col in self.yoy_monthly_tree["columns"]:
            self.yoy_monthly_tree.column(col, width=100, anchor=tk.CENTER)

        self.yoy_monthly_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        m_scroll = ttk.Scrollbar(monthly_frame, orient=tk.VERTICAL, command=self.yoy_monthly_tree.yview)
        self.yoy_monthly_tree.configure(yscrollcommand=m_scroll.set)
        m_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ================================================================
        # BUTTONS
        # ================================================================
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="Refresh Data",
            command=self._refresh_yoy_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Export to Excel",
            command=self._export_yoy_excel
        ).pack(side=tk.LEFT, padx=5)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _populate_month_combos(self):
        """Populate the start/end month combo boxes with next 18 months."""
        months = []
        now = datetime.now()
        for i in range(18):
            month_date = now + relativedelta(months=i)
            month_str = month_date.strftime("%b %Y")
            months.append(month_str)

        self.start_month_combo['values'] = months
        self.end_month_combo['values'] = months

        if months:
            self.start_month_var.set(months[0])
            self.end_month_var.set(months[3] if len(months) > 3 else months[-1])

    def _on_period_type_changed(self):
        """Handle period type radio button change."""
        if self.period_type_var.get() == "rolling":
            self.period_combo.config(state="readonly")
            self.start_month_combo.config(state="disabled")
            self.end_month_combo.config(state="disabled")
        else:
            self.period_combo.config(state="disabled")
            self.start_month_combo.config(state="readonly")
            self.end_month_combo.config(state="readonly")

    def _on_recon_mode_changed(self):
        """Handle reconciliation mode change."""
        if self.recon_mode_var.get() == "projected":
            self.recon_months_combo.config(state="readonly")
            self.recon_fill_scale.config(state="normal")
        else:
            self.recon_months_combo.config(state="disabled")
            self.recon_fill_scale.config(state="disabled")

    def _update_fill_label(self, *args):
        """Update the fill percentage label."""
        self.recon_fill_label.config(text=f"{self.recon_fill_var.get()}%")

    def _load_reference_data(self):
        """Load reference data in background."""
        self.progress_var.set("Loading reference data...")
        self.progress_bar.start()

        def load():
            try:
                self.data_loader = DataLoader()
                self.manufacturers = self.data_loader.get_manufacturers()
                self.zones = self.data_loader.get_zones()
                self.veh_types = self.data_loader.get_veh_types()

                self.root.after(0, self._update_combos)
            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"Error loading data: {e}"))
            finally:
                self.root.after(0, self._loading_complete)

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _update_combos(self):
        """Update combo boxes with loaded data."""
        # Order tab
        self.mfg_combo['values'] = self.manufacturers
        if self.manufacturers:
            self.mfg_combo.current(0)
            self._on_manufacturer_changed(None)

        self.zones_combo['values'] = [""] + self.zones

        # Create vehicle type checkboxes in Order tab
        for widget in self.veh_check_frame.winfo_children():
            widget.destroy()

        for vt in self.veh_types[:6]:
            var = tk.BooleanVar(value=False)
            self.veh_type_vars[vt] = var
            cb = ttk.Checkbutton(self.veh_check_frame, text=vt, variable=var)
            cb.pack(side=tk.LEFT, padx=5)

        # Capacity tab
        self.cap_zone_combo['values'] = ["All Zones"] + self.zones

        # Reconciliation tab
        self.recon_mfg_combo['values'] = ["All Manufacturers"] + self.manufacturers

        # YoY tab
        self.yoy_mfg_combo['values'] = ["All Manufacturers"] + self.manufacturers
        self.yoy_veh_combo['values'] = ["All Types"] + self.veh_types

        # Load adjustments
        self._reload_adjustments()

    def _on_manufacturer_changed(self, event):
        """Update make listbox when manufacturer changes."""
        manufacturer = self.mfg_var.get().strip()
        self.make_listbox.delete(0, tk.END)

        if not manufacturer:
            self.make_count_var.set("0 makes selected")
            return

        if manufacturer in self.makes:
            makes_list = self.makes[manufacturer]
        else:
            if self.data_loader:
                try:
                    makes_list = self.data_loader.get_makes_for_manufacturer(manufacturer)
                    self.makes[manufacturer] = makes_list
                except Exception:
                    makes_list = []
            else:
                makes_list = []

        for make in makes_list:
            self.make_listbox.insert(tk.END, make)

        self._select_all_makes()

    def _on_makes_changed(self, event=None):
        """Update make count when selection changes."""
        selected = self.make_listbox.curselection()
        total = self.make_listbox.size()
        self.make_count_var.set(f"{len(selected)} of {total} makes selected")

    def _select_all_makes(self):
        """Select all makes in the listbox."""
        self.make_listbox.select_set(0, tk.END)
        self._on_makes_changed()

    def _clear_all_makes(self):
        """Clear all make selections."""
        self.make_listbox.selection_clear(0, tk.END)
        self._on_makes_changed()

    def _loading_complete(self):
        """Called when loading is complete."""
        self.progress_bar.stop()
        self.progress_var.set("Ready - Select a tab to begin")

    def _show_error(self, message):
        """Show error message."""
        self.progress_bar.stop()
        self.progress_var.set("Error")
        messagebox.showerror("Error", message)

    def _refresh_data(self):
        """Refresh all data."""
        if self.data_loader:
            self.data_loader.clear_cache()
        self._load_reference_data()

    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_path = default_config.output_path
        output_path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(f'explorer "{output_path}"')

    def _get_selected_makes(self):
        """Get list of selected makes from the listbox."""
        selected_indices = self.make_listbox.curselection()
        return [self.make_listbox.get(i) for i in selected_indices]

    # =========================================================================
    # ORDER GENERATOR METHODS
    # =========================================================================

    def _generate_orders(self):
        """Generate order plan at MAKE level with selected makes."""
        manufacturer = self.mfg_var.get().strip()
        if not manufacturer:
            messagebox.showwarning("Warning", "Please select a manufacturer")
            return

        selected_makes = self._get_selected_makes()
        if not selected_makes:
            if not messagebox.askyesno(
                "No Makes Selected",
                "No makes selected. This will generate orders for ALL makes.\n\n"
                "Do you want to continue?"
            ):
                return
            selected_makes = None

        if self.period_type_var.get() == "rolling":
            try:
                period = int(self.period_var.get())
            except ValueError:
                period = 3
            start_month = None
            end_month = None
        else:
            period = None
            start_month = self.start_month_var.get()
            end_month = self.end_month_var.get()

        budget = None
        if self.budget_var.get().strip():
            try:
                budget = float(self.budget_var.get().replace(",", "").replace("$", ""))
            except ValueError:
                messagebox.showwarning("Warning", "Invalid budget value")
                return

        units = None
        if self.units_var.get().strip():
            try:
                units = int(self.units_var.get())
            except ValueError:
                messagebox.showwarning("Warning", "Invalid units value")
                return

        veh_types = [vt for vt, var in self.veh_type_vars.items() if var.get()]
        if not veh_types:
            veh_types = None

        zone = self.zones_var.get().strip()
        zones = [zone] if zone else None

        view_by = self.view_by_var.get()

        make_desc = f"{len(selected_makes)} makes" if selected_makes else "all makes"
        order_desc = f"{manufacturer} ({make_desc})"

        self.generate_btn.config(state=tk.DISABLED)
        self.progress_var.set(f"Generating order plan for {order_desc}...")
        self.progress_bar.start()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Processing order plan for {order_desc}...\n")

        if selected_makes and len(selected_makes) < self.make_listbox.size():
            excluded = self.make_listbox.size() - len(selected_makes)
            self.results_text.insert(
                tk.END,
                f"\nNote: {excluded} discontinued/excluded makes - volume redistributed to selected makes.\n\n"
            )

        def generate():
            try:
                config = default_config
                if zones:
                    config = config.with_filters(zones=zones)

                engine = RecommendationEngine(config=config)
                report_gen = ReportGenerator(config=config)

                order_plan = engine.generate_order_plan(
                    manufacturer=manufacturer,
                    makes=selected_makes,
                    budget=budget,
                    target_units=units,
                    forecast_months=period,
                    start_month=start_month,
                    end_month=end_month,
                    veh_types=veh_types,
                    view_by=view_by
                )

                if order_plan.get("status") == "error":
                    self.root.after(0, lambda: self._show_results_error(order_plan.get("message", "Unknown error")))
                    return

                output_file = report_gen.generate_order_plan_report(order_plan)
                summary_text = report_gen.generate_quick_summary(order_plan)

                self.root.after(0, lambda: self._show_results_success(summary_text, output_file))

            except Exception as e:
                import traceback
                error_msg = f"Error: {e}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self._show_results_error(error_msg))

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def _show_results_success(self, summary_text, output_file):
        """Show successful results."""
        self.progress_bar.stop()
        self.generate_btn.config(state=tk.NORMAL)
        self.progress_var.set(f"Complete! Report saved to: {output_file}")

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, summary_text)
        self.results_text.insert(tk.END, f"\n\nReport saved to:\n{output_file}")

        if messagebox.askyesno("Success", "Order plan generated successfully!\n\nWould you like to open the report?"):
            subprocess.Popen(f'explorer /select,"{output_file}"')

    def _show_results_error(self, error_msg):
        """Show error results."""
        self.progress_bar.stop()
        self.generate_btn.config(state=tk.NORMAL)
        self.progress_var.set("Error generating order plan")

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR:\n{error_msg}")

    def _upload_order_form(self):
        """Upload and parse a manufacturer order form with preview dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Order Form",
            filetypes=[
                ("Excel files", "*.xlsx;*.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.progress_var.set(f"Parsing order form: {Path(file_path).name}...")
        self.progress_bar.start()

        def parse():
            try:
                from order_engine.order_form_parser import OrderFormParser

                parser = OrderFormParser()
                result = parser.parse_file(file_path)

                if result.success:
                    self.root.after(0, lambda: self._show_upload_preview(result, parser))
                else:
                    error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
                    self.root.after(0, lambda: self._show_results_error(f"Failed to parse order form:\n{error_msg}"))

            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._show_results_error(f"Error parsing order form: {e}\n\n{traceback.format_exc()}"))

        thread = threading.Thread(target=parse, daemon=True)
        thread.start()

    def _show_upload_preview(self, result, parser):
        """Show upload preview dialog."""
        self.progress_bar.stop()
        self.progress_var.set("Order form parsed - showing preview")

        # Create preview dialog
        preview = tk.Toplevel(self.root)
        preview.title("Order Form Preview")
        preview.geometry("800x600")
        preview.transient(self.root)
        preview.grab_set()

        main_frame = ttk.Frame(preview, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Summary
        summary_frame = ttk.LabelFrame(main_frame, text="Summary", padding="10")
        summary_frame.pack(fill=tk.X, pady=5)

        ttk.Label(summary_frame, text=f"File: {result.source_file}").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Sheet: {result.sheet_name}").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Total Lines: {len(result.orders)}").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Total Units: {result.total_units:,}").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Total Cost: ${result.total_cost:,.0f}").pack(anchor=tk.W)

        # Column mapping
        mapping_frame = ttk.LabelFrame(main_frame, text="Detected Column Mapping", padding="10")
        mapping_frame.pack(fill=tk.X, pady=5)

        mapping = result.column_mapping
        ttk.Label(mapping_frame, text=f"Model Column: {mapping.model_col or 'Not Found'}").pack(anchor=tk.W)
        ttk.Label(mapping_frame, text=f"Qty Column: {mapping.qty_col or 'Not Found'}").pack(anchor=tk.W)
        ttk.Label(mapping_frame, text=f"Cost Column: {mapping.cost_col or 'Not Found'}").pack(anchor=tk.W)
        ttk.Label(mapping_frame, text=f"Make Column: {mapping.make_col or 'Not Found'}").pack(anchor=tk.W)
        ttk.Label(mapping_frame, text=f"Confidence: {mapping.confidence:.0%}").pack(anchor=tk.W)

        # Orders preview
        preview_frame = ttk.LabelFrame(main_frame, text="Orders Preview (first 20)", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        tree = ttk.Treeview(
            preview_frame,
            columns=("model", "make", "qty", "cost", "total"),
            show="headings",
            height=10
        )
        tree.heading("model", text="Model")
        tree.heading("make", text="Make")
        tree.heading("qty", text="Qty")
        tree.heading("cost", text="Unit Cost")
        tree.heading("total", text="Total")

        for order in result.orders[:20]:
            tree.insert("", tk.END, values=(
                order.model,
                order.make or "",
                order.qty,
                f"${order.unit_cost:,.0f}" if order.unit_cost else "",
                f"${order.qty * order.unit_cost:,.0f}" if order.unit_cost else ""
            ))

        tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        tree_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Warnings
        if result.warnings:
            warn_frame = ttk.LabelFrame(main_frame, text=f"Warnings ({len(result.warnings)})", padding="5")
            warn_frame.pack(fill=tk.X, pady=5)

            warn_text = tk.Text(warn_frame, height=3, wrap=tk.WORD)
            warn_text.pack(fill=tk.X)
            warn_text.insert(tk.END, "\n".join(result.warnings[:10]))
            if len(result.warnings) > 10:
                warn_text.insert(tk.END, f"\n... and {len(result.warnings) - 10} more")
            warn_text.config(state=tk.DISABLED)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        def accept():
            preview.destroy()
            self._apply_parsed_orders(result)

        ttk.Button(btn_frame, text="Accept", command=accept).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=preview.destroy).pack(side=tk.LEFT, padx=5)

    def _apply_parsed_orders(self, result):
        """Apply parsed orders to the results display."""
        summary = f"Order Form Parsed Successfully!\n\n"
        summary += f"File: {result.source_file}\n"
        summary += f"Total Lines: {len(result.orders)}\n"
        summary += f"Total Units: {result.total_units:,}\n"
        summary += f"Total Cost: ${result.total_cost:,.0f}\n"

        if result.warnings:
            summary += f"\nWarnings ({len(result.warnings)}):\n"
            for w in result.warnings[:5]:
                summary += f"  - {w}\n"
            if len(result.warnings) > 5:
                summary += f"  ... and {len(result.warnings) - 5} more\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, summary)
        self.progress_var.set("Order form loaded successfully")

    # =========================================================================
    # CAPACITY ANALYSIS METHODS
    # =========================================================================

    def _run_capacity_analysis(self):
        """Run capacity analysis."""
        self.progress_var.set("Running capacity analysis...")
        self.progress_bar.start()

        # Get selected vehicle types
        selected_veh_types = [vt for vt, var in self.cap_veh_type_vars.items() if var.get()]

        # Get selected zone
        zone = self.cap_zone_var.get()
        if zone == "All Zones":
            zone = None

        def analyze():
            try:
                from order_engine.capacity_planner import CapacityPlanner

                planner = CapacityPlanner()
                report = planner.generate_capacity_report()

                self.root.after(0, lambda: self._display_capacity_results(report, selected_veh_types, zone))

            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._show_error(f"Capacity analysis error: {e}\n{traceback.format_exc()}"))

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

    def _display_capacity_results(self, report, veh_types_filter=None, zone_filter=None):
        """Display capacity analysis results."""
        self.progress_bar.stop()
        self.progress_var.set("Capacity analysis complete")

        # Update summary labels
        summary = report.get("summary", {})
        self.cap_lot_capacity_var.set(f"{summary.get('total_lot_capacity', 0):,}")
        self.cap_current_inv_var.set(f"{summary.get('total_current_inventory', 0):,}")
        self.cap_pipeline_var.set(f"{summary.get('total_pipeline', 0):,}")
        self.cap_gap_var.set(f"{summary.get('total_gap', 0):,}")

        # Clear and populate zone tree
        for item in self.cap_zone_tree.get_children():
            self.cap_zone_tree.delete(item)

        by_zone = report.get("by_zone", {})
        for zone, data in sorted(by_zone.items(), key=lambda x: x[1].get("gap", 0), reverse=True):
            if zone_filter and zone != zone_filter:
                continue

            available = data.get("current", 0) + data.get("pipeline", 0)
            capacity = data.get("capacity", 0)
            fill_pct = available / capacity * 100 if capacity > 0 else 0

            self.cap_zone_tree.insert("", tk.END, values=(
                zone,
                f"{capacity:,}",
                f"{data.get('current', 0):,}",
                f"{data.get('pipeline', 0):,}",
                f"{data.get('gap', 0):,}",
                f"{fill_pct:.1f}%"
            ))

        # Clear and populate type tree
        for item in self.cap_type_tree.get_children():
            self.cap_type_tree.delete(item)

        by_type = report.get("by_type", {})
        for veh_type, data in sorted(by_type.items()):
            if veh_types_filter and veh_type not in veh_types_filter:
                continue

            self.cap_type_tree.insert("", tk.END, values=(
                veh_type,
                f"{data.get('capacity', 0):,}",
                f"{data.get('current', 0):,}",
                f"{data.get('pipeline', 0):,}",
                f"{data.get('gap', 0):,}",
                f"{data.get('fill_pct', 0):.1f}%"
            ))

        # Update location gaps text
        self.cap_location_text.delete(1.0, tk.END)
        location_gaps = report.get("location_gaps", [])
        for gap in location_gaps[:15]:
            brands = ", ".join(gap.get("licensed_brands", [])[:5])
            if len(gap.get("licensed_brands", [])) > 5:
                brands += "..."

            self.cap_location_text.insert(tk.END,
                f"{gap['location']:8} | {gap['zone']:15} | {gap['veh_type']:4} | Gap: {gap['gap']:5,} | Brands: {brands}\n"
            )

        # Store report for Excel export
        self._capacity_report = report

    def _generate_capacity_excel(self):
        """Generate capacity Excel report."""
        if not hasattr(self, '_capacity_report') or not self._capacity_report:
            messagebox.showwarning("Warning", "Please run capacity analysis first")
            return

        self.progress_var.set("Generating Excel report...")
        self.progress_bar.start()

        def generate():
            try:
                report_gen = ReportGenerator()
                output_file = report_gen.generate_capacity_report(self._capacity_report)
                self.root.after(0, lambda: self._capacity_excel_complete(output_file))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"Error generating report: {e}"))

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def _capacity_excel_complete(self, output_file):
        """Called when capacity Excel is complete."""
        self.progress_bar.stop()
        self.progress_var.set(f"Report saved: {output_file}")

        if messagebox.askyesno("Success", f"Report saved to:\n{output_file}\n\nOpen file?"):
            subprocess.Popen(f'explorer /select,"{output_file}"')

    # =========================================================================
    # RECONCILIATION METHODS
    # =========================================================================

    def _edit_adjustments(self):
        """Open adjustments file in notepad."""
        try:
            from order_engine.reconciliation import ReconciliationEngine
            engine = ReconciliationEngine()
            adj_file = engine.adjustments_file

            # Create file if it doesn't exist
            if not adj_file.exists():
                engine._create_default_adjustments_file()

            subprocess.Popen(['notepad.exe', str(adj_file)])
        except Exception as e:
            self._show_error(f"Error opening adjustments file: {e}")

    def _reload_adjustments(self):
        """Reload adjustments from file."""
        try:
            from order_engine.reconciliation import ReconciliationEngine
            engine = ReconciliationEngine()

            # Clear tree
            for item in self.recon_adj_tree.get_children():
                self.recon_adj_tree.delete(item)

            # Populate tree
            for adj in engine.adjustments:
                self.recon_adj_tree.insert("", tk.END, values=(
                    adj.adj_type,
                    adj.target,
                    f"{adj.adjustment_pct:+.0%}",
                    adj.note
                ))

            if not engine.adjustments:
                self.recon_adj_tree.insert("", tk.END, values=(
                    "(none)", "No adjustments configured", "", "Edit order_adjustments.yaml to add"
                ))

        except Exception as e:
            self._show_error(f"Error loading adjustments: {e}")

    def _run_reconciliation(self):
        """Run reconciliation analysis."""
        self.progress_var.set("Running reconciliation analysis...")
        self.progress_bar.start()

        # Get parameters
        mode = self.recon_mode_var.get()
        planning_months = int(self.recon_months_var.get()) if mode == "projected" else 0
        target_fill = self.recon_fill_var.get() / 100.0

        mfg = self.recon_mfg_var.get()
        if mfg == "All Manufacturers":
            mfg = None

        def analyze():
            try:
                from order_engine.reconciliation import ReconciliationEngine

                engine = ReconciliationEngine()
                result = engine.generate_reconciliation(
                    planning_months=planning_months,
                    target_fill_pct=target_fill,
                    manufacturer=mfg
                )

                self.root.after(0, lambda: self._display_reconciliation_results(result))

            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._show_error(f"Reconciliation error: {e}\n{traceback.format_exc()}"))

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

    def _display_reconciliation_results(self, result):
        """Display reconciliation results."""
        self.progress_bar.stop()
        self.progress_var.set("Reconciliation complete")

        # Update summary labels
        capacity = result.get("capacity_summary", {})
        self.recon_total_gap_var.set(f"{capacity.get('total_gap', 0):,} units")

        final = result.get("final_allocation", [])
        total_allocated = sum(f.get("recommended_units", 0) for f in final)
        self.recon_total_alloc_var.set(f"{total_allocated:,} units")

        # Clear and populate allocation tree
        for item in self.recon_alloc_tree.get_children():
            self.recon_alloc_tree.delete(item)

        for alloc in final:
            delta = alloc.get("adjustment_delta", 0)
            delta_str = f"+{delta:,}" if delta > 0 else f"{delta:,}" if delta < 0 else "-"

            self.recon_alloc_tree.insert("", tk.END, values=(
                alloc.get("manufacturer", ""),
                f"{alloc.get('base_units', 0):,}",
                f"{alloc.get('recommended_units', 0):,}",
                f"{alloc.get('share_of_total', 0):.1%}",
                delta_str
            ))

        # Reload adjustments to show what was applied
        self._reload_adjustments()

        # Store result for Excel export
        self._reconciliation_result = result

    def _generate_reconciliation_excel(self):
        """Generate reconciliation Excel report."""
        if not hasattr(self, '_reconciliation_result') or not self._reconciliation_result:
            messagebox.showwarning("Warning", "Please run reconciliation first")
            return

        self.progress_var.set("Generating Excel report...")
        self.progress_bar.start()

        def generate():
            try:
                from order_engine.reconciliation import ReconciliationReportGenerator
                report_gen = ReconciliationReportGenerator()
                output_file = report_gen.generate_report(self._reconciliation_result)
                self.root.after(0, lambda: self._reconciliation_excel_complete(output_file))
            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"Error generating report: {e}"))

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def _reconciliation_excel_complete(self, output_file):
        """Called when reconciliation Excel is complete."""
        self.progress_bar.stop()
        self.progress_var.set(f"Report saved: {output_file}")

        if messagebox.askyesno("Success", f"Report saved to:\n{output_file}\n\nOpen file?"):
            subprocess.Popen(f'explorer /select,"{output_file}"')

    # =========================================================================
    # YOY DASHBOARD METHODS
    # =========================================================================

    def _refresh_yoy_data(self):
        """Refresh YoY data."""
        self.progress_var.set("Loading YoY data...")
        self.progress_bar.start()

        # Get filters
        mfg = self.yoy_mfg_var.get()
        if mfg == "All Manufacturers":
            mfg = None

        veh_type = self.yoy_veh_var.get()
        if veh_type == "All Types":
            veh_type = None

        adjust_for_locations = self.yoy_adj_var.get()

        def analyze():
            try:
                from order_engine.yoy_analyzer import YoYAnalyzer

                analyzer = YoYAnalyzer(data_loader=self.data_loader)

                # YTD comparison
                yoy = analyzer.compare_yoy(
                    manufacturer=mfg,
                    veh_type=veh_type,
                    adjust_for_locations=adjust_for_locations,
                    use_ytd=True
                )

                # Quarterly comparison
                quarterly = analyzer.compare_yoy_by_quarter(
                    manufacturer=mfg,
                    adjust_for_locations=adjust_for_locations
                )

                # Monthly comparison
                monthly = analyzer.compare_yoy_by_month(
                    manufacturer=mfg,
                    adjust_for_locations=adjust_for_locations
                )

                self.root.after(0, lambda: self._display_yoy_results(yoy, quarterly, monthly))

            except Exception as e:
                import traceback
                self.root.after(0, lambda: self._show_error(f"YoY analysis error: {e}\n{traceback.format_exc()}"))

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

    def _display_yoy_results(self, yoy, quarterly, monthly):
        """Display YoY results."""
        self.progress_bar.stop()
        self.progress_var.set("YoY data loaded")

        if "error" in yoy:
            self._show_error(f"YoY Error: {yoy['error']}")
            return

        # Update YTD summary
        self.yoy_ly_units_var.set(f"{yoy.get('last_year_units', 0):,}")
        self.yoy_cy_units_var.set(f"{yoy.get('current_year_units', 0):,}")
        self.yoy_change_var.set(yoy.get('units_change_pct_formatted', '--'))
        self.yoy_adj_change_var.set(yoy.get('adjusted_change_pct_formatted', '--'))

        self.yoy_ly_locs_var.set(str(yoy.get('last_year_locations', '--')))
        self.yoy_cy_locs_var.set(str(yoy.get('current_year_locations', '--')))

        loc_growth = yoy.get('location_growth', 0)
        self.yoy_loc_change_var.set(f"+{loc_growth}" if loc_growth > 0 else str(loc_growth))

        self.yoy_interp_var.set(yoy.get('interpretation', ''))

        # Update quarterly tree
        for item in self.yoy_quarterly_tree.get_children():
            self.yoy_quarterly_tree.delete(item)

        if not quarterly.empty:
            current_year = datetime.now().year
            last_year = current_year - 1

            for _, row in quarterly.iterrows():
                self.yoy_quarterly_tree.insert("", tk.END, values=(
                    row.get("Quarter", ""),
                    row.get(str(last_year), 0),
                    row.get(str(current_year), 0),
                    row.get("Change %", "--"),
                    row.get("Adj. Change %", "--")
                ))

        # Update monthly tree
        for item in self.yoy_monthly_tree.get_children():
            self.yoy_monthly_tree.delete(item)

        if not monthly.empty:
            current_year = datetime.now().year
            last_year = current_year - 1

            for _, row in monthly.iterrows():
                self.yoy_monthly_tree.insert("", tk.END, values=(
                    row.get("Month", ""),
                    row.get(str(last_year), 0),
                    row.get(str(current_year), 0),
                    row.get("Change %", "--")
                ))

        # Store for export
        self._yoy_data = {"yoy": yoy, "quarterly": quarterly, "monthly": monthly}

    def _export_yoy_excel(self):
        """Export YoY data to Excel."""
        if not hasattr(self, '_yoy_data') or not self._yoy_data:
            messagebox.showwarning("Warning", "Please refresh YoY data first")
            return

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save YoY Report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"YoY_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )

        if not file_path:
            return

        self.progress_var.set("Exporting to Excel...")
        self.progress_bar.start()

        def export():
            try:
                import pandas as pd

                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # Summary sheet
                    yoy = self._yoy_data["yoy"]
                    summary_data = [
                        ["YoY Comparison Summary", ""],
                        ["", ""],
                        ["Last Year", yoy.get("last_year", "")],
                        ["Current Year", yoy.get("current_year", "")],
                        ["", ""],
                        ["Last Year Units", yoy.get("last_year_units", 0)],
                        ["Current Year Units", yoy.get("current_year_units", 0)],
                        ["Change %", yoy.get("units_change_pct_formatted", "")],
                        ["", ""],
                        ["Last Year Locations", yoy.get("last_year_locations", "")],
                        ["Current Year Locations", yoy.get("current_year_locations", "")],
                        ["Location Growth", yoy.get("location_growth", "")],
                        ["", ""],
                        ["Adjusted Change %", yoy.get("adjusted_change_pct_formatted", "")],
                        ["Interpretation", yoy.get("interpretation", "")]
                    ]
                    pd.DataFrame(summary_data, columns=["Metric", "Value"]).to_excel(
                        writer, sheet_name="Summary", index=False
                    )

                    # Quarterly sheet
                    if not self._yoy_data["quarterly"].empty:
                        self._yoy_data["quarterly"].to_excel(
                            writer, sheet_name="Quarterly", index=False
                        )

                    # Monthly sheet
                    if not self._yoy_data["monthly"].empty:
                        self._yoy_data["monthly"].to_excel(
                            writer, sheet_name="Monthly", index=False
                        )

                self.root.after(0, lambda: self._yoy_export_complete(file_path))

            except Exception as e:
                self.root.after(0, lambda: self._show_error(f"Error exporting: {e}"))

        thread = threading.Thread(target=export, daemon=True)
        thread.start()

    def _yoy_export_complete(self, file_path):
        """Called when YoY export is complete."""
        self.progress_bar.stop()
        self.progress_var.set(f"Exported: {file_path}")

        if messagebox.askyesno("Success", f"Report saved to:\n{file_path}\n\nOpen file?"):
            subprocess.Popen(f'explorer /select,"{file_path}"')


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    app = OrderRecommendationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
