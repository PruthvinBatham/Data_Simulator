"""
Path Simulation App with Comprehensive Validation
A clean interface for running return path simulations with detailed validation displays
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import uuid
import zipfile
import io

from simulation_engine import SimulationConfig, PathSimulator

# Page configuration
st.set_page_config(
    page_title="Path Simulation",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .validation-pass {
        color: #10b981;
        font-weight: bold;
    }
    .validation-fail {
        color: #ef4444;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #1f2937;
    }
    .alert-success {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
        color: #1f2937;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None

# Presets
PRESETS = {
    "Current (2-year)": {
        "time_horizon": 2,
        "target_correlation": 0.50,
        "n_paths_predictable": 15,
        "n_paths_iid": 15,
        "mu": 0.0607,
        "sigma_eps": 0.192,
        "phi": 0.92,
        "sigma_delta": 0.152,
        "rho": -0.72,
    },
    "Alternative (5-year)": {
        "time_horizon": 5,
        "target_correlation": 0.57,
        "n_paths_predictable": 15,
        "n_paths_iid": 15,
        "mu": 0.0607,
        "sigma_eps": 0.192,
        "phi": 0.92,
        "sigma_delta": 0.152,
        "rho": -0.72,
    }
}

# App header
st.markdown('<div class="main-header">📈 Path Simulation</div>', unsafe_allow_html=True)

# Sidebar - Parameter inputs
with st.sidebar:
    st.header("Simulation Parameters")
    
    # Preset selector
    preset_choice = st.selectbox(
        "Preset Configuration",
        ["Custom"] + list(PRESETS.keys()),
        help="Select a preset or choose Custom to set your own parameters"
    )
    
    # Load preset if selected
    if preset_choice != "Custom":
        preset_params = PRESETS[preset_choice]
    else:
        preset_params = PRESETS["Current (2-year)"]  # Default values
    
    st.divider()
    
    # Path Configuration
    st.subheader("Path Configuration")
    
    time_horizon = st.number_input(
        "Time Horizon (years)",
        min_value=1,
        max_value=10,
        value=preset_params["time_horizon"],
        help="Returns are averaged over H years"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_paths_predictable = st.number_input(
            "Predictable Paths",
            min_value=0,
            max_value=50,
            value=preset_params["n_paths_predictable"],
            help="Set to 0 to skip predictable paths"
        )
    with col2:
        n_paths_iid = st.number_input(
            "I.I.D. Paths",
            min_value=0,
            max_value=50,
            value=preset_params["n_paths_iid"],
            help="Set to 0 to skip I.I.D. paths"
        )
    
    st.divider()
    
    # Correlation Settings
    st.subheader("Correlation Settings")
    
    target_correlation = st.slider(
        "Target Correlation",
        min_value=0.0,
        max_value=1.0,
        value=preset_params["target_correlation"],
        step=0.01,
        help=f"For {time_horizon}-year returns. R² ≈ {preset_params['target_correlation']**2:.3f}"
    )
    
    correlation_window = st.slider(
        "Correlation Window",
        min_value=0.01,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Accept correlations within ± this amount"
    )
    
    st.divider()
    
    # Advanced Parameters (collapsible)
    with st.expander("Advanced Parameters"):
        st.caption("Return Parameters")
        mu = st.number_input(
            "Mean Return (μ)",
            value=preset_params["mu"],
            format="%.4f",
            help="Mean annual log return"
        )
        
        sigma_eps = st.number_input(
            "Return Volatility (σ_ε)",
            value=preset_params["sigma_eps"],
            format="%.4f",
            help="Annual return volatility"
        )
        
        st.caption("Signal Parameters (AR(1))")
        phi = st.slider(
            "Persistence (φ)",
            min_value=0.0,
            max_value=0.99,
            value=preset_params["phi"],
            step=0.01
        )
        
        sigma_delta = st.number_input(
            "Signal Volatility (σ_δ)",
            value=preset_params["sigma_delta"],
            format="%.4f"
        )
        
        rho = st.slider(
            "Shock Correlation (ρ)",
            min_value=-1.0,
            max_value=1.0,
            value=preset_params["rho"],
            step=0.01,
            help="Correlation between signal and return shocks"
        )
        
        st.caption("Simulation Settings")
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=50000,
            value=10000,
            step=1000,
            help="Generate this many paths before selecting the best"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
    
    st.divider()
    
    # Validation settings
    st.subheader("Validation Settings")
    
    # SPX Data Path
    spx_path = st.text_input(
        "S&P 500 Data Path (optional)",
        value="",
        help="Path to SPX_5Y_Returns.xlsx for validation"
    )
    
    # Check if SPX data exists
    default_spx = Path(__file__).parent.parent.parent / "Data" / "data used in original code" / "SPX_5Y_Returns.xlsx"
    spx_available = (spx_path and Path(spx_path).exists()) or default_spx.exists()
    
    # Validation toggle
    enable_validation = st.checkbox(
        "Enable SPX Validation",
        value=spx_available,  # Default to True if SPX data exists
        help="Enable validation against S&P 500 historical data",
        disabled=not spx_available,  # Disable checkbox if no SPX data found
        key="enable_validation_checkbox"
    )
    
    # Store in session state for debugging
    st.session_state.enable_validation = enable_validation
    
    # Show path info
    if spx_path and Path(spx_path).exists():
        st.info(f"✓ Using custom SPX data: {spx_path}")
    elif default_spx.exists():
        st.info(f"✓ Using default SPX data: {default_spx}")
    else:
        st.warning("⚠️ No SPX data found. Validation disabled. Provide path above to enable.")
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Main content area
if run_button:
    with st.spinner("Running simulation..."):
        # Determine SPX path and validation setting
        spx_data_path = None
        validate_spx = False
        
        if enable_validation:
            if spx_path and Path(spx_path).exists():
                spx_data_path = spx_path
                validate_spx = True
            else:
                # Try default path
                default_path = Path(__file__).parent.parent.parent / "Data" / "data used in original code" / "SPX_5Y_Returns.xlsx"
                if default_path.exists():
                    spx_data_path = str(default_path)
                    validate_spx = True
        
        # Debug: Show what values we're using
        st.write("**Debug - Before Simulation:**")
        st.write(f"- enable_validation checkbox: {enable_validation}")
        st.write(f"- validate_spx computed: {validate_spx}")
        st.write(f"- spx_data_path: {spx_data_path}")
        st.write("---")
        
        # Create configuration
        config = SimulationConfig(
            n_paths_predictable=n_paths_predictable,
            n_paths_iid=n_paths_iid,
            time_horizon=time_horizon,
            target_correlation=target_correlation,
            correlation_window=correlation_window,
            mu=mu,
            sigma_eps=sigma_eps,
            phi=phi,
            sigma_delta=sigma_delta,
            rho=rho,
            n_simulations=n_simulations,
            random_seed=random_seed,
            validate_with_spx=validate_spx,
            spx_data_path=spx_data_path,
            save_example_plots=False
        )
        
        # Run simulation
        simulator = PathSimulator(config)
        results = simulator.run()
        
        # Generate run ID
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Store in session state
        st.session_state.results = results
        st.session_state.run_id = run_id
        st.session_state.config = config
        
        st.success("✓ Simulation completed!")
        
        st.rerun()

# Display results
if st.session_state.results is not None:
    results = st.session_state.results
    config = st.session_state.config
    validation = results.get('validation', {})
    
    # Debug: Show validation configuration
    with st.expander("🔧 Debug: Validation Configuration", expanded=False):
        st.write(f"**Validation Enabled:** {config.validate_with_spx}")
        st.write(f"**SPX Data Path:** {config.spx_data_path}")
        st.write(f"**Validation Results Present:** {bool(validation)}")
        if validation:
            st.write(f"**Validation Keys:** {list(validation.keys())}")
            if 'spx_stats' in validation:
                st.write("✓ SPX statistics loaded")
            if 'predictable' in validation:
                st.write(f"✓ Predictable validation: {len(validation['predictable'].get('regressions', []))} paths")
            if 'iid' in validation:
                st.write(f"✓ IID validation: {len(validation['iid'].get('regressions', []))} paths")
    
    # Alert/Warning System (PRIORITY 6)
    if validation:
        pred_compliance = validation.get('predictable', {}).get('pass_rates', {}).get('full_compliance', 1.0)
        iid_compliance = validation.get('iid', {}).get('pass_rates', {}).get('full_compliance', 1.0)
        
        # Calculate R² warnings
        target_r2 = config.target_correlation ** 2
        warnings = []
        
        if 'predictable' in validation:
            pred_regs = validation['predictable'].get('regressions', [])
            if pred_regs:
                mean_r2_signal = np.mean([r['r2_signal'] for r in pred_regs if not np.isnan(r['r2_signal'])])
                mean_r2_lag = np.mean([r['r2_lag'] for r in pred_regs if not np.isnan(r['r2_lag'])])
                
                if abs(mean_r2_signal - target_r2) > 0.10:
                    warnings.append(f"Mean R²(signal) ({mean_r2_signal:.3f}) differs from target ({target_r2:.3f}) by more than 0.10")
                if mean_r2_lag > 0.10:
                    warnings.append(f"Mean R²(lag) ({mean_r2_lag:.3f}) > 0.10 indicates serial correlation problem")
        
        if pred_compliance < 0.7 or iid_compliance < 0.7 or warnings:
            st.markdown(f"""
            <div class="alert-warning">
                <strong>⚠️ Validation Warning</strong><br>
                Some paths have low compliance rates (Predictable: {pred_compliance*100:.0f}%, IID: {iid_compliance*100:.0f}%).<br>
                {('<br>'.join(warnings) + '<br>') if warnings else ''}
                Consider: (1) Increasing N_SIMULATIONS to {config.n_simulations * 2:,}, or (2) Loosening correlation window to {correlation_window + 0.02:.2f}
            </div>
            """, unsafe_allow_html=True)
        elif pred_compliance >= 0.9 and iid_compliance >= 0.9:
            st.markdown(f"""
            <div class="alert-success">
                <strong>✓ Excellent Validation</strong><br>
                High compliance rates achieved (Predictable: {pred_compliance*100:.0f}%, IID: {iid_compliance*100:.0f}%).
            </div>
            """, unsafe_allow_html=True)
    
    # Configuration Summary (PRIORITY 7)
    with st.expander("📋 Configuration Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Path Settings**")
            st.write(f"Time Horizon: {config.time_horizon} years")
            st.write(f"Predictable Paths: {config.n_paths_predictable}")
            st.write(f"I.I.D. Paths: {config.n_paths_iid}")
        
        with col2:
            st.write("**Correlation**")
            st.write(f"Target: {config.target_correlation:.3f}")
            st.write(f"Window: ±{config.correlation_window:.3f}")
            st.write(f"Target R²: {config.target_correlation**2:.3f}")
        
        with col3:
            st.write("**Simulation**")
            st.write(f"N Simulations: {config.n_simulations:,}")
            st.write(f"Random Seed: {config.random_seed}")
            st.write(f"SPX Validation: {'Yes' if config.validate_with_spx else 'No'}")
    
    # Summary metrics
    st.header("Simulation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Time Horizon",
            f"{config.time_horizon} years"
        )
    
    with col2:
        st.metric(
            "Total Paths",
            f"{config.n_paths_predictable + config.n_paths_iid}"
        )
    
    with col3:
        if results['correlations_pred'] is not None:
            mean_corr_pred = np.mean(results['correlations_pred'])
            st.metric(
                "Mean Correlation (Pred)",
                f"{mean_corr_pred:.3f}"
            )
        else:
            st.metric("Mean Correlation (Pred)", "N/A")
    
    with col4:
        if results['correlations_iid'] is not None:
            mean_corr_iid = np.mean(results['correlations_iid'])
            st.metric(
                "Mean Correlation (IID)",
                f"{mean_corr_iid:.3f}"
            )
        else:
            st.metric("Mean Correlation (IID)", "N/A")
    
    st.divider()
    
    # SPX Baseline Statistics Card (PRIORITY 2)
    if validation and 'spx_stats' in validation:
        st.subheader("📊 S&P 500 Reference Data")
        spx_stats = validation['spx_stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**{config.time_horizon}-Year Returns**")
            spx_h = spx_stats['h_year']
            st.write(f"Mean: {spx_h['mean']:.4f} ({spx_h['mean']*100:.2f}%)")
            st.write(f"Std Dev: {spx_h['std']:.4f}")
            st.write(f"Min: {spx_h['min']:.4f} | Max: {spx_h['max']:.4f}")
            st.write(f"Observations: {spx_h['n_obs']}")
        
        with col2:
            st.write("**Annual Returns**")
            spx_1 = spx_stats['annual']
            st.write(f"Mean: {spx_1['mean']:.4f} ({spx_1['mean']*100:.2f}%)")
            st.write(f"Std Dev: {spx_1['std']:.4f}")
            st.write(f"Observations: {spx_1['n_obs']}")
        
        st.divider()
    
    # Validation Results Panel (PRIORITY 1)

    # R² Summary Card (HIGH PRIORITY)
    if validation and ('predictable' in validation or 'iid' in validation):
        st.subheader("📈 R² Analysis")
        
        target_r2 = config.target_correlation ** 2
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Target R²:** {target_r2:.3f} (H={config.time_horizon} years)")
            st.write(f"**Theory:** R² = ρ² = {config.target_correlation:.3f}² = {target_r2:.3f}")
        
        with col2:
            st.write("**Interpretation:**")
            st.write("• R²(signal): Variance explained by signal")
            st.write("• R²(lag): Variance explained by past returns")
        
        st.write("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'predictable' in validation:
                st.write("**Predictable Paths:**")
                pred_regs = validation['predictable'].get('regressions', [])
                if pred_regs:
                    r2_signal_vals = [r['r2_signal'] for r in pred_regs if not np.isnan(r['r2_signal'])]
                    r2_lag_vals = [r['r2_lag'] for r in pred_regs if not np.isnan(r['r2_lag'])]
                    
                    if r2_signal_vals and r2_lag_vals:
                        mean_r2_signal = np.mean(r2_signal_vals)
                        std_r2_signal = np.std(r2_signal_vals)
                        mean_r2_lag = np.mean(r2_lag_vals)
                        std_r2_lag = np.std(r2_lag_vals)
                        
                        # Color coding
                        signal_diff = abs(mean_r2_signal - target_r2)
                        signal_status = "✓" if signal_diff <= 0.05 else ("~" if signal_diff <= 0.10 else "✗")
                        lag_status = "✓" if mean_r2_lag < 0.05 else ("~" if mean_r2_lag < 0.10 else "✗")
                        
                        signal_color = "validation-pass" if signal_diff <= 0.05 else ("" if signal_diff <= 0.10 else "validation-fail")
                        lag_color = "validation-pass" if mean_r2_lag < 0.05 else ("" if mean_r2_lag < 0.10 else "validation-fail")
                        
                        st.markdown(f'<span class="{signal_color}">Signal R²: {mean_r2_signal:.3f} ± {std_r2_signal:.3f} {signal_status}</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="{lag_color}">Lagged R²: {mean_r2_lag:.3f} ± {std_r2_lag:.3f} {lag_status}</span>', unsafe_allow_html=True)
                    else:
                        st.write("Insufficient data for analysis")
                else:
                    st.write("No regression data available")
        
        with col2:
            if 'iid' in validation:
                st.write("**I.I.D. Paths:**")
                iid_regs = validation['iid'].get('regressions', [])
                if iid_regs:
                    r2_signal_vals = [r['r2_signal'] for r in iid_regs if not np.isnan(r['r2_signal'])]
                    r2_lag_vals = [r['r2_lag'] for r in iid_regs if not np.isnan(r['r2_lag'])]
                    
                    if r2_signal_vals and r2_lag_vals:
                        mean_r2_signal = np.mean(r2_signal_vals)
                        std_r2_signal = np.std(r2_signal_vals)
                        mean_r2_lag = np.mean(r2_lag_vals)
                        std_r2_lag = np.std(r2_lag_vals)
                        
                        # Both should be ~0 for IID
                        signal_status = "✓" if mean_r2_signal < 0.05 else ("~" if mean_r2_signal < 0.10 else "✗")
                        lag_status = "✓" if mean_r2_lag < 0.05 else ("~" if mean_r2_lag < 0.10 else "✗")
                        
                        signal_color = "validation-pass" if mean_r2_signal < 0.05 else ("" if mean_r2_signal < 0.10 else "validation-fail")
                        lag_color = "validation-pass" if mean_r2_lag < 0.05 else ("" if mean_r2_lag < 0.10 else "validation-fail")
                        
                        st.markdown(f'<span class="{signal_color}">Signal R²: {mean_r2_signal:.3f} ± {std_r2_signal:.3f} {signal_status}</span>', unsafe_allow_html=True)
                        st.markdown(f'<span class="{lag_color}">Lagged R²: {mean_r2_lag:.3f} ± {std_r2_lag:.3f} {lag_status}</span>', unsafe_allow_html=True)
                    else:
                        st.write("Insufficient data for analysis")
                else:
                    st.write("No regression data available")
        
        st.divider()
    
    if validation and ('predictable' in validation or 'iid' in validation):
        st.header("🔍 Validation Results")
        
        # Determine which columns to show
        has_pred = 'predictable' in validation
        has_iid = 'iid' in validation
        
        if has_pred and has_iid:
            col1, col2 = st.columns(2)
        else:
            col1 = st.container()
            col2 = None
        
        # Predictable paths column
        if has_pred:
            val_pred = validation['predictable']
            
            with col1:
                st.subheader("Predictable Paths")
                
                # Full compliance
                n_compliant = sum(val_pred.get('full_compliant', []))
                n_total = len(val_pred.get('full_compliant', [1]))
                compliance_rate = n_compliant / n_total if n_total > 0 else 0
                
                st.metric(
                    "Full Compliance",
                    f"{n_compliant}/{n_total}",
                    f"{compliance_rate*100:.0f}%"
                )
                
                # Individual checks
                pass_rates = val_pred.get('pass_rates', {})
                
                checks = [
                    ("KS Test (h-year)", pass_rates.get('ks_h_year', 0), 
                     "Tests if h-year return distribution matches S&P 500 historical distribution"),
                    ("KS Test (annual)", pass_rates.get('ks_annual', 0),
                     "Tests if annual return distribution matches S&P 500 historical distribution"),
                    ("Return Bounds", pass_rates.get('return_bounds', 0),
                     "Checks if min/max returns fall within S&P 500 historical range"),
                    ("Volatility Bounds", pass_rates.get('volatility_bounds', 0),
                     "Checks if return volatility is within ±5% of S&P 500 volatility"),
                ]
                
                for check_name, rate, help_text in checks:
                    status = "✓" if rate >= 0.7 else "✗"
                    color = "validation-pass" if rate >= 0.7 else "validation-fail"
                    st.markdown(f'<span class="{color}" title="{help_text}">{status} {check_name}: {rate*100:.0f}% pass</span>', unsafe_allow_html=True)
                    st.caption(help_text)
                
                # Autocorrelation
                autocorrs = val_pred.get('autocorrelations', [])
                mean_autocorr = np.nanmean(autocorrs) if autocorrs else 0
                st.write(f"Mean Autocorrelation: {mean_autocorr:.4f} (should be ~0)")
        
        # I.I.D. paths column
        if has_iid:
            val_iid = validation['iid']
            
            # Use col2 if it exists, otherwise use a new container
            with (col2 if col2 is not None else st.container()):
                st.subheader("I.I.D. Paths")
                
                # Full compliance
                n_compliant = sum(val_iid.get('full_compliant', []))
                n_total = len(val_iid.get('full_compliant', [1]))
                compliance_rate = n_compliant / n_total if n_total > 0 else 0
                
                st.metric(
                    "Full Compliance",
                    f"{n_compliant}/{n_total}",
                    f"{compliance_rate*100:.0f}%"
                )
                
                # Individual checks
                pass_rates = val_iid.get('pass_rates', {})
            
            checks = [
                ("KS Test (h-year)", pass_rates.get('ks_h_year', 0),
                 "Tests if h-year return distribution matches S&P 500 historical distribution"),
                ("KS Test (annual)", pass_rates.get('ks_annual', 0),
                 "Tests if annual return distribution matches S&P 500 historical distribution"),
                ("Return Bounds", pass_rates.get('return_bounds', 0),
                 "Checks if min/max returns fall within S&P 500 historical range"),
                ("Volatility Bounds", pass_rates.get('volatility_bounds', 0),
                 "Checks if return volatility is within ±5% of S&P 500 volatility"),
            ]
            
            for check_name, rate, help_text in checks:
                status = "✓" if rate >= 0.7 else "✗"
                color = "validation-pass" if rate >= 0.7 else "validation-fail"
                st.markdown(f'<span class="{color}" title="{help_text}">{status} {check_name}: {rate*100:.0f}% pass</span>', unsafe_allow_html=True)
                st.caption(help_text)
            
            # Autocorrelation
            autocorrs = val_iid.get('autocorrelations', [])
            mean_autocorr = np.nanmean(autocorrs) if autocorrs else 0
            st.write(f"Mean Autocorrelation: {mean_autocorr:.4f} (should be ~0)")
        
        # Detailed validation breakdown
        with st.expander("📋 Detailed Validation by Check"):
            tab1, tab2, tab3, tab4 = st.tabs(["KS Tests", "Bounds", "Volatility", "Autocorrelation"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        ks_h = validation['predictable'].get('ks_h_pass', [])
                        ks_1 = validation['predictable'].get('ks_annual_pass', [])
                        for i, (h, a) in enumerate(zip(ks_h, ks_1)):
                            st.write(f"Path {i+1}: H={'✓' if h else '✗'} | Ann={'✓' if a else '✗'}")
                    else:
                        st.write("No data available")
                
                with col2:
                    st.write("**I.I.D.**")
                    if 'iid' in validation:
                        ks_h = validation['iid'].get('ks_h_pass', [])
                        ks_1 = validation['iid'].get('ks_annual_pass', [])
                        for i, (h, a) in enumerate(zip(ks_h, ks_1)):
                            st.write(f"Path {i+1}: H={'✓' if h else '✗'} | Ann={'✓' if a else '✗'}")
                    else:
                        st.write("No data available")
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        bounds = validation['predictable'].get('bounds_pass', [])
                        for i, b in enumerate(bounds):
                            st.write(f"Path {i+1}: {'✓' if b else '✗'}")
                    else:
                        st.write("No data available")
                
                with col2:
                    st.write("**I.I.D.**")
                    if 'iid' in validation:
                        bounds = validation['iid'].get('bounds_pass', [])
                        for i, b in enumerate(bounds):
                            st.write(f"Path {i+1}: {'✓' if b else '✗'}")
                    else:
                        st.write("No data available")
            
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        vol = validation['predictable'].get('volatility_pass', [])
                        for i, v in enumerate(vol):
                            st.write(f"Path {i+1}: {'✓' if v else '✗'}")
                    else:
                        st.write("No data available")
                
                with col2:
                    st.write("**I.I.D.**")
                    if 'iid' in validation:
                        vol = validation['iid'].get('volatility_pass', [])
                        for i, v in enumerate(vol):
                            st.write(f"Path {i+1}: {'✓' if v else '✗'}")
                    else:
                        st.write("No data available")
            
            with tab4:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Predictable**")
                    if 'predictable' in validation:
                        autocorrs = validation['predictable'].get('autocorrelations', [])
                        for i, ac in enumerate(autocorrs):
                            st.write(f"Path {i+1}: {ac:.4f}")
                    else:
                        st.write("No data available")
                
                with col2:
                    st.write("**I.I.D.**")
                    if 'iid' in validation:
                        autocorrs = validation['iid'].get('autocorrelations', [])
                        for i, ac in enumerate(autocorrs):
                            st.write(f"Path {i+1}: {ac:.4f}")
                    else:
                        st.write("No data available")
        
        st.divider()
    
    # Individual Path Details Table (PRIORITY 3)
    if validation and ('predictable' in validation or 'iid' in validation):
        st.subheader("📋 Path Details")
        
        # Build dataframe
        path_details = []
        
        # Predictable paths
        if 'predictable' in validation and results['correlations_pred'] is not None:
            val_pred = validation['predictable']
            for i in range(len(results['correlations_pred'])):
                reg = val_pred['regressions'][i]
                path_details.append({
                    'Path ID': i + 1,
                    'Type': 'Predictable',
                    'Correlation': f"{results['correlations_pred'][i]:.4f}",
                    'β(signal)': f"{reg['beta_signal']:.4f}" if not np.isnan(reg['beta_signal']) else 'NaN',
                    'R²': f"{reg['r2_signal']:.4f}" if not np.isnan(reg['r2_signal']) else 'NaN',
                    'KS(h-yr)': '✓' if val_pred['ks_h_pass'][i] else '✗',
                    'KS(ann)': '✓' if val_pred['ks_annual_pass'][i] else '✗',
                    'Bounds': '✓' if val_pred['bounds_pass'][i] else '✗',
                    'Vol': '✓' if val_pred['volatility_pass'][i] else '✗',
                    'Status': 'Full ✓' if val_pred['full_compliant'][i] else 'Partial'
                })
        
        # IID paths
        if 'iid' in validation and results['correlations_iid'] is not None:
            val_iid = validation['iid']
            for i in range(len(results['correlations_iid'])):
                reg = val_iid['regressions'][i]
                path_details.append({
                    'Path ID': config.n_paths_predictable + i + 1,
                    'Type': 'I.I.D.',
                    'Correlation': f"{results['correlations_iid'][i]:.4f}",
                    'β(signal)': f"{reg['beta_signal']:.4f}" if not np.isnan(reg['beta_signal']) else 'NaN',
                    'R²': f"{reg['r2_signal']:.4f}" if not np.isnan(reg['r2_signal']) else 'NaN',
                    'KS(h-yr)': '✓' if val_iid['ks_h_pass'][i] else '✗',
                    'KS(ann)': '✓' if val_iid['ks_annual_pass'][i] else '✗',
                    'Bounds': '✓' if val_iid['bounds_pass'][i] else '✗',
                    'Vol': '✓' if val_iid['volatility_pass'][i] else '✗',
                    'Status': 'Full ✓' if val_iid['full_compliant'][i] else 'Partial'
                })
        
        if path_details:  # Only show table if there are paths
            df_paths = pd.DataFrame(path_details)
            st.dataframe(df_paths, use_container_width=True, height=400)
        
        st.divider()
    
    # Correlation distributions
    if results['correlations_pred'] is not None or results['correlations_iid'] is not None:
        st.subheader("Correlation Distributions")
        
        fig = go.Figure()
        
        if results['correlations_pred'] is not None:
            fig.add_trace(go.Histogram(
                x=results['correlations_pred'],
                name='Predictable',
                marker_color='#3b82f6',
                opacity=0.7,
                nbinsx=20
            ))
        
        if results['correlations_iid'] is not None:
            fig.add_trace(go.Histogram(
                x=results['correlations_iid'],
                name='I.I.D.',
                marker_color='#ef4444',
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        barmode='overlay',
        xaxis_title='Correlation',
        yaxis_title='Count',
        showlegend=True,
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Sample paths visualization with Regression Statistics (PRIORITY 5)
    st.subheader("Sample Paths")
    
    # Determine which tabs to show
    tabs_to_show = []
    if config.n_paths_predictable > 0:
        tabs_to_show.append("Predictable")
    if config.n_paths_iid > 0:
        tabs_to_show.append("I.I.D.")
    
    if not tabs_to_show:
        st.info("No paths generated")
    elif len(tabs_to_show) == 1:
        # Single type - don't use tabs
        pass
    else:
        # Both types - use tabs
        pass
    
    # Create tabs only if both exist
    if len(tabs_to_show) == 2:
        tab1, tab2 = st.tabs(tabs_to_show)
    
    # Render predictable paths
    if config.n_paths_predictable > 0:
        # Use tab if both types exist, otherwise render directly
        if len(tabs_to_show) == 2:
            with tab1:
                render_pred_paths = True
        else:
            render_pred_paths = True
        
        if render_pred_paths:
            # Show first 3 predictable paths
            for i in range(min(3, config.n_paths_predictable)):
                fig = make_subplots()
                
                H = config.time_horizon
                T_is = config.years_insample
                N_is = T_is // H
                
                r_plot = results['returns_pred'][H:T_is+1:H, i] * 100
                s_plot = results['signal_pred'][0:T_is:H, i] * 100
                periods = np.arange(len(r_plot))
                
                fig.add_trace(go.Scatter(
                    x=periods, y=r_plot,
                    name='Returns',
                    line=dict(color='#ef4444', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=periods, y=s_plot,
                    name='Signal',
                    line=dict(color='#3b82f6', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"Predictable Path {i+1} (r={results['correlations_pred'][i]:.3f})",
                    xaxis_title=f'Time ({H}-year periods)',
                    yaxis_title='Annualized Return (%)',
                    height=300,
                    template='plotly_white',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Regression statistics (PRIORITY 5)
                if validation and 'predictable' in validation:
                    with st.expander(f"📊 Path {i+1} Regression Statistics"):
                        reg = validation['predictable']['regressions'][i]
                        autocorr = validation['predictable']['autocorrelations'][i]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Signal Regression**")
                            status = "✓" if reg['pval_signal'] < 0.05 else "✗"
                            st.write(f"β = {reg['beta_signal']:.4f} {status}")
                            st.write(f"p-value = {reg['pval_signal']:.4f}")
                            st.write(f"R² = {reg['r2_signal']:.4f}")
                        
                        with col2:
                            st.write("**Lag Regression**")
                            status = "✓" if reg['pval_lag'] > 0.05 else "✗"
                            st.write(f"β = {reg['beta_lag']:.4f} {status}")
                            st.write(f"p-value = {reg['pval_lag']:.4f}")
                            st.write(f"R² = {reg['r2_lag']:.4f}")
                        
                        st.write(f"**Autocorrelation:** ρ = {autocorr:.4f}")
    
    # Render IID paths
    if config.n_paths_iid > 0:
        # Use tab if both types exist, otherwise render directly
        if len(tabs_to_show) == 2:
            with tab2:
                render_iid_paths = True
        else:
            render_iid_paths = True
        
        if render_iid_paths:
            # Show first 3 IID paths
            for i in range(min(3, config.n_paths_iid)):
                fig = make_subplots()
                
                H = config.time_horizon
                T_is = config.years_insample
                N_is = T_is // H
                
                r_plot = results['returns_iid'][H:T_is+1:H, i] * 100
                s_plot = results['signal_iid'][0:T_is:H, i] * 100
                periods = np.arange(len(r_plot))
                
                fig.add_trace(go.Scatter(
                    x=periods, y=r_plot,
                    name='Returns',
                    line=dict(color='#ef4444', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=periods, y=s_plot,
                    name='Signal',
                    line=dict(color='#3b82f6', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"I.I.D. Path {i+1} (r={results['correlations_iid'][i]:.3f})",
                    xaxis_title=f'Time ({H}-year periods)',
                    yaxis_title='Annualized Return (%)',
                    height=300,
                    template='plotly_white',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Regression statistics
                if validation and 'iid' in validation:
                    with st.expander(f"📊 Path {i+1} Regression Statistics"):
                        reg = validation['iid']['regressions'][i]
                        autocorr = validation['iid']['autocorrelations'][i]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Signal Regression**")
                            status = "✓" if reg['pval_signal'] < 0.05 else "✗"
                            st.write(f"β = {reg['beta_signal']:.4f} {status}")
                            st.write(f"p-value = {reg['pval_signal']:.4f}")
                            st.write(f"R² = {reg['r2_signal']:.4f}")
                        
                        with col2:
                            st.write("**Lag Regression**")
                            status = "✓" if reg['pval_lag'] > 0.05 else "✗"
                            st.write(f"β = {reg['beta_lag']:.4f} {status}")
                            st.write(f"p-value = {reg['pval_lag']:.4f}")
                            st.write(f"R² = {reg['r2_lag']:.4f}")
                        
                        st.write(f"**Autocorrelation:** ρ = {autocorr:.4f}")
    
    st.divider()
    
    # Export Controls (PRIORITY 4)
    st.header("💾 Export Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download path details CSV
        if validation:
            csv = df_paths.to_csv(index=False)
            st.download_button(
                label="Download All Data (CSV)",
                data=csv,
                file_name=f"path_details_{st.session_state.run_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Download summary JSON
        summary_data = {
            'run_id': st.session_state.run_id,
            'config': {
                'time_horizon': config.time_horizon,
                'target_correlation': config.target_correlation,
                'n_paths_predictable': config.n_paths_predictable,
                'n_paths_iid': config.n_paths_iid,
            },
            'correlations_pred': results['correlations_pred'].tolist() if results['correlations_pred'] is not None else [],
            'correlations_iid': results['correlations_iid'].tolist() if results['correlations_iid'] is not None else [],
        }
        
        # Convert validation data, handling numpy types
        if validation:
            import copy
            validation_copy = copy.deepcopy(validation)
            
            # Convert numpy bools to Python bools in validation dict
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                else:
                    return obj
            
            summary_data['validation'] = convert_numpy_types(validation_copy)
        
        json_str = json.dumps(summary_data, indent=2)
        st.download_button(
            label="Download Summary (JSON)",
            data=json_str,
            file_name=f"summary_{st.session_state.run_id}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        # Save to disk
        if st.button("💾 Save to Disk", use_container_width=True):
            save_dir = Path(__file__).parent / "saved_runs" / st.session_state.run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(save_dir / "config.json", 'w') as f:
                json.dump(summary_data['config'], f, indent=2)
            
            # Save validation
            with open(save_dir / "validation.json", 'w') as f:
                json.dump(validation, f, indent=2)
            
            # Save paths
            np.savez(
                save_dir / "paths.npz",
                returns_pred=results['returns_pred'],
                signal_pred=results['signal_pred'],
                returns_iid=results['returns_iid'],
                signal_iid=results['signal_iid']
            )
            
            st.success(f"✓ Saved to {save_dir}")
    
    
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to get started")
