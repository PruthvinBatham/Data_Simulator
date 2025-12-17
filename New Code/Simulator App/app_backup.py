"""
Path Simulation App
A clean, minimalistic Streamlit interface for running return path simulations
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

from simulation_engine import SimulationConfig, PathSimulator

# Page configuration
st.set_page_config(
    page_title="Path Simulation",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for minimalistic styling
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
            min_value=1,
            max_value=50,
            value=preset_params["n_paths_predictable"]
        )
    with col2:
        n_paths_iid = st.number_input(
            "I.I.D. Paths",
            min_value=1,
            max_value=50,
            value=preset_params["n_paths_iid"]
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
    
    # SPX Data Path
    spx_path = st.text_input(
        "S&P 500 Data Path (optional)",
        value="",
        help="Path to SPX_5Y_Returns.xlsx for validation"
    )
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Main content area
if run_button:
    with st.spinner("Running simulation..."):
        # Determine SPX path
        spx_data_path = None
        validate_spx = False
        
        if spx_path and Path(spx_path).exists():
            spx_data_path = spx_path
            validate_spx = True
        else:
            # Try default path
            default_path = Path(__file__).parent.parent.parent / "Data" / "data used in original code" / "SPX_5Y_Returns.xlsx"
            if default_path.exists():
                spx_data_path = str(default_path)
                validate_spx = True
        
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

# Display results
if st.session_state.results is not None:
    results = st.session_state.results
    config = st.session_state.config
    
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
        mean_corr_pred = np.mean(results['correlations_pred'])
        st.metric(
            "Mean Correlation (Pred)",
            f"{mean_corr_pred:.3f}"
        )
    
    with col4:
        mean_corr_iid = np.mean(results['correlations_iid'])
        st.metric(
            "Mean Correlation (IID)",
            f"{mean_corr_iid:.3f}"
        )
    
    st.divider()
    
    # Correlation distributions
    st.subheader("Correlation Distributions")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=results['correlations_pred'],
        name='Predictable',
        marker_color='#3b82f6',
        opacity=0.7,
        nbinsx=20
    ))
    
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
    
    # Sample paths visualization
    st.subheader("Sample Paths")
    
    tab1, tab2 = st.tabs(["Predictable", "I.I.D."])
    
    with tab1:
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
    
    with tab2:
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
    
    st.divider()
    
    # Validation results (if available)
    if results['validation']:
        st.subheader("Validation Results")
        
        val_pred = results['validation']['predictable']
        val_iid = results['validation']['iid']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predictable Paths**")
            if val_pred['ks_h_pass']:
                pass_rate_h = sum(val_pred['ks_h_pass']) / len(val_pred['ks_h_pass']) * 100
                st.metric(f"KS Test ({config.time_horizon}-year)", f"{pass_rate_h:.0f}% pass")
            if val_pred['ks_annual_pass']:
                pass_rate_annual = sum(val_pred['ks_annual_pass']) / len(val_pred['ks_annual_pass']) * 100
                st.metric("KS Test (annual)", f"{pass_rate_annual:.0f}% pass")
        
        with col2:
            st.write("**I.I.D. Paths**")
            if val_iid['ks_h_pass']:
                pass_rate_h = sum(val_iid['ks_h_pass']) / len(val_iid['ks_h_pass']) * 100
                st.metric(f"KS Test ({config.time_horizon}-year)", f"{pass_rate_h:.0f}% pass")
            if val_iid['ks_annual_pass']:
                pass_rate_annual = sum(val_iid['ks_annual_pass']) / len(val_iid['ks_annual_pass']) * 100
                st.metric("KS Test (annual)", f"{pass_rate_annual:.0f}% pass")
    
    st.divider()
    
    # Save results section
    st.header("Save Results")
    
    save_results = st.checkbox("Save simulation results", value=True)
    
    if save_results:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text_input(
                "Run ID",
                value=st.session_state.run_id,
                disabled=True,
                help="Unique identifier for this simulation run"
            )
        
        with col2:
            if st.button("💾 Save", use_container_width=True):
                # Create save directory
                save_dir = Path(__file__).parent / "saved_runs" / st.session_state.run_id
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save configuration
                config_dict = {
                    'run_id': st.session_state.run_id,
                    'timestamp': datetime.now().isoformat(),
                    'parameters': {
                        'n_paths_predictable': config.n_paths_predictable,
                        'n_paths_iid': config.n_paths_iid,
                        'time_horizon': config.time_horizon,
                        'target_correlation': config.target_correlation,
                        'correlation_window': config.correlation_window,
                        'mu': config.mu,
                        'sigma_eps': config.sigma_eps,
                        'phi': config.phi,
                        'sigma_delta': config.sigma_delta,
                        'rho': config.rho,
                        'n_simulations': config.n_simulations,
                        'random_seed': config.random_seed
                    }
                }
                
                with open(save_dir / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Save results summary
                summary = {
                    'correlations_predictable': results['correlations_pred'].tolist(),
                    'correlations_iid': results['correlations_iid'].tolist(),
                    'mean_correlation_predictable': float(np.mean(results['correlations_pred'])),
                    'mean_correlation_iid': float(np.mean(results['correlations_iid'])),
                }
                
                with open(save_dir / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                # Save path data
                np.savez(
                    save_dir / "paths.npz",
                    returns_pred=results['returns_pred'],
                    signal_pred=results['signal_pred'],
                    returns_iid=results['returns_iid'],
                    signal_iid=results['signal_iid']
                )
                
                st.success(f"✓ Results saved to: {save_dir}")
    
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to get started")
