"""
Return Path Simulation for Behavioral Finance Experiment
Based on Andries et al. (2025) methodology

Generates predictable and i.i.d. return paths with a predictive signal


1. simulate_predictable_paths() and simulate_iid_paths() now return BOTH:
   - signal_actual: The raw AR(1) signal (always stochastic)
   - signal_fitted: The fitted values MU + b_prime * signal (flat when B=0)
   
2. Plots now show signal_actual instead of signal_fitted
   - This makes the signal visible even when TARGET_CORRELATION = 0
   
3. CSV output includes both signal_actual and signal_fitted columns

Why this matters:
- When TARGET_CORRELATION = 0, the coefficient B = 0
- This means: returns = MU + 0*signal + eps (signal has no predictive power)
- The fitted signal becomes: signal_fitted = MU + 0*signal = MU (constant!)
- But the actual signal still varies according to the AR(1) process
- Participants should see the varying signal, just with zero predictive power
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETERS - Adjust these as needed
# ============================================================================

# Fix file paths - Data folder is 2 levels up from this script
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "Data" / "data used in original code"

filename = "SPX_5Y_Returns.xlsx"   # you can change this file name for different file for SPX returns
SPX_DATA_PATH = DATA_DIR / filename

# Number of paths to select
N_PATHS_PREDICTABLE = 30  
N_PATHS_IID = 0          # Optional

# Time horizon
H = 2  # Returns are averaged over H years 

# Target correlation for predictable paths
# For 2-year returns: correlation ~ 0.50 gives R² ~ 0.25
# For 5-year returns: correlation ~ 0.57 gives R² ~ 0.33
TARGET_CORRELATION = 0.50
CORRELATION_WINDOW = 0.02  
# Return parameters (from Cochrane 2009, US equity data)
MU = 0.0607        # Mean annual log return (6.07%) | I calibrated these like the notebook you sent me over via email and not the SPX data directly
SIGMA_EPS = 0.192  # Annual return volatility (19.2%) | Same here

# Signal parameters (AR(1) process)
PHI = 0.92            
SIGMA_DELTA = 0.152   # Signal volatility
RHO = -0.72           # Correlation between signal and return shocks

# Simulation settings
N_SIMULATIONS = 10000  # Generate this many before selecting best ones
YEARS_INSAMPLE = 200   # Years of historical data
YEARS_OOS = 25         # Years of out-of-sample data
RANDOM_SEED = 42       # For reproducibility

# Validation settings
VALIDATE_WITH_SPX = False  # Compare simulated distributions with real S&P 500 data, I set false because our simulator is not calibrated to SPX data 
KS_ALPHA = 0.01  # Significance level for Kolmogorov-Smirnov test 

# Output settings
SAVE_EXAMPLE_PLOTS = True   # Save a few example plots
N_EXAMPLE_PLOTS = 6         # How many examples (3 predictable + 3 i.i.d.)

# Validate path configuration
if N_PATHS_PREDICTABLE < 0 or N_PATHS_IID < 0:
    raise ValueError("Number of paths cannot be negative")

if N_PATHS_PREDICTABLE == 0 and N_PATHS_IID == 0:
    raise ValueError("At least one path type must be requested (predictable or i.i.d.)")

total_paths = N_PATHS_PREDICTABLE + N_PATHS_IID

print("="*80)
print("SIMULATION PARAMETERS")
print("="*80)
print(f"Paths: {N_PATHS_PREDICTABLE} predictable + {N_PATHS_IID} i.i.d. = {total_paths} total")
print(f"Time horizon: {H} years")
print(f"Target correlation: {TARGET_CORRELATION:.3f} (R² ≈ {TARGET_CORRELATION**2:.3f})")
print(f"Correlation window: ±{CORRELATION_WINDOW:.3f}")
print(f"Simulations to generate: {N_SIMULATIONS:,}")
print("="*80)
print()

# ============================================================================
# LOAD SPX DATA FOR VALIDATION (if enabled)
# ============================================================================

spx_returns_h = None
spx_returns_annual = None

if VALIDATE_WITH_SPX:
    print("Loading S&P 500 data for validation...")
    try:
        # Load SPX data from Excel - header is in row 7 (0-indexed = row 6)
        spx_data = pd.read_excel(SPX_DATA_PATH, header=6)
        
        # Clean column names
        spx_data.columns = [str(col).strip() for col in spx_data.columns]
        
        print(f"  Columns found: {spx_data.columns.tolist()}")
        
        # Get date column
        if 'Date' in spx_data.columns:
            spx_data['Date'] = pd.to_datetime(spx_data['Date'], errors='coerce')
            spx_data.set_index('Date', inplace=True)
            spx_data.sort_index(inplace=True)
        
        # Find return columns - look for 'Return (A)' for annual returns
        annual_return_col = None
        for col in spx_data.columns:
            if 'Return (A)' in col and '5Y' not in col:
                annual_return_col = col
                break
        
        # Find 5Y return column
        h_year_return_col = None
        for col in spx_data.columns:
            if '5Y Return' in col:
                h_year_return_col = col
                break
        
        if annual_return_col is None or h_year_return_col is None:
            raise ValueError(f"Could not find return columns. Available columns: {spx_data.columns.tolist()}")
        
        print(f"  Using annual returns from: '{annual_return_col}'")
        print(f"  Using {H}-year returns from: '{h_year_return_col}'")
        
        # Convert to numeric - values are already in decimal format, NOT percentages
        # So we do NOT divide by 100
        spx_data[annual_return_col] = pd.to_numeric(spx_data[annual_return_col], errors='coerce')
        spx_data[h_year_return_col] = pd.to_numeric(spx_data[h_year_return_col], errors='coerce')
        
        # Store returns (remove any NaN values)
        spx_returns_annual = spx_data[annual_return_col].dropna().values
        spx_returns_h = spx_data[h_year_return_col].dropna().values
        
        print(f"  Loaded {len(spx_returns_annual)} annual return observations from S&P 500")
        print(f"  Loaded {len(spx_returns_h)} {H}-year return observations from S&P 500")
        print(f"  SPX {H}-year returns: mean={np.mean(spx_returns_h):.4f}, std={np.std(spx_returns_h):.4f}")
        print(f"  SPX annual returns: mean={np.mean(spx_returns_annual):.4f}, std={np.std(spx_returns_annual):.4f}")
        print(f"  KS test significance level: α={KS_ALPHA}")
        print()
        
    except FileNotFoundError:
        print(f"  WARNING: Could not find SPX data at {SPX_DATA_PATH}")
        print(f"  Continuing without validation...")
        VALIDATE_WITH_SPX = False
        print()
    except Exception as e:
        print(f"  WARNING: Error loading SPX data: {e}")
        print(f"  Continuing without validation...")
        VALIDATE_WITH_SPX = False
        print()

# Set random seed
np.random.seed(RANDOM_SEED)

# Calculate derived parameters
N_is = YEARS_INSAMPLE // H
T_is = N_is * H
N_oos = YEARS_OOS // H
T_oos = N_oos * H
T = T_is + T_oos + 1  # Total time periods

# Calculate b' coefficient for h-year returns
# b' = b * sum(phi^i) for i=0 to h-1
# This comes from summing the AR(1) process over h periods
b_prime_factor = sum(PHI**i for i in range(H))

# Calculate b to achieve target R²
# For predictable returns: r = mu + b*signal + noise
# R² = Var(b*signal) / Var(r)
# We solve for b given target R²
target_r2 = TARGET_CORRELATION ** 2
var_signal = SIGMA_DELTA**2 / (1 - PHI**2)  # Variance of AR(1) process
var_return = SIGMA_EPS**2 * H  # Variance of h-year returns
B = np.sqrt(target_r2 * var_return / (var_signal * b_prime_factor**2))

print(f"Calculated b coefficient: {B:.4f}")
print(f"Calculated b' (h-year): {B * b_prime_factor:.4f}")
print()

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def generate_correlated_shocks(T, N, rho, sig1, sig2):
    """
    Generate two correlated Normal shocks using Cholesky decomposition
    
    Math: If we want X and Y with correlation rho:
    1. Generate independent Z1, Z2 ~ N(0,1)
    2. Set X = sig1 * Z1
    3. Set Y = sig2 * (rho*Z1 + sqrt(1-rho²)*Z2)
    
    This is equivalent to: [X, Y] = L * [Z1, Z2] where L is Cholesky factor
    """
    # Correlation matrix
    R = np.array([[1, rho], [rho, 1]])
    
    # Cholesky decomposition: R = L * L'
    L = np.linalg.cholesky(R)
    
    # Generate standard normal shocks
    Z = np.random.randn(T, 2, N)
    
    # Apply correlation structure: M = L * Z
    M = np.einsum('jk,tks->tjs', L, Z)
    
    # Scale by volatilities
    shock1 = M[:, 0, :] * sig1
    shock2 = M[:, 1, :] * sig2
    
    return shock1, shock2


def simulate_predictable_paths(N, T, H):
    """
    Simulate predictable return paths
    
    Model:
        signal_t = phi * signal_{t-1} + delta_t
        r_t = mu + b * signal_{t-1} + eps_t
    
    Where delta and eps are correlated with correlation rho
    
    For h-year returns:
        r_h,t = (1/h) * sum(r_t to r_{t+h-1})
        E[r_h,t] = mu + b' * signal_t  where b' = b * sum(phi^i)
    """
    # Initialize arrays
    signal = np.zeros((T, N))
    returns_annual = np.zeros((T, N))
    returns_h = np.zeros((T, N))
    signal_fitted = np.zeros((T, N))
    
    # Generate correlated shocks
    # delta affects signal, eps affects returns
    delta, eps = generate_correlated_shocks(T, N, RHO, SIGMA_DELTA, SIGMA_EPS)
    
    # Simulate signal process (AR(1))
    for t in range(1, T):
        signal[t, :] = PHI * signal[t-1, :] + delta[t, :]
    
    # Simulate annual returns
    for t in range(1, T):
        returns_annual[t, :] = MU + B * signal[t-1, :] + eps[t, :]
    
    # Aggregate to h-year returns
    # Sum h annual returns and divide by h to get annualized rate
    for t in range(H, T, H):
        returns_h[t, :] = np.sum(returns_annual[t-(H-1):t+1, :], axis=0) / H
    
    # Calculate fitted values (conditional expectations)
    # This is what the signal predicts for next h-year return
    b_prime = B * b_prime_factor
    for t in range(0, T-H, H):
        signal_fitted[t, :] = MU + b_prime * signal[t, :]
    
    return returns_h, signal, signal_fitted, returns_annual


def simulate_iid_paths(N, T, H):
    """
    Simulate i.i.d. return paths (no predictability)
    
    Model:
        r_t = mu + eps_t  where eps ~ N(0, sigma²)
    
    Signal is generated independently with same distribution as predictable case
    This ensures visual similarity but zero correlation
    """
    # Initialize arrays
    returns_annual = np.zeros((T, N))
    returns_h = np.zeros((T, N))
    signal_fitted = np.zeros((T, N))
    
    # Generate i.i.d. returns
    # Adjust sigma so unconditional variance matches predictable case
    sigma_adj = np.sqrt(SIGMA_EPS**2 + (B * SIGMA_DELTA)**2)
    eps = np.random.randn(T, N) * sigma_adj
    
    for t in range(1, T):
        returns_annual[t, :] = MU + eps[t, :]
    
    # Aggregate to h-year returns
    for t in range(H, T, H):
        returns_h[t, :] = np.sum(returns_annual[t-(H-1):t+1, :], axis=0) / H
    
    # Generate independent signal (for visual similarity only)
    signal = np.zeros((T, N))
    delta = np.random.randn(T, N) * SIGMA_DELTA
    
    for t in range(1, T):
        signal[t, :] = PHI * signal[t-1, :] + delta[t, :]
    
    b_prime = B * b_prime_factor
    for t in range(0, T-H, H):
        signal_fitted[t, :] = MU + b_prime * signal[t, :]
    
    return returns_h, signal, signal_fitted, returns_annual


def calculate_correlation(returns_h, signal_h, H):
    """Calculate correlation between h-year returns and lagged signal"""
    # Extract h-year observation points
    r = returns_h[H:T_is:H, :]
    s = signal_h[0:T_is-H:H, :]
    
    # Calculate correlation for each simulation
    correlations = []
    for i in range(r.shape[1]):
        # Remove any NaN values
        mask = ~(np.isnan(r[:, i]) | np.isnan(s[:, i]))
        if mask.sum() > 2:
            corr = np.corrcoef(r[mask, i], s[mask, i])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    
    return np.array(correlations)


def select_best_paths(correlations, target, window, n_select):
    """
    Select paths with correlation closest to target (within window)
    
    Returns indices of selected paths and their correlations
    """
    # Find paths within window
    valid_mask = ~np.isnan(correlations) & (np.abs(correlations - target) <= window)
    valid_indices = np.where(valid_mask)[0]
    valid_corrs = correlations[valid_mask]
    
    if len(valid_indices) < n_select:
        print(f"  WARNING: Only {len(valid_indices)} paths within window")
        print(f"  Requested {n_select} paths. Consider widening window.")
        # Fall back to closest correlations
        distances = np.abs(correlations - target)
        distances[np.isnan(correlations)] = 999
        selected_indices = np.argsort(distances)[:n_select]
    else:
        # Sort by distance from target and select closest
        distances = np.abs(valid_corrs - target)
        sorted_idx = np.argsort(distances)[:n_select]
        selected_indices = valid_indices[sorted_idx]
    
    selected_corrs = correlations[selected_indices]
    return selected_indices, selected_corrs


def run_validation_regression(returns_h, signal_h, H):
    """
    Run validation regressions:
    1. r_t = alpha + beta * signal_{t-1} + error
    2. r_t = alpha + beta * r_{t-1} + error
    
    For predictable paths, expect:
    - Regression 1: beta ~ 1, R² ~ target_r2, p < 0.01
    - Regression 2: beta ~ 0, R² ~ 0, p > 0.05
    """
    results = []
    
    for i in range(returns_h.shape[1]):
        # Extract h-year data points
        r = returns_h[H:T_is:H, i]
        s = signal_h[0:T_is-H:H, i]
        r_lag = returns_h[0:T_is-H:H, i]
        
        # Remove NaN values
        mask = ~(np.isnan(r) | np.isnan(s) | np.isnan(r_lag))
        r, s, r_lag = r[mask], s[mask], r_lag[mask]
        
        if len(r) < 3:
            results.append({
                'beta_signal': np.nan, 'pval_signal': np.nan, 'r2_signal': np.nan,
                'beta_lag': np.nan, 'pval_lag': np.nan, 'r2_lag': np.nan
            })
            continue
        
        # Regression on signal
        X_signal = np.column_stack([np.ones(len(s)), s])
        beta_signal = np.linalg.lstsq(X_signal, r, rcond=None)[0][1]
        resid_signal = r - X_signal @ np.linalg.lstsq(X_signal, r, rcond=None)[0]
        ss_res_signal = np.sum(resid_signal**2)
        ss_tot = np.sum((r - np.mean(r))**2)
        r2_signal = 1 - (ss_res_signal / ss_tot) if ss_tot > 0 else 0
        
        # t-test for beta_signal
        se_signal = np.sqrt(ss_res_signal / (len(r) - 2)) / np.sqrt(np.sum((s - np.mean(s))**2))
        t_stat_signal = beta_signal / se_signal if se_signal > 0 else 0
        pval_signal = 2 * (1 - stats.t.cdf(abs(t_stat_signal), len(r) - 2))
        
        # Regression on lagged return
        X_lag = np.column_stack([np.ones(len(r_lag)), r_lag])
        beta_lag = np.linalg.lstsq(X_lag, r, rcond=None)[0][1]
        resid_lag = r - X_lag @ np.linalg.lstsq(X_lag, r, rcond=None)[0]
        ss_res_lag = np.sum(resid_lag**2)
        r2_lag = 1 - (ss_res_lag / ss_tot) if ss_tot > 0 else 0
        
        # t-test for beta_lag
        se_lag = np.sqrt(ss_res_lag / (len(r) - 2)) / np.sqrt(np.sum((r_lag - np.mean(r_lag))**2))
        t_stat_lag = beta_lag / se_lag if se_lag > 0 else 0
        pval_lag = 2 * (1 - stats.t.cdf(abs(t_stat_lag), len(r) - 2))
        
        results.append({
            'beta_signal': beta_signal, 'pval_signal': pval_signal, 'r2_signal': r2_signal,
            'beta_lag': beta_lag, 'pval_lag': pval_lag, 'r2_lag': r2_lag
        })
    
    return results


def calculate_autocorrelation(returns_h, H):
    """
    Calculate lagged return autocorrelation
    
    Tests if returns are serially correlated (they shouldn't be!)
    Computes: correlation(r_t, r_{t-1})
    
    For valid simulation, should be close to 0
    """
    autocorrs = []
    
    for i in range(returns_h.shape[1]):
        r = returns_h[H:T_is:H, i]
        r = r[~np.isnan(r)]
        
        if len(r) < 3:
            autocorrs.append(np.nan)
            continue
        
        # Correlation between r[t] and r[t-1]
        if len(r) > 1:
            autocorr = np.corrcoef(r[:-1], r[1:])[0, 1]
        else:
            autocorr = 0
        
        autocorrs.append(autocorr)
    
    return np.array(autocorrs)


def validate_distribution_ks(returns_h, returns_annual, spx_returns_h, spx_returns_annual, H, alpha=0.01):
    """
    Validate simulated returns using Kolmogorov-Smirnov (KS) two-sample test
    
    Two tests performed:
    1. KS test on h-year returns vs SPX h-year returns
    2. KS test on annual returns vs SPX annual returns
    
    KS TEST LOGIC (CORRECT STATISTICAL INTERPRETATION):
    - Pass if p-value > alpha (p > 0.01)
    - This means: distributions are SIMILAR (fail to reject null) ✓
    
    NOTE: The original notebook uses p ≤ alpha as "passing" which is backwards!
    We use the CORRECT interpretation here for meaningful validation.
    
    Returns lists of booleans: True if simulation passes test (distributions similar)
    """
    results_h = []
    results_annual = []
    ks_stats = []
    
    for i in range(returns_h.shape[1]):
        # Extract h-year returns for this simulation
        r_h = returns_h[H:T_is:H, i]
        r_h = r_h[~np.isnan(r_h)]
        
        # Extract annual returns
        r_1 = returns_annual[1:T_is+1, i]
        r_1 = r_1[~np.isnan(r_1)]
        
        if len(r_h) < 3 or len(r_1) < 3:
            results_h.append(False)
            results_annual.append(False)
            ks_stats.append({'ks_h_stat': np.nan, 'ks_h_pval': np.nan, 
                           'ks_1_stat': np.nan, 'ks_1_pval': np.nan})
            continue
        
        # Run KS test on h-year returns
        ks_h_stat, ks_h_pval = stats.ks_2samp(r_h, spx_returns_h)
        
        # Run KS test on annual returns
        ks_1_stat, ks_1_pval = stats.ks_2samp(r_1, spx_returns_annual)
        
        # USING CORRECT KS INTERPRETATION (not notebook's backwards logic)
        # Pass if p-value > alpha (distributions are similar)
        # This is the statistically correct interpretation:
        # - p > alpha: fail to reject null = distributions SIMILAR ✓
        # - p ≤ alpha: reject null = distributions DIFFERENT ✗
        pass_h = ks_h_pval > alpha
        pass_annual = ks_1_pval > alpha
        
        results_h.append(pass_h)
        results_annual.append(pass_annual)
        ks_stats.append({
            'ks_h_stat': ks_h_stat, 'ks_h_pval': ks_h_pval,
            'ks_1_stat': ks_1_stat, 'ks_1_pval': ks_1_pval
        })
    
    return results_h, results_annual, ks_stats


def validate_return_bounds(returns_h, spx_min, spx_max, H):
    """
    Check if min and max returns fall within SPX historical bounds
    
    Ensures simulated returns aren't unrealistically extreme
    """
    results = []
    
    for i in range(returns_h.shape[1]):
        r = returns_h[H:T_is:H, i]
        r = r[~np.isnan(r)]
        
        if len(r) < 1:
            results.append(False)
            continue
        
        r_min = r.min()
        r_max = r.max()
        
        # Both min and max should be within SPX range
        min_ok = (spx_min <= r_min <= spx_max)
        max_ok = (spx_min <= r_max <= spx_max)
        
        results.append(min_ok and max_ok)
    
    return results


def validate_volatility_bounds(returns_h, spx_std, H, tolerance=0.05):
    """
    Check if return volatility is within acceptable range of SPX
    
    Default tolerance: ±0.05 (same as notebook)
    """
    results = []
    
    for i in range(returns_h.shape[1]):
        r = returns_h[H:T_is:H, i]
        r = r[~np.isnan(r)]
        
        if len(r) < 2:
            results.append(False)
            continue
        
        r_std = r.std()
        
        # Check if within tolerance
        ok = (spx_std - tolerance <= r_std <= spx_std + tolerance)
        results.append(ok)
    
    return results


# ============================================================================
# MAIN SIMULATION
# ============================================================================

# Initialize variables
returns_pred = signal_pred_actual = signal_pred = returns_pred_annual = None
returns_iid = signal_iid_actual = signal_iid = returns_iid_annual = None
idx_pred = selected_corr_pred = None
idx_iid = selected_corr_iid = None

if N_PATHS_PREDICTABLE > 0:
    print("Generating predictable paths...")
    returns_pred, signal_pred_actual, signal_pred, returns_pred_annual = simulate_predictable_paths(N_SIMULATIONS, T, H)

if N_PATHS_IID > 0:
    print("Generating i.i.d. paths...")
    returns_iid, signal_iid_actual, signal_iid, returns_iid_annual = simulate_iid_paths(N_SIMULATIONS, T, H)

print()
print("Calculating correlations and selecting best paths...")

# Calculate correlations and select paths
if N_PATHS_PREDICTABLE > 0:
    corr_pred = calculate_correlation(returns_pred, signal_pred, H)
    idx_pred, selected_corr_pred = select_best_paths(
        corr_pred, TARGET_CORRELATION, CORRELATION_WINDOW, N_PATHS_PREDICTABLE
    )

if N_PATHS_IID > 0:
    corr_iid = calculate_correlation(returns_iid, signal_iid, H)
    idx_iid, selected_corr_iid = select_best_paths(
        corr_iid, 0.0, CORRELATION_WINDOW, N_PATHS_IID
    )

# Validate distributions with SPX data if enabled
if VALIDATE_WITH_SPX and spx_returns_h is not None:
    print()
    print("="*80)
    print("COMPREHENSIVE VALIDATION AGAINST S&P 500 DATA")
    print("="*80)
    print()
    
    # Extract SPX statistics
    spx_min = spx_returns_h.min()
    spx_max = spx_returns_h.max()
    spx_std = spx_returns_h.std()
    
    print(f"SPX {H}-year returns statistics:")
    print(f"  Min: {spx_min:.4f}, Max: {spx_max:.4f}, Std: {spx_std:.4f}")
    print()
    
    # Run all validation checks on selected paths
    print("Running validation checks...")
    
    # Initialize validation variables
    ks_h_pred = ks_1_pred = ks_stats_pred = None
    ks_h_iid = ks_1_iid = ks_stats_iid = None
    bounds_pred = bounds_iid = None
    vol_pred = vol_iid = None
    autocorr_pred = autocorr_iid = None
    full_compliant_pred = full_compliant_iid = None
    
    # Validate predictable paths
    if N_PATHS_PREDICTABLE > 0:
        ks_h_pred, ks_1_pred, ks_stats_pred = validate_distribution_ks(
            returns_pred[:, idx_pred], 
            returns_pred_annual[:, idx_pred],
            spx_returns_h, 
            spx_returns_annual,
            H, KS_ALPHA
        )
        bounds_pred = validate_return_bounds(returns_pred[:, idx_pred], spx_min, spx_max, H)
        vol_pred = validate_volatility_bounds(returns_pred[:, idx_pred], spx_std, H)
        autocorr_pred = calculate_autocorrelation(returns_pred[:, idx_pred], H)
        
        # Overall compliance
        full_compliant_pred = [ks_h and ks_1 and bounds and vol 
                               for ks_h, ks_1, bounds, vol in zip(ks_h_pred, ks_1_pred, bounds_pred, vol_pred)]
    
    # Validate i.i.d. paths
    if N_PATHS_IID > 0:
        ks_h_iid, ks_1_iid, ks_stats_iid = validate_distribution_ks(
            returns_iid[:, idx_iid],
            returns_iid_annual[:, idx_iid],
            spx_returns_h,
            spx_returns_annual, 
            H, KS_ALPHA
        )
        bounds_iid = validate_return_bounds(returns_iid[:, idx_iid], spx_min, spx_max, H)
        vol_iid = validate_volatility_bounds(returns_iid[:, idx_iid], spx_std, H)
        autocorr_iid = calculate_autocorrelation(returns_iid[:, idx_iid], H)
        
        # Overall compliance
        full_compliant_iid = [ks_h and ks_1 and bounds and vol
                              for ks_h, ks_1, bounds, vol in zip(ks_h_iid, ks_1_iid, bounds_iid, vol_iid)]
    
    # Print results
    print()
    if N_PATHS_PREDICTABLE > 0:
        print("PREDICTABLE PATHS VALIDATION:")
        print(f"  KS test ({H}-year): {sum(ks_h_pred)}/{len(ks_h_pred)} pass")
        print(f"  KS test (annual): {sum(ks_1_pred)}/{len(ks_1_pred)} pass")
        print(f"  Return bounds: {sum(bounds_pred)}/{len(bounds_pred)} pass")
        print(f"  Volatility bounds: {sum(vol_pred)}/{len(vol_pred)} pass")
        print(f"  Mean autocorrelation: {np.mean(autocorr_pred):.4f} (should be ~0)")
        print()
    
    if N_PATHS_IID > 0:
        print("I.I.D. PATHS VALIDATION:")
        print(f"  KS test ({H}-year): {sum(ks_h_iid)}/{len(ks_h_iid)} pass")
        print(f"  KS test (annual): {sum(ks_1_iid)}/{len(ks_1_iid)} pass")
        print(f"  Return bounds: {sum(bounds_iid)}/{len(bounds_iid)} pass")
        print(f"  Volatility bounds: {sum(vol_iid)}/{len(vol_iid)} pass")
        print(f"  Mean autocorrelation: {np.mean(autocorr_iid):.4f} (should be ~0)")
        print()
    
    # Full compliance summary
    print("FULL COMPLIANCE (all 4 checks):")
    if N_PATHS_PREDICTABLE > 0:
        print(f"  Predictable: {sum(full_compliant_pred)}/{len(full_compliant_pred)} fully compliant")
    if N_PATHS_IID > 0:
        print(f"  I.I.D.: {sum(full_compliant_iid)}/{len(full_compliant_iid)} fully compliant")
    
    # Check if warnings needed
    show_warning = False
    if N_PATHS_PREDICTABLE > 0 and sum(full_compliant_pred) < N_PATHS_PREDICTABLE:
        show_warning = True
    if N_PATHS_IID > 0 and sum(full_compliant_iid) < N_PATHS_IID:
        show_warning = True
    
    if show_warning:
        print()
        print("  NOTE: Some paths don't pass all validation checks")
        print("  This is normal - simulated data can differ from historical")
        print("  Consider: (1) Increasing N_SIMULATIONS, or (2) Loosening criteria")
    else:
        print("  ✓ All paths pass full validation!")
    
    print("="*80)
    print()

print()
print("="*80)
print("SELECTION RESULTS")
print("="*80)

if N_PATHS_PREDICTABLE > 0:
    print(f"\nPredictable paths (n={len(idx_pred)}):")
    print(f"  Mean correlation: {np.mean(selected_corr_pred):.4f}")
    print(f"  Std deviation: {np.std(selected_corr_pred):.4f}")
    print(f"  Range: [{np.min(selected_corr_pred):.4f}, {np.max(selected_corr_pred):.4f}]")

if N_PATHS_IID > 0:
    print(f"\nI.I.D. paths (n={len(idx_iid)}):")
    print(f"  Mean correlation: {np.mean(selected_corr_iid):.4f}")
    print(f"  Std deviation: {np.std(selected_corr_iid):.4f}")
    print(f"  Range: [{np.min(selected_corr_iid):.4f}, {np.max(selected_corr_iid):.4f}]")

print("="*80)
print()

# Extract selected paths
returns_pred_selected = signal_pred_actual_selected = signal_pred_selected = None
returns_iid_selected = signal_iid_actual_selected = signal_iid_selected = None

if N_PATHS_PREDICTABLE > 0:
    returns_pred_selected = returns_pred[:, idx_pred]
    signal_pred_actual_selected = signal_pred_actual[:, idx_pred]  # Actual stochastic signal
    signal_pred_selected = signal_pred[:, idx_pred]  # Fitted signal

if N_PATHS_IID > 0:
    returns_iid_selected = returns_iid[:, idx_iid]
    signal_iid_actual_selected = signal_iid_actual[:, idx_iid]  # Actual stochastic signal
    signal_iid_selected = signal_iid[:, idx_iid]  # Fitted signal

# ============================================================================
# VALIDATION
# ============================================================================

print("Running validation regressions...")
print()

val_pred = val_iid = None

if N_PATHS_PREDICTABLE > 0:
    val_pred = run_validation_regression(returns_pred_selected, signal_pred_selected, H)

if N_PATHS_IID > 0:
    val_iid = run_validation_regression(returns_iid_selected, signal_iid_selected, H)

if N_PATHS_PREDICTABLE > 0:
    print("="*80)
    print("VALIDATION - Predictable Paths")
    print("="*80)
    print(f"{'Path':<6} {'β(signal)':<12} {'p-value':<10} {'R²':<8} {'β(r_lag)':<12} {'p-value':<10} {'R²':<8}")
    print("-"*80)
    for i, v in enumerate(val_pred):
        print(f"{i+1:<6} {v['beta_signal']:<12.4f} {v['pval_signal']:<10.4f} {v['r2_signal']:<8.4f} "
              f"{v['beta_lag']:<12.4f} {v['pval_lag']:<10.4f} {v['r2_lag']:<8.4f}")
    print()

if N_PATHS_IID > 0:
    print("="*80)
    print("VALIDATION - I.I.D. Paths")
    print("="*80)
    print(f"{'Path':<6} {'β(signal)':<12} {'p-value':<10} {'R²':<8} {'β(r_lag)':<12} {'p-value':<10} {'R²':<8}")
    print("-"*80)
    for i, v in enumerate(val_iid):
        print(f"{i+1:<6} {v['beta_signal']:<12.4f} {v['pval_signal']:<10.4f} {v['r2_signal']:<8.4f} "
              f"{v['beta_lag']:<12.4f} {v['pval_lag']:<10.4f} {v['r2_lag']:<8.4f}")
    print()

print("="*80)
print()

# ============================================================================
# SAVE DATA
# ============================================================================

print("Saving results...")

# Create output directory
os.makedirs('output', exist_ok=True)

# Build master dataframe with all paths
all_data = []

# Add predictable paths
if N_PATHS_PREDICTABLE > 0:
    for i in range(len(idx_pred)):
        for t in range(T):
            all_data.append({
                'path_id': i + 1,
                'path_type': 'predictable',
                'time': t,
                'return_h': returns_pred_selected[t, i],
                'signal_actual': signal_pred_actual_selected[t, i],  # Actual stochastic signal
                'signal_fitted': signal_pred_selected[t, i],  # Fitted/predicted signal
                'is_observation': 1 if t % H == 0 else 0
            })

# Add i.i.d. paths
if N_PATHS_IID > 0:
    for i in range(len(idx_iid)):
        for t in range(T):
            all_data.append({
                'path_id': N_PATHS_PREDICTABLE + i + 1,
                'path_type': 'iid',
                'time': t,
                'return_h': returns_iid_selected[t, i],
                'signal_actual': signal_iid_actual_selected[t, i],  # Actual stochastic signal
                'signal_fitted': signal_iid_selected[t, i],  # Fitted/predicted signal
                'is_observation': 1 if t % H == 0 else 0
            })

# Save master CSV
df_all = pd.DataFrame(all_data)
df_all.to_csv('output/simulation_results.csv', index=False)
print("  Saved: output/simulation_results.csv")

# Save summary statistics
summary = {
    'parameters': {
        'n_paths_predictable': N_PATHS_PREDICTABLE,
        'n_paths_iid': N_PATHS_IID,
        'time_horizon_years': H,
        'target_correlation': TARGET_CORRELATION,
        'target_r2': target_r2,
        'correlation_window': CORRELATION_WINDOW,
        'mean_return': MU,
        'volatility': SIGMA_EPS,
        'random_seed': RANDOM_SEED
    }
}

# Add predictable path statistics if they exist
if N_PATHS_PREDICTABLE > 0:
    summary['predictable'] = {
        'selected_indices': idx_pred.tolist(),
        'correlations': selected_corr_pred.tolist(),
        'mean_correlation': float(np.mean(selected_corr_pred)),
        'std_correlation': float(np.std(selected_corr_pred)),
        'mean_r2_signal': float(np.mean([v['r2_signal'] for v in val_pred])),
        'std_r2_signal': float(np.std([v['r2_signal'] for v in val_pred])),
        'mean_r2_lag': float(np.mean([v['r2_lag'] for v in val_pred])),
        'std_r2_lag': float(np.std([v['r2_lag'] for v in val_pred])),
        'path_details': [
            {
                'path_id': i + 1,
                'correlation': float(selected_corr_pred[i]),
                'r2_signal': float(val_pred[i]['r2_signal']),
                'r2_lag': float(val_pred[i]['r2_lag']),
                'beta_signal': float(val_pred[i]['beta_signal']),
                'pval_signal': float(val_pred[i]['pval_signal']),
                'beta_lag': float(val_pred[i]['beta_lag']),
                'pval_lag': float(val_pred[i]['pval_lag'])
            }
            for i in range(len(val_pred))
        ]
    }

# Add i.i.d. path statistics if they exist
if N_PATHS_IID > 0:
    summary['iid'] = {
        'selected_indices': idx_iid.tolist(),
        'correlations': selected_corr_iid.tolist(),
        'mean_correlation': float(np.mean(selected_corr_iid)),
        'std_correlation': float(np.std(selected_corr_iid)),
        'mean_r2_signal': float(np.mean([v['r2_signal'] for v in val_iid])),
        'std_r2_signal': float(np.std([v['r2_signal'] for v in val_iid])),
        'mean_r2_lag': float(np.mean([v['r2_lag'] for v in val_iid])),
        'std_r2_lag': float(np.std([v['r2_lag'] for v in val_iid])),
        'path_details': [
            {
                'path_id': N_PATHS_PREDICTABLE + i + 1,
                'correlation': float(selected_corr_iid[i]),
                'r2_signal': float(val_iid[i]['r2_signal']),
                'r2_lag': float(val_iid[i]['r2_lag']),
                'beta_signal': float(val_iid[i]['beta_signal']),
                'pval_signal': float(val_iid[i]['pval_signal']),
                'beta_lag': float(val_iid[i]['beta_lag']),
                'pval_lag': float(val_iid[i]['pval_lag'])
            }
            for i in range(len(val_iid))
        ]
    }

# Add KS validation results if performed
if VALIDATE_WITH_SPX and spx_returns_h is not None:
    summary['validation'] = {
        'ks_alpha': KS_ALPHA,
        'spx_h_year': {
            'mean': float(np.mean(spx_returns_h)),
            'std': float(spx_std),
            'min': float(spx_min),
            'max': float(spx_max),
            'n_obs': len(spx_returns_h)
        },
        'spx_annual': {
            'mean': float(np.mean(spx_returns_annual)),
            'std': float(np.std(spx_returns_annual)),
            'n_obs': len(spx_returns_annual)
        }
    }
    
    # Add predictable validation results if they exist
    if N_PATHS_PREDICTABLE > 0:
        # Re-run validation on final selected paths
        ks_h_pred_final, ks_1_pred_final, ks_stats_pred_final = validate_distribution_ks(
            returns_pred_selected, returns_pred_annual[:, idx_pred],
            spx_returns_h, spx_returns_annual, H, KS_ALPHA
        )
        bounds_pred_final = validate_return_bounds(returns_pred_selected, spx_min, spx_max, H)
        vol_pred_final = validate_volatility_bounds(returns_pred_selected, spx_std, H)
        autocorr_pred_final = calculate_autocorrelation(returns_pred_selected, H)
        
        summary['validation']['predictable'] = {
            'ks_h_year_pass': [bool(x) for x in ks_h_pred_final],
            'ks_annual_pass': [bool(x) for x in ks_1_pred_final],
            'return_bounds_pass': [bool(x) for x in bounds_pred_final],
            'volatility_bounds_pass': [bool(x) for x in vol_pred_final],
            'autocorrelations': [float(x) if not np.isnan(x) else None for x in autocorr_pred_final],
            'pass_rates': {
                'ks_h_year': float(sum(ks_h_pred_final) / len(ks_h_pred_final)),
                'ks_annual': float(sum(ks_1_pred_final) / len(ks_1_pred_final)),
                'return_bounds': float(sum(bounds_pred_final) / len(bounds_pred_final)),
                'volatility_bounds': float(sum(vol_pred_final) / len(vol_pred_final)),
                'full_compliance': float(sum([a and b and c and d for a,b,c,d in 
                    zip(ks_h_pred_final, ks_1_pred_final, bounds_pred_final, vol_pred_final)]) / len(ks_h_pred_final))
            }
        }
    
    # Add i.i.d. validation results if they exist
    if N_PATHS_IID > 0:
        ks_h_iid_final, ks_1_iid_final, ks_stats_iid_final = validate_distribution_ks(
            returns_iid_selected, returns_iid_annual[:, idx_iid],
            spx_returns_h, spx_returns_annual, H, KS_ALPHA
        )
        bounds_iid_final = validate_return_bounds(returns_iid_selected, spx_min, spx_max, H)
        vol_iid_final = validate_volatility_bounds(returns_iid_selected, spx_std, H)
        autocorr_iid_final = calculate_autocorrelation(returns_iid_selected, H)
        
        summary['validation']['iid'] = {
            'ks_h_year_pass': [bool(x) for x in ks_h_iid_final],
            'ks_annual_pass': [bool(x) for x in ks_1_iid_final],
            'return_bounds_pass': [bool(x) for x in bounds_iid_final],
            'volatility_bounds_pass': [bool(x) for x in vol_iid_final],
            'autocorrelations': [float(x) if not np.isnan(x) else None for x in autocorr_iid_final],
            'pass_rates': {
                'ks_h_year': float(sum(ks_h_iid_final) / len(ks_h_iid_final)),
                'ks_annual': float(sum(ks_1_iid_final) / len(ks_1_iid_final)),
                'return_bounds': float(sum(bounds_iid_final) / len(bounds_iid_final)),
                'volatility_bounds': float(sum(vol_iid_final) / len(vol_iid_final)),
                'full_compliance': float(sum([a and b and c and d for a,b,c,d in 
                    zip(ks_h_iid_final, ks_1_iid_final, bounds_iid_final, vol_iid_final)]) / len(ks_h_iid_final))
            }
        }

with open('output/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("  Saved: output/summary.json")

# ============================================================================
# GENERATE EXAMPLE PLOTS
# ============================================================================

if SAVE_EXAMPLE_PLOTS and (N_PATHS_PREDICTABLE > 0 or N_PATHS_IID > 0):
    # Adjust number of plots based on what's available
    n_pred_to_plot = 0
    n_iid_to_plot = 0
    
    if N_PATHS_PREDICTABLE > 0 and N_PATHS_IID > 0:
        # Both types available - split evenly
        n_pred_to_plot = min(N_EXAMPLE_PLOTS // 2, N_PATHS_PREDICTABLE)
        n_iid_to_plot = min(N_EXAMPLE_PLOTS - n_pred_to_plot, N_PATHS_IID)
    elif N_PATHS_PREDICTABLE > 0:
        # Only predictable
        n_pred_to_plot = min(N_EXAMPLE_PLOTS, N_PATHS_PREDICTABLE)
    else:
        # Only i.i.d.
        n_iid_to_plot = min(N_EXAMPLE_PLOTS, N_PATHS_IID)
    
    actual_plots = n_pred_to_plot + n_iid_to_plot
    print()
    print(f"Generating {actual_plots} example plots...")
    
    os.makedirs('output/plots', exist_ok=True)
    
    # Plot predictable examples
    if n_pred_to_plot > 0:
        for i in range(n_pred_to_plot):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Show last 40 h-year periods for decision making
            start_period = max(0, N_is - 40)
            periods = np.arange(start_period, N_is)
            
            # Extract data
            r_plot = returns_pred_selected[H:T_is+1:H, i] * 100
            s_plot = signal_pred_actual_selected[0:T_is:H, i] * 100
            
            # Plot
            ax.plot(periods, r_plot[start_period:], '-r', linewidth=2, label='Returns')
            ax.plot(periods, s_plot[start_period:N_is], '-.b', linewidth=1.5, label='Signal')
            ax.plot(N_is-1, s_plot[N_is-1], 'oy', markersize=12, markeredgewidth=2.5)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel(f'Time (in {H}-year periods)')
            ax.set_ylabel('Annualized Return (%)')
            ax.set_title(f'Predictable Path {i+1} (r={selected_corr_pred[i]:.3f}, R²={val_pred[i]["r2_signal"]:.3f})')
            
            plt.tight_layout()
            plt.savefig(f'output/plots/predictable_example_{i+1}.png', dpi=150)
            plt.close()
    
    # Plot i.i.d. examples
    if n_iid_to_plot > 0:
        for i in range(n_iid_to_plot):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            start_period = max(0, N_is - 40)
            periods = np.arange(start_period, N_is)
            
            r_plot = returns_iid_selected[H:T_is+1:H, i] * 100
            s_plot = signal_iid_actual_selected[0:T_is:H, i] * 100
            
            ax.plot(periods, r_plot[start_period:], '-r', linewidth=2, label='Returns')
            ax.plot(periods, s_plot[start_period:N_is], '-.b', linewidth=1.5, label='Signal')
            ax.plot(N_is-1, s_plot[N_is-1], 'oy', markersize=12, markeredgewidth=2.5)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel(f'Time (in {H}-year periods)')
            ax.set_ylabel('Annualized Return (%)')
            ax.set_title(f'I.I.D. Path {i+1} (r={selected_corr_iid[i]:.3f}, R²={val_iid[i]["r2_signal"]:.3f})')
            
            plt.tight_layout()
            plt.savefig(f'output/plots/iid_example_{i+1}.png', dpi=150)
            plt.close()
    
    print(f"  Saved {actual_plots} plots in output/plots/")

print()
print("="*80)
print("DONE!")
print("="*80)
print("Output files:")
print(f"  output/simulation_results.csv  - All {total_paths} paths in one file")
print("  output/summary.json            - Summary statistics and parameters")
if SAVE_EXAMPLE_PLOTS and (N_PATHS_PREDICTABLE > 0 or N_PATHS_IID > 0):
    print(f"  output/plots/                  - Example plots")
print("="*80)