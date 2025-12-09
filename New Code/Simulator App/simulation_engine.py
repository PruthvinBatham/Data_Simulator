"""
Return Path Simulation Engine for Behavioral Finance Experiment
Based on Andries et al. (2025) methodology

This module provides the core simulation functionality for generating
predictable and i.i.d. return paths with a predictive signal.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    # Path configuration
    n_paths_predictable: int = 15
    n_paths_iid: int = 15
    
    # Time horizon
    time_horizon: int = 2  # H: Returns averaged over H years
    
    # Target correlation for predictable paths
    target_correlation: float = 0.50
    correlation_window: float = 0.02
    
    # Return parameters (from Cochrane 2009, US equity data)
    mu: float = 0.0607        # Mean annual log return
    sigma_eps: float = 0.192  # Annual return volatility
    
    # Signal parameters (AR(1) process)
    phi: float = 0.92            # Persistence coefficient
    sigma_delta: float = 0.152   # Signal volatility
    rho: float = -0.72           # Correlation between signal and return shocks
    
    # Simulation settings
    n_simulations: int = 10000   # Generate this many before selecting best
    years_insample: int = 200    # Years of historical data
    years_oos: int = 25          # Years of out-of-sample data
    random_seed: int = 42        # For reproducibility
    
    # Validation settings
    validate_with_spx: bool = True
    spx_data_path: Optional[str] = None
    ks_alpha: float = 0.01
    
    # Output settings
    save_example_plots: bool = True
    n_example_plots: int = 6
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.n_paths_predictable < 0 or self.n_paths_iid < 0:
            raise ValueError("Number of paths cannot be negative")
        
        if self.n_paths_predictable == 0 and self.n_paths_iid == 0:
            raise ValueError("At least one path type must be requested (predictable or i.i.d.)")



class PathSimulator:
    """Main simulator class for generating return paths"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results = None
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Calculate derived parameters
        self.N_is = config.years_insample // config.time_horizon
        self.T_is = self.N_is * config.time_horizon
        self.N_oos = config.years_oos // config.time_horizon
        self.T_oos = self.N_oos * config.time_horizon
        self.T = self.T_is + self.T_oos + 1
        
        # Calculate b' coefficient for h-year returns
        self.b_prime_factor = sum(config.phi**i for i in range(config.time_horizon))
        
        # Calculate b to achieve target R²
        target_r2 = config.target_correlation ** 2
        var_signal = config.sigma_delta**2 / (1 - config.phi**2)
        var_return = config.sigma_eps**2 * config.time_horizon
        self.B = np.sqrt(target_r2 * var_return / (var_signal * self.b_prime_factor**2))
        
        # Load SPX data if validation enabled
        self.spx_returns_h = None
        self.spx_returns_annual = None
        if config.validate_with_spx and config.spx_data_path:
            print(f"\n{'='*80}")
            print(f"LOADING SPX DATA FOR VALIDATION")
            print(f"Path: {config.spx_data_path}")
            print(f"Time Horizon: {config.time_horizon}")
            print(f"{'='*80}")
            self._load_spx_data()
    
    def _load_spx_data(self):
        """Load S&P 500 data for validation"""
        try:
            print(f"  Reading Excel file...")
            # SPX data from Excel - header is in row 7 (0-indexed = row 6)
            spx_data = pd.read_excel(self.config.spx_data_path, header=6)
            
            # Strip column names
            spx_data.columns = [str(col).strip() for col in spx_data.columns]
            
            print(f"  SPX columns found: {spx_data.columns.tolist()}")
            
            # Get date column
            if 'Date' in spx_data.columns:
                spx_data['Date'] = pd.to_datetime(spx_data['Date'], errors='coerce')
                spx_data.set_index('Date', inplace=True)
                spx_data.sort_index(inplace=True)
            
            # Find return columns - look for 'Return (A)' for annual returns
            annual_return_col = None
            for col in spx_data.columns:
                if 'Return (A)' in col and 'Y' not in col:
                    annual_return_col = col
                    break
            
            if annual_return_col is None:
                raise ValueError(
                    f"Could not find annual return column. "
                    f"Available columns: {spx_data.columns.tolist()}"
                )
            
            print(f"  Using annual returns from: '{annual_return_col}'")
            
            # Convert to numeric
            spx_data[annual_return_col] = pd.to_numeric(spx_data[annual_return_col], errors='coerce')
            
            # Store annual returns
            self.spx_returns_annual = spx_data[annual_return_col].dropna().values
            
            # Try to find h-year return column dynamically based on time_horizon
            h_year_return_col = None
            for col in spx_data.columns:
                if f'{self.config.time_horizon}Y Return' in col:
                    h_year_return_col = col
                    break
            
            if h_year_return_col is not None:
                # Use existing h-year column
                print(f"  Using {self.config.time_horizon}-year returns from: '{h_year_return_col}'")
                spx_data[h_year_return_col] = pd.to_numeric(spx_data[h_year_return_col], errors='coerce')
                self.spx_returns_h = spx_data[h_year_return_col].dropna().values
            else:
                # Calculate h-year returns from annual returns
                print(f"  {self.config.time_horizon}Y Return column not found, calculating from annual data...")
                
                # Calculate cumulative returns over H years
                returns_cumulative = []
                annual_returns = spx_data[annual_return_col].values
                
                for i in range(len(annual_returns) - self.config.time_horizon + 1):
                    # Get H consecutive annual returns
                    h_year_slice = annual_returns[i:i+self.config.time_horizon]
                    
                    # Skip if any NaN values
                    if np.any(np.isnan(h_year_slice)):
                        continue
                    
                    # Calculate h-year average return
                    avg_return = np.mean(h_year_slice)
                    returns_cumulative.append(avg_return)
                
                self.spx_returns_h = np.array(returns_cumulative)
                print(f"  Calculated {len(self.spx_returns_h)} {self.config.time_horizon}-year returns from annual data")
            
            print(f"  Loaded {len(self.spx_returns_annual)} annual return observations from S&P 500")
            print(f"  Loaded {len(self.spx_returns_h)} {self.config.time_horizon}-year return observations from S&P 500")
            print(f"  SPX {self.config.time_horizon}-year returns: mean={np.mean(self.spx_returns_h):.4f}, std={np.std(self.spx_returns_h):.4f}")
            print(f"  SPX annual returns: mean={np.mean(self.spx_returns_annual):.4f}, std={np.std(self.spx_returns_annual):.4f}")
            print(f"  KS test significance level: α={self.config.ks_alpha}")
            print()
            
        except FileNotFoundError:
            print(f"  WARNING: Could not find SPX data at {self.config.spx_data_path}")
            print(f"  Continuing without validation...")
            print(f"  Note: Validation remains enabled in config but will be skipped due to missing data")
            print()
        except Exception as e:
            print(f"  WARNING: Error loading SPX data: {e}")
            print(f"  Continuing without validation...")
            print(f"  Note: Validation remains enabled in config but will be skipped due to data loading error")
            print()
    
    def generate_correlated_shocks(self, T: int, N: int, rho: float, sig1: float, sig2: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two correlated Normal shocks using Cholesky decomposition"""
        R = np.array([[1, rho], [rho, 1]])
        L = np.linalg.cholesky(R)
        Z = np.random.randn(T, 2, N)
        M = np.einsum('jk,tks->tjs', L, Z)
        shock1 = M[:, 0, :] * sig1
        shock2 = M[:, 1, :] * sig2
        return shock1, shock2
    
    def simulate_predictable_paths(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate predictable return paths"""
        T = self.T
        H = self.config.time_horizon
        
        signal = np.zeros((T, N))
        returns_annual = np.zeros((T, N))
        returns_h = np.zeros((T, N))
        signal_fitted = np.zeros((T, N))
        
        delta, eps = self.generate_correlated_shocks(
            T, N, self.config.rho, self.config.sigma_delta, self.config.sigma_eps
        )
        
        for t in range(1, T):
            signal[t, :] = self.config.phi * signal[t-1, :] + delta[t, :]
        
        for t in range(1, T):
            returns_annual[t, :] = self.config.mu + self.B * signal[t-1, :] + eps[t, :]
        
        for t in range(H, T, H):
            returns_h[t, :] = np.sum(returns_annual[t-(H-1):t+1, :], axis=0) / H
        
        b_prime = self.B * self.b_prime_factor
        for t in range(0, T-H, H):
            signal_fitted[t, :] = self.config.mu + b_prime * signal[t, :]
        
        return returns_h, signal_fitted, returns_annual
    
    def simulate_iid_paths(self, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate i.i.d. return paths (no predictability)"""
        T = self.T
        H = self.config.time_horizon
        
        returns_annual = np.zeros((T, N))
        returns_h = np.zeros((T, N))
        signal_fitted = np.zeros((T, N))
        
        sigma_adj = np.sqrt(self.config.sigma_eps**2 + (self.B * self.config.sigma_delta)**2)
        eps = np.random.randn(T, N) * sigma_adj
        
        for t in range(1, T):
            returns_annual[t, :] = self.config.mu + eps[t, :]
        
        for t in range(H, T, H):
            returns_h[t, :] = np.sum(returns_annual[t-(H-1):t+1, :], axis=0) / H
        
        signal = np.zeros((T, N))
        delta = np.random.randn(T, N) * self.config.sigma_delta
        
        for t in range(1, T):
            signal[t, :] = self.config.phi * signal[t-1, :] + delta[t, :]
        
        b_prime = self.B * self.b_prime_factor
        for t in range(0, T-H, H):
            signal_fitted[t, :] = self.config.mu + b_prime * signal[t, :]
        
        return returns_h, signal_fitted, returns_annual
    
    def calculate_correlation(self, returns_h: np.ndarray, signal_h: np.ndarray) -> np.ndarray:
        """Calculate correlation between h-year returns and lagged signal"""
        H = self.config.time_horizon
        r = returns_h[H:self.T_is:H, :]
        s = signal_h[0:self.T_is-H:H, :]
        
        correlations = []
        for i in range(r.shape[1]):
            mask = ~(np.isnan(r[:, i]) | np.isnan(s[:, i]))
            if mask.sum() > 2:
                corr = np.corrcoef(r[mask, i], s[mask, i])[0, 1]
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        
        return np.array(correlations)
    
    def select_best_paths(self, correlations: np.ndarray, target: float, window: float, n_select: int) -> Tuple[np.ndarray, np.ndarray]:
        """Select paths with correlation closest to target"""
        valid_mask = ~np.isnan(correlations) & (np.abs(correlations - target) <= window)
        valid_indices = np.where(valid_mask)[0]
        valid_corrs = correlations[valid_mask]
        
        if len(valid_indices) < n_select:
            distances = np.abs(correlations - target)
            distances[np.isnan(correlations)] = 999
            selected_indices = np.argsort(distances)[:n_select]
        else:
            distances = np.abs(valid_corrs - target)
            sorted_idx = np.argsort(distances)[:n_select]
            selected_indices = valid_indices[sorted_idx]
        
        selected_corrs = correlations[selected_indices]
        return selected_indices, selected_corrs
    
    def calculate_autocorrelation(self, returns_h: np.ndarray) -> np.ndarray:
        """Calculate lagged return autocorrelation"""
        H = self.config.time_horizon
        autocorrs = []
        
        for i in range(returns_h.shape[1]):
            r = returns_h[H:self.T_is:H, i]
            r = r[~np.isnan(r)]
            
            if len(r) < 3:
                autocorrs.append(np.nan)
                continue
            
            if len(r) > 1:
                autocorr = np.corrcoef(r[:-1], r[1:])[0, 1]
            else:
                autocorr = 0
            
            autocorrs.append(autocorr)
        
        return np.array(autocorrs)
    
    def validate_regression(self, returns_h: np.ndarray, signal_h: np.ndarray) -> List[Dict]:
        """Run validation regressions and return statistics"""
        H = self.config.time_horizon
        results = []
        
        for i in range(returns_h.shape[1]):
            r = returns_h[H:self.T_is:H, i]
            s = signal_h[0:self.T_is-H:H, i]
            r_lag = returns_h[0:self.T_is-H:H, i]
            
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
            
            se_signal = np.sqrt(ss_res_signal / (len(r) - 2)) / np.sqrt(np.sum((s - np.mean(s))**2))
            t_stat_signal = beta_signal / se_signal if se_signal > 0 else 0
            pval_signal = 2 * (1 - stats.t.cdf(abs(t_stat_signal), len(r) - 2))
            
            # Regression on lagged return
            X_lag = np.column_stack([np.ones(len(r_lag)), r_lag])
            beta_lag = np.linalg.lstsq(X_lag, r, rcond=None)[0][1]
            resid_lag = r - X_lag @ np.linalg.lstsq(X_lag, r, rcond=None)[0]
            ss_res_lag = np.sum(resid_lag**2)
            r2_lag = 1 - (ss_res_lag / ss_tot) if ss_tot > 0 else 0
            
            se_lag = np.sqrt(ss_res_lag / (len(r) - 2)) / np.sqrt(np.sum((r_lag - np.mean(r_lag))**2))
            t_stat_lag = beta_lag / se_lag if se_lag > 0 else 0
            pval_lag = 2 * (1 - stats.t.cdf(abs(t_stat_lag), len(r) - 2))
            
            results.append({
                'beta_signal': beta_signal, 'pval_signal': pval_signal, 'r2_signal': r2_signal,
                'beta_lag': beta_lag, 'pval_lag': pval_lag, 'r2_lag': r2_lag
            })
        
        return results
    
    def validate_distribution_ks(self, returns_h: np.ndarray, returns_annual: np.ndarray) -> Tuple[List[bool], List[bool], List[Dict]]:
        """
        Validate distributions using KS tests
        
        KS TEST LOGIC (CORRECT STATISTICAL INTERPRETATION):
        - Pass if p-value > alpha (p > 0.01)
        - This means: distributions are SIMILAR (fail to reject null) ✓
        
        NOTE: The original notebook uses p ≤ alpha as "passing" which is backwards!
        We use the CORRECT interpretation here for meaningful validation.
        
        Returns lists of booleans: True if simulation passes test (distributions similar)
        """
        if not self.config.validate_with_spx or self.spx_returns_h is None:
            return [], [], []
        
        H = self.config.time_horizon
        results_h = []
        results_annual = []
        ks_stats = []
        
        for i in range(returns_h.shape[1]):
            r_h = returns_h[H:self.T_is:H, i]
            r_h = r_h[~np.isnan(r_h)]
            
            r_1 = returns_annual[1:self.T_is+1, i]
            r_1 = r_1[~np.isnan(r_1)]
            
            if len(r_h) < 3 or len(r_1) < 3:
                results_h.append(False)
                results_annual.append(False)
                ks_stats.append({'ks_h_stat': np.nan, 'ks_h_pval': np.nan, 
                               'ks_1_stat': np.nan, 'ks_1_pval': np.nan})
                continue
            
            ks_h_stat, ks_h_pval = stats.ks_2samp(r_h, self.spx_returns_h)
            ks_1_stat, ks_1_pval = stats.ks_2samp(r_1, self.spx_returns_annual)
            
            # USING CORRECT KS INTERPRETATION (not notebook's backwards logic)
            # Pass if p-value > alpha (distributions are similar)
            # This is the statistically correct interpretation:
            # - p > alpha: fail to reject null = distributions SIMILAR ✓
            # - p ≤ alpha: reject null = distributions DIFFERENT ✗
            pass_h = ks_h_pval > self.config.ks_alpha
            pass_annual = ks_1_pval > self.config.ks_alpha
            
            results_h.append(pass_h)
            results_annual.append(pass_annual)
            ks_stats.append({
                'ks_h_stat': ks_h_stat, 'ks_h_pval': ks_h_pval,
                'ks_1_stat': ks_1_stat, 'ks_1_pval': ks_1_pval
            })
        
        return results_h, results_annual, ks_stats
    
    def validate_return_bounds(self, returns_h: np.ndarray, spx_min: float, spx_max: float) -> List[bool]:
        """Check if min and max returns fall within SPX historical bounds"""
        H = self.config.time_horizon
        results = []
        
        for i in range(returns_h.shape[1]):
            r = returns_h[H:self.T_is:H, i]
            r = r[~np.isnan(r)]
            
            if len(r) < 1:
                results.append(False)
                continue
            
            r_min = r.min()
            r_max = r.max()
            
            min_ok = (spx_min <= r_min <= spx_max)
            max_ok = (spx_min <= r_max <= spx_max)
            
            results.append(min_ok and max_ok)
        
        return results
    
    def validate_volatility_bounds(self, returns_h: np.ndarray, spx_std: float, tolerance: float = 0.05) -> List[bool]:
        """Check if return volatility is within acceptable range of SPX"""
        H = self.config.time_horizon
        results = []
        
        for i in range(returns_h.shape[1]):
            r = returns_h[H:self.T_is:H, i]
            r = r[~np.isnan(r)]
            
            if len(r) < 2:
                results.append(False)
                continue
            
            r_std = r.std()
            ok = (spx_std - tolerance <= r_std <= spx_std + tolerance)
            results.append(ok)
        
        return results
    
    def run(self) -> Dict:
        """Run the complete simulation"""
        # Initialize variables
        returns_pred = signal_pred = returns_pred_annual = None
        returns_iid = signal_iid = returns_iid_annual = None
        idx_pred = selected_corr_pred = None
        idx_iid = selected_corr_iid = None
        
        if self.config.n_paths_predictable > 0:
            print("Generating predictable paths...")
            returns_pred, signal_pred, returns_pred_annual = self.simulate_predictable_paths(
                self.config.n_simulations
            )
        
        if self.config.n_paths_iid > 0:
            print("Generating i.i.d. paths...")
            returns_iid, signal_iid, returns_iid_annual = self.simulate_iid_paths(
                self.config.n_simulations
            )
        
        print("Calculating correlations and selecting best paths...")
        
        # Calculate correlations and select paths
        if self.config.n_paths_predictable > 0:
            corr_pred = self.calculate_correlation(returns_pred, signal_pred)
            idx_pred, selected_corr_pred = self.select_best_paths(
                corr_pred, self.config.target_correlation, self.config.correlation_window,
                self.config.n_paths_predictable
            )
        
        if self.config.n_paths_iid > 0:
            corr_iid = self.calculate_correlation(returns_iid, signal_iid)
            idx_iid, selected_corr_iid = self.select_best_paths(
                corr_iid, 0.0, self.config.correlation_window, self.config.n_paths_iid
            )
        
        # Extract selected paths
        returns_pred_selected = signal_pred_selected = returns_pred_annual_selected = None
        returns_iid_selected = signal_iid_selected = returns_iid_annual_selected = None
        
        if self.config.n_paths_predictable > 0:
            returns_pred_selected = returns_pred[:, idx_pred]
            signal_pred_selected = signal_pred[:, idx_pred]
            returns_pred_annual_selected = returns_pred_annual[:, idx_pred]
        
        if self.config.n_paths_iid > 0:
            returns_iid_selected = returns_iid[:, idx_iid]
            signal_iid_selected = signal_iid[:, idx_iid]
            returns_iid_annual_selected = returns_iid_annual[:, idx_iid]
        
        # Comprehensive validation
        validation_results = {}
        if self.config.validate_with_spx and self.spx_returns_h is not None:
            print("Running comprehensive validation...")
            
            # SPX statistics
            spx_min = self.spx_returns_h.min()
            spx_max = self.spx_returns_h.max()
            spx_std = self.spx_returns_h.std()
            spx_mean = self.spx_returns_h.mean()
            spx_annual_mean = self.spx_returns_annual.mean()
            spx_annual_std = self.spx_returns_annual.std()
            
            validation_results = {
                'spx_stats': {
                    'h_year': {
                        'mean': float(spx_mean),
                        'std': float(spx_std),
                        'min': float(spx_min),
                        'max': float(spx_max),
                        'n_obs': len(self.spx_returns_h)
                    },
                    'annual': {
                        'mean': float(spx_annual_mean),
                        'std': float(spx_annual_std),
                        'n_obs': len(self.spx_returns_annual)
                    }
                }
            }
            
            # Validate predictable paths
            if self.config.n_paths_predictable > 0:
                ks_h_pred, ks_1_pred, ks_stats_pred = self.validate_distribution_ks(
                    returns_pred_selected, returns_pred_annual_selected
                )
                bounds_pred = self.validate_return_bounds(returns_pred_selected, spx_min, spx_max)
                vol_pred = self.validate_volatility_bounds(returns_pred_selected, spx_std)
                autocorr_pred = self.calculate_autocorrelation(returns_pred_selected)
                regression_pred = self.validate_regression(returns_pred_selected, signal_pred_selected)
                
                # Calculate full compliance (all 4 checks)
                full_compliant_pred = [
                    ks_h and ks_1 and bounds and vol
                    for ks_h, ks_1, bounds, vol in zip(ks_h_pred, ks_1_pred, bounds_pred, vol_pred)
                ]
                
                validation_results['predictable'] = {
                    'ks_h_pass': ks_h_pred,
                    'ks_annual_pass': ks_1_pred,
                    'ks_stats': ks_stats_pred,
                    'bounds_pass': bounds_pred,
                    'volatility_pass': vol_pred,
                    'autocorrelations': autocorr_pred.tolist(),
                    'regressions': regression_pred,
                    'full_compliant': full_compliant_pred,
                    'pass_rates': {
                        'ks_h_year': float(sum(ks_h_pred) / len(ks_h_pred)),
                        'ks_annual': float(sum(ks_1_pred) / len(ks_1_pred)),
                        'return_bounds': float(sum(bounds_pred) / len(bounds_pred)),
                        'volatility_bounds': float(sum(vol_pred) / len(vol_pred)),
                        'full_compliance': float(sum(full_compliant_pred) / len(full_compliant_pred))
                    }
                }
            
            # IID paths validation
            if self.config.n_paths_iid > 0:
                ks_h_iid, ks_1_iid, ks_stats_iid = self.validate_distribution_ks(
                    returns_iid_selected, returns_iid_annual_selected
                )
                bounds_iid = self.validate_return_bounds(returns_iid_selected, spx_min, spx_max)
                vol_iid = self.validate_volatility_bounds(returns_iid_selected, spx_std)
                autocorr_iid = self.calculate_autocorrelation(returns_iid_selected)
                regression_iid = self.validate_regression(returns_iid_selected, signal_iid_selected)
                
                # Calculate full compliance (all 4 checks)
                full_compliant_iid = [
                    ks_h and ks_1 and bounds and vol
                    for ks_h, ks_1, bounds, vol in zip(ks_h_iid, ks_1_iid, bounds_iid, vol_iid)
                ]
                
                validation_results['iid'] = {
                    'ks_h_pass': ks_h_iid,
                    'ks_annual_pass': ks_1_iid,
                    'ks_stats': ks_stats_iid,
                    'bounds_pass': bounds_iid,
                    'volatility_pass': vol_iid,
                    'autocorrelations': autocorr_iid.tolist(),
                    'regressions': regression_iid,
                    'full_compliant': full_compliant_iid,
                    'pass_rates': {
                        'ks_h_year': float(sum(ks_h_iid) / len(ks_h_iid)),
                        'ks_annual': float(sum(ks_1_iid) / len(ks_1_iid)),
                        'return_bounds': float(sum(bounds_iid) / len(bounds_iid)),
                        'volatility_bounds': float(sum(vol_iid) / len(vol_iid)),
                        'full_compliance': float(sum(full_compliant_iid) / len(full_compliant_iid))
                    }
                }
        
        # Store results
        self.results = {
            'returns_pred': returns_pred_selected,
            'signal_pred': signal_pred_selected,
            'returns_pred_annual': returns_pred_annual_selected,
            'returns_iid': returns_iid_selected,
            'signal_iid': signal_iid_selected,
            'returns_iid_annual': returns_iid_annual_selected,
            'correlations_pred': selected_corr_pred,
            'correlations_iid': selected_corr_iid,
            'validation': validation_results,
            'config': self.config
        }
        
        print("Simulation complete!")
        return self.results
