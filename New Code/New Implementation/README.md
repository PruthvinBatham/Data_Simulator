# Simulated Paths Script

A standalone Python script for generating simulated return paths based on the Andries et al. (2025) behavioral finance methodology.

## Purpose

This script generates return paths that can be either:
1.  **Predictable**: Returns are partially predictable by a persistent signal (AR(1) process).
2.  **I.I.D.**: Returns are independent and identically distributed (no predictability), but with similar unconditional distributions to the predictable paths.

The script is designed to select paths that match a specific target correlation between the signal and future $H$-year returns, facilitating experiments on whether subjects can detect this predictability.

## Features

-   **Targeted Correlation**: Simulates thousands of paths and selects those that match a specific target correlation (e.g., $\rho \approx 0.50$ for 2-year returns).
-   **Validation**: Optionally validates simulated paths against historical S&P 500 data (K-S tests, bounds checks).
-   **Comprehensive Output**: Saves full simulation data, summary statistics, and visualization plots.

## Requirements

The script requires Python and the following packages:
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `scipy`
-   `openpyxl` (for reading Excel data)

Install them via:
```bash
pip install -r requirements.txt
```

## Usage

1.  **Configure Parameters**:
    Open `simulated_paths.py` and modify the parameters in the top section:
    ```python
    # Example configurations
    N_PATHS_PREDICTABLE = 30   # Number of predictable paths to keep
    N_PATHS_IID = 0           # Number of I.I.D. paths to keep
    H = 2                     # Time horizon (years)
    TARGET_CORRELATION = 0.50 # Target signal-return correlation
    VALIDATE_WITH_SPX = True  # Enable/disable validation
    ```

2.  **Run the Script**:
    ```bash
    python simulated_paths.py
    ```

## Outputs

All outputs are saved to the `output/` directory:

-   **`simulation_results.csv`**: A single CSV containing time-series data for all selected paths.
    -   Columns: `path_id`, `path_type`, `time`, `return_h`, `signal`, etc.
-   **`summary.json`**: JSON file with simulation parameters and validation statistics for all selected paths.
-   **`plots/`**: Directory containing PNG plots of selected paths (Signal vs. Returns).

## Directory Structure

```
New Implementation/
├── simulated_paths.py    # Main script
├── requirements.txt      # Dependencies
└── output/               # Generated results
    ├── simulation_results.csv
    ├── summary.json
    └── plots/
```
