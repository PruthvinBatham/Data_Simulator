# Return Path Simulation Project

This project implements behavioral finance simulations based on Andries et al. (2025). It generates return paths that are either predictable (by a persistent signal) or i.i.d., allowing for experimental testing of human ability to detect return predictability.

## Project Structure

-   **`New Code/`**: The modern implementations of the simulation logic.
    -   **`Simulator App/`**: An interactive Streamlit tailored for easy viewing and parameterized execution. **(Recommended)**
    -   **`New Implementation/`**: A standalone Python script (`simulated_paths.py`) for generating bulk data.
-   **`Data/`**: Contains the necessary data files for validation.
-   **`Original Code/`**: Reference implementations (Jupyter Notebooks).
-   **`Research Articles/`**: Background literature.

## ⚠️ Important: Data Setup for Validation

To enable the validation features (comparing simulated paths against S&P 500 history), you must ensure the reference data file is present.

**This file is NOT included in the GitHub repository** due to licensing/distribution reasons.

1.  **Obtain the file**: `SPX_5Y_Returns.xlsx`
2.  **Place it here**:
    ```
    Data/data used in original code/SPX_5Y_Returns.xlsx
    ```

**Note**: Both the Simulator App and the Python script will automatically look for this file in that location. If it is missing, validation features will be disabled.

## Getting Started

### Option 1: Interactive App (Recommended)
Go to `New Code/Simulator App` and follow the [README](New%20Code/Simulator%20App/README.md) to run the Streamlit interface.

### Option 2: Python Script
Go to `New Code/New Implementation` and follow the [README](New%20Code/New%20Implementation/README.md) to run the simulation script.
