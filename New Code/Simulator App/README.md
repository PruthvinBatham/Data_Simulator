# Path Simulation App

## Installation

1.  **Navigate to the app directory**:
    ```bash
    cd "New Code/Simulator App"
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Start the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser**:
    The app should automatically open in your default browser at `http://localhost:8501`.

3.  **Run a simulation**:
    - Select a **Preset** (e.g., "Current (2-year)") or choose "Custom".
    - Adjust parameters if needed (Time Horizon, Target Correlation, etc.).
    - Ensure **Enable SPX Validation** is checked if you have the `SPX_5Y_Returns.xlsx` data file (autodetected if in Data/data used in original code path).
    - Click ** Run Simulation**.

## Project Structure

- `app.py`: Main Streamlit application and UI logic.
- `simulation_engine.py`: Core logic for generating paths and performing validation.
- `requirements.txt`: Python package dependencies.
