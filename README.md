# Weather Energy Predictor

A high school data science project by Mahmoud Kaddour. The Python script models hourly electricity use (kWh) with linear regression using temperature, humidity, wind speed, pressure, and calendar features.

## Run locally

Install Python 3.11 or newer and Git, then run:

```sh
git clone https://github.com/Mahmoud22002/weather-energy-predictor.git
cd weather-energy-predictor
python -m venv .venv
```

Activate the virtual environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. If your system uses `python3`, use that command to create the environment.

```sh
python -m pip install -r requirements.txt
python Code.py
```

The script reads the bundled CSV beside `Code.py`. Close each plot window to continue to the next step.

## Data and results

The CSV contains 26,328 hourly weather and electricity records. The script drops five incomplete rows, displays a correlation heatmap, and fits a scikit-learn linear regression baseline. The shuffled 80/20 split (`random_state=42`) gave an R² of 0.130 and a mean absolute error of 0.1215 kWh. The script displays actual versus predicted values and a residual histogram.

Possible extensions include lag features, time-based validation, and comparisons with nonlinear models.
