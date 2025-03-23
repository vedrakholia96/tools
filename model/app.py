from flask import Flask, request, render_template, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import yaml
from scipy.optimize import curve_fit
import math
import traceback

app = Flask(__name__)

##################################
# Hard-coded data & initial fits
##################################
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

DEFAULT_GLUCOSE = np.array(config["default_glucose"])
MU_37 = np.array(config["mu_37"])
MU_33 = np.array(config["mu_33"])

def log_func(x, a, b):
    """Logarithmic growth model: mu = a ln(x) + b."""
    return a * np.log(x) + b

# Fit parameters from your config
params_37, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_37)
params_33, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_33)

@app.route('/', methods=['GET'])
def home():
    """Renders the home page (home.html)."""
    return render_template('home.html')

@app.route('/plot')
def plot():
    """
    Returns a PNG image of the Growth Rate vs Glucose,
    annotated with final cell density if user has requested multiple days.
    Query params:
      ?temp=33 or ?temp=37
      &glucose=...
      &days=...
      &initial_density=...
      &volume=...
    """
    temp_str = request.args.get('temp', '33')
    try:
        glucose_val = float(request.args.get('glucose', '1.2'))
    except ValueError:
        glucose_val = 1.2

    try:
        days = float(request.args.get('days', '1'))
    except ValueError:
        days = 1.0

    try:
        initial_density = float(request.args.get('initial_density', '300000'))
    except ValueError:
        initial_density = 3e5

    try:
        initial_volume = float(request.args.get('volume', '2000'))
    except ValueError:
        initial_volume = 2000

    hours = days * 24.0
    initial_count = initial_density * initial_volume

    # Evaluate mu from the appropriate fit
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)
        color = 'red'
        label = '37°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)
        color = '#BBA14F'
        label = '33°C Fit'
        scatter_vals = log_func(DEFAULT_GLUCOSE, *params_33)

    # Compute final cell count with continuous model
    final_cell_count = initial_count * np.exp(mu * hours)
    final_cell_density = final_cell_count / initial_volume  # cells/mL

    # Fetch daily glucose needed values
    daily_data_response = daily_data()
    daily_data_json = daily_data_response.get_json()
    daily_glucose_needed_values = daily_data_json["daily_glucose_needed_values"]
    avg_daily_glucose_needed = np.mean(daily_glucose_needed_values)

    # PLOT: Growth Rate vs Glucose
    fig, ax = plt.subplots(figsize=(5,4), dpi=100)
    x_vals = np.linspace(DEFAULT_GLUCOSE.min(), DEFAULT_GLUCOSE.max(), 100)
    if temp_str == '37':
        y_fit = log_func(x_vals, *params_37)
    else:
        y_fit = log_func(x_vals, *params_33)

    # Plot fitted curve
    ax.plot(x_vals, y_fit, color=color, label=label)
    # Plot scatter for known points
    ax.scatter(DEFAULT_GLUCOSE, scatter_vals, color=color)
    # Mark the chosen glucose
    ax.scatter([glucose_val], [mu], color='blue', zorder=5)

    ax.set_xlabel('Glucose (g/L)')
    ax.set_ylabel('Specific Growth Rate (hr^-1)')
    ax.set_title('Growth Rate vs Glucose')
    ax.legend()

    # Annotate chosen point
    ax.text(glucose_val + 0.1, mu, f"mu={mu:.4f}", color='blue')

    # Info text box
    info_text = (
        f"Days: {days}\n"
        f"Temp: {temp_str}°C\n"
        f"mu = {mu:.4f} hr^-1\n"
        f"Expected Doubling Time: {math.log(2) / mu:.2f} hr\n"
        f"Initial Density: {initial_density:.2e} cells/mL\n"
        f"Final Density: {final_cell_density:.2e} cells/mL\n"
        f"Avg Daily Glucose Need: {avg_daily_glucose_needed:.3f} g/day"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='gray'))

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/daily_data')
def daily_data():
    """
    Returns JSON with daily predictions for the selected parameters,
    using the log_func and fitted parameters for growth.
    """
    temp_str = request.args.get('temp', '33')
    try:
        glucose_val = float(request.args.get('glucose', '1.2'))
    except ValueError:
        glucose_val = 1.2

    try:
        days = int(request.args.get('days', '1'))
    except ValueError:
        days = 1

    try:
        initial_density = float(request.args.get('initial_density', '300000'))
    except ValueError:
        initial_density = 3e5

    try:
        initial_volume = float(request.args.get('volume', '2000'))
    except ValueError:
        initial_volume = 2000

    initial_count = initial_density * initial_volume

    # Evaluate mu from the appropriate fit
    if temp_str == '37':
        mu = log_func(glucose_val, *params_37)
    else:
        mu = log_func(glucose_val, *params_33)

    daily_growth_factor = np.exp(mu * 24)

    results = []
    daily_glucose_needed_values = []
    X = initial_count
    glucose_consumption_rate = 24 * 0.2  # pmol/cell/day
    MW_GLUCOSE = 180
    VOLUME = initial_volume / 1000  # Convert mL to L
    glucose_to_lactate_conversion_factor = 0.8
    lactate_level = 0
    remaining_glucose = glucose_val

    # pH parameters
    pH_initial = 7.2
    alpha = 0.015  # Empirical factor

    # Day 0 entry
    results.append({
        "day": 0,
        "predicted_density": initial_density,
        "daily_glucose_concentration": remaining_glucose,
        "daily_glucose_needed": 0,
        "lactate_level": lactate_level,
        "pH": pH_initial
    })

    for d in range(1, days+1):
        old_X = X
        X = X * daily_growth_factor
        # daily glucose usage
        daily_glc_needed = (X - old_X) * glucose_consumption_rate * 1e-12 * MW_GLUCOSE
        daily_glucose_needed_values.append(daily_glc_needed)

        remaining_glucose -= daily_glc_needed / VOLUME
        # add lactate
        lactate_level += (daily_glc_needed / MW_GLUCOSE) * glucose_to_lactate_conversion_factor * 2 * 1000 / VOLUME
        cell_density = X / (VOLUME * 1000)  # X per mL

        # Calculate pH
        pH = pH_initial - alpha * lactate_level

        results.append({
            "day": d,
            "predicted_density": cell_density,
            "daily_glucose_concentration": remaining_glucose,
            "daily_glucose_needed": daily_glc_needed,
            "lactate_level": lactate_level,
            "pH": pH
        })

    return jsonify({
        "results": results,
        "daily_glucose_needed_values": daily_glucose_needed_values
    })

############################
# NEW CODE FOR UPLOAD + Actual Predictions
############################
def compute_average_doubling_time(df):
    """
    Given a DataFrame with columns for Time and CellDensity,
    compute the consecutive growth rates and return the average doubling time.
    Also compute viability if 'LiveCells' and 'TotalCells' exist.
    """
    # Ensure time is sorted
    df = df.sort_values(by='day')
    
    # Convert actual_density to numeric to handle any potential string inputs
    df['actual_density'] = pd.to_numeric(df['actual_density'], errors='coerce')
    
    # Remove any rows with missing or invalid density values
    df = df.dropna(subset=['actual_density'])
    
    # Skip calculation if we don't have enough data points
    if len(df) < 2:
        return None, None, df['actual_density'].tolist(), 0

    # Consecutive growth rates
    mus = []
    for i in range(len(df) - 1):
        t1 = df.iloc[i]['day']
        t2 = df.iloc[i+1]['day']
        x1 = df.iloc[i]['actual_density']
        x2 = df.iloc[i+1]['actual_density']

        # Avoid any zero or negative densities
        if x1 <= 0 or x2 <= 0 or (t2 == t1):
            continue

        mu = (math.log(x2) - math.log(x1)) / (t2 - t1)  # day^-1
        mus.append(mu)

    avg_mu = np.mean(mus) if len(mus) > 0 else 0
    if avg_mu <= 0:
        avg_doubling_time = None
    else:
        avg_doubling_time = math.log(2) / avg_mu  # in days

    # If the file has LiveCells and TotalCells, compute viability
    viability = None
    if {'LiveCells', 'TotalCells'}.issubset(df.columns):
        viability = (
            (df['LiveCells'].sum() / df['TotalCells'].sum()) * 100
            if df['TotalCells'].sum() > 0 else None
        )

    return avg_doubling_time, viability, df['actual_density'].tolist(), avg_mu

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload Excel, parse time and cell density,
    compute average doubling time & produce an "actual predictions" table.
    Return JSON so the frontend can display it.
    """
    excel_file = request.files.get('excel_file')
    if not excel_file:
        return "No file chosen. Please select a file.", 400

    try:
        # Read the file into a DataFrame
        df = pd.read_excel(excel_file)

        # We assume the file has columns 'Time' and 'CellDensity'.
        # If your column names differ, rename them or adjust accordingly.
        required_cols = {'Time', 'CellDensity'}
        if not required_cols.issubset(df.columns):
            return "Excel file must contain 'Time' and 'CellDensity' columns.", 400

        # Rename columns to match our expected format
        df = df.rename(columns={'Time': 'day', 'CellDensity': 'actual_density'})
        
        avg_td, viability, actual_densities, avg_mu = compute_average_doubling_time(df)
        if avg_td is None:
            return jsonify({
                "message": "Could not compute doubling time (check data).",
                "actual_predictions": []
            })

        # Get the number of days from the form data
        num_days = int(request.form.get('days', 5))

        # Now generate a day-by-day "Actual Predictions" using the observed doubling time.
        daily_growth_factor = np.exp(avg_mu)

        # Suppose we just take the initial cell density from the first row of the data:
        initial_density = df.iloc[0]['actual_density']

        actual_results = []
        X = initial_density
        actual_results.append({
            "day": 0,
            "actual_density": float(actual_densities[0]) if len(actual_densities) > 0 else float(initial_density),
            "predicted_density": float(X)
        })
        
        for d in range(1, num_days + 1):
            X = X * daily_growth_factor
            actual_results.append({
                "day": d,
                "actual_density": float(actual_densities[d]) if d < len(actual_densities) else None,
                "predicted_density": float(X)
            })

        # Build the response
        response_data = {
            "message": "File submitted successfully!",
            "average_doubling_time": float(avg_td),
            "viability": float(viability) if viability is not None else None,
            "actual_predictions": actual_results
        }
        return jsonify(response_data), 200

    except Exception as e:
        print("Error in upload:", str(e))
        print(traceback.format_exc())
        return f"Error reading or processing file: {e}", 400

@app.route('/submit_observed_data', methods=['POST'])
def submit_observed_data():
    """
    Handle the submission of observed data and recalculate avg_mu and doubling time.
    """
    try:
        print("Received data:", request.json)  # Debug print
        observed_data = request.json.get('observed_data', [])
        if not observed_data:
            return jsonify({"message": "No observed data provided."}), 400
            
        # Filter out entries where actual_density is '-'
        observed_data = [d for d in observed_data if d.get('actual_density') != '-']

        # Convert observed data to DataFrame
        df = pd.DataFrame(observed_data)
        
        # Convert actual_density to numeric
        df['actual_density'] = pd.to_numeric(df['actual_density'], errors='coerce')
        
        # Drop rows with NaN values
        df = df.dropna(subset=['actual_density'])
        
        print("Processed DataFrame:", df.to_dict('records'))  # Debug print

        # Ensure the DataFrame has the required columns and is not empty
        required_cols = {'day', 'actual_density'}
        if not required_cols.issubset(df.columns) or df.empty:
            return jsonify({"message": "Observed data must contain valid 'day' and 'actual_density' columns."}), 400

        # Compute average doubling time and viability
        avg_td, viability, actual_densities, avg_mu = compute_average_doubling_time(df)
        print(f"Calculated values: avg_td={avg_td}, viability={viability}, avg_mu={avg_mu}")  # Debug print
        
        if avg_td is None or avg_mu <= 0:
            return jsonify({
                "message": "Could not compute doubling time (check data). Need at least two valid density measurements.",
                "actual_predictions": []
            })

        # Get days param or maximum day from data + some extra days
        days_from_request = request.json.get('days')
        if days_from_request is not None:
            num_days = int(days_from_request)
        else:
            max_day = df['day'].max() if not df.empty else 0
            num_days = max_day + 3  # Add a few extra days for prediction

        # Calculate daily growth factor
        daily_growth_factor = np.exp(avg_mu)
        print(f"Daily growth factor: {daily_growth_factor}")  # Debug print

        # Find the last valid observation to base predictions on
        valid_days = df[~pd.isna(df['actual_density'])]['day'].tolist()
        if not valid_days:
            return jsonify({"message": "No valid density measurements found."}), 400
            
        last_observed_day = max(valid_days)
        last_density = float(df.loc[df['day'] == last_observed_day, 'actual_density'].iloc[0])
        print(f"Last observed day: {last_observed_day}, Last density: {last_density}")  # Debug print

        # Generate predictions for all days
        actual_results = []
        for d in range(num_days + 1):  # Include day 0
            day_data = {"day": int(d)}
            
            # Get actual density if it exists
            actual_row = df[df['day'] == d]
            if not actual_row.empty and not pd.isna(actual_row['actual_density'].iloc[0]):
                day_data["actual_density"] = float(actual_row['actual_density'].iloc[0])
            else:
                day_data["actual_density"] = None
            
            # Calculate predicted density
            if d <= last_observed_day and day_data["actual_density"] is not None:
                # Use actual density for days with observations
                day_data["predicted_density"] = day_data["actual_density"]
            elif d <= last_observed_day:
                # For days before last observation but without data
                days_before = last_observed_day - d
                day_data["predicted_density"] = float(last_density / (daily_growth_factor ** days_before))
            else:
                # For future days after last observation
                days_after = d - last_observed_day
                day_data["predicted_density"] = float(last_density * (daily_growth_factor ** days_after))
            
            actual_results.append(day_data)
        
        print(f"Generated {len(actual_results)} predictions")  # Debug print

        # Build the response
        response_data = {
            "message": "Observed data processed successfully!",
            "average_doubling_time": float(avg_td),
            "viability": float(viability) if viability is not None else None,
            "actual_predictions": actual_results
        }
        return jsonify(response_data), 200

    except Exception as e:
        print("Error in submit_observed_data:", str(e))
        print(traceback.format_exc())  # Print full traceback
        return jsonify({"message": f"Error processing observed data: {str(e)}"}), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
