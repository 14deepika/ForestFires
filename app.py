# app.py

from flask import Flask, request, jsonify
import numpy as np
import joblib
import logging
from cellular_automata import CellularAutomataFire

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

def parse_numeric(value):
    """
    Convert a value to float. If you want to support strings like 'Medium',
    you can map them to numeric values here. Otherwise, just do float(value).
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Cannot parse numeric value from: {value}")

# Load the trained model
try:
    model = joblib.load('forest_fire_model.pkl')
    app.logger.info("ML model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")

@app.route('/')
def home():
    return "Forest Fire Prediction + CA Simulation API"

@app.route('/predict', methods=['POST'])
def predict_fire_risk():
    """
    Predict ignition risk for a single cell based on input features:
    {
      "X":4, "Y":5, "month":8, "day":15,
      "FFMC":90.0, "DMC":35.0, "DC":100.0,
      "ISI":5.0, "temp":20.0, "RH":40,
      "wind":3.0, "rain":0.0
    }
    """
    feature_cols = ["X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain"]
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data for /predict: {data}")

        # Convert to numeric
        input_list = [parse_numeric(data[col]) for col in feature_cols]
        input_array = np.array(input_list).reshape(1, -1)

        # Model prediction
        prediction = model.predict(input_array)
        prob = model.predict_proba(input_array)[0][1]

        return jsonify({
            "fire_risk_class": int(prediction[0]),
            "fire_risk_prob": float(prob)
        })

    except Exception as e:
        app.logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/simulate', methods=['POST'])
def simulate_ca():
    """
    Runs the CA simulation with given rows, cols, steps, and env_data.
    Expects JSON:
    {
      "rows": 10,
      "cols": 10,
      "steps": 5,
      "env_data": {
        "month": 8, "day": 15, "FFMC":90.0, "DMC":35.0,
        "DC":100.0, "ISI":5.0, "temp":20.0, "RH":40,
        "wind":3.0, "rain":0.0
      }
    }
    Returns final_grid as a 2D list (0=unburned, 1=burning, 2=burned).
    """
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data for /simulate: {data}")

        rows = data.get("rows", 10)
        cols = data.get("cols", 10)
        steps = data.get("steps", 5)
        env_data = data.get("env_data", {})

        # Convert environment data to numeric
        for k, v in env_data.items():
            env_data[k] = parse_numeric(v)

        # Initialize CA
        ca = CellularAutomataFire(rows, cols, model_path='forest_fire_model.pkl')
        # Overwrite default environment data
        ca.env_data.update(env_data)

        # Run simulation
        final_grid = ca.run_simulation(steps=steps)
        return jsonify({"final_grid": final_grid.tolist()})

    except Exception as e:
        app.logger.error(f"Error in /simulate: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # If port 5000 is taken on macOS, either disable AirPlay or choose a different port
    app.run(debug=True, host='0.0.0.0', port=5000)
