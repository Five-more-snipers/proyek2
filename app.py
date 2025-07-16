# app.py (Updated)

import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from prometheus_flask_exporter import PrometheusMetrics # <-- Add this import

# --- 1. Initialize App and Add Metrics ---
app = Flask(__name__)
metrics = PrometheusMetrics(app) # <-- Add this line to instrument your app
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', '1752031553')

# (The rest of your code remains the same...)

try:
    model = tf.keras.models.load_model(MODEL_DIR)
    print(f"Model loaded successfully from: {MODEL_DIR}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

NUMERIC_FEATURES = [
    'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'sleep_hours', 'mental_health_rating', 'exercise_frequency'
]

def transformed_name(key):
    return f"{key}_xf"

@app.route("/", methods=["GET", "POST"])
def interactive_predict():
    prediction_result = None
    if request.method == "POST":
        if not model:
            return "Model is not loaded, cannot make a prediction.", 500
        try:
            input_data = {}
            for key in NUMERIC_FEATURES:
                input_data[key] = float(request.form[key])
            processed_data = {
                transformed_name(key): tf.constant([[value]]) 
                for key, value in input_data.items()
            }
            prediction = model.predict(processed_data)
            predicted_score = prediction[0][0]
            prediction_result = f"{predicted_score:.2f}"
        except Exception as e:
            return f"An error occurred: {e}", 400
    return render_template("index.html", prediction=prediction_result)

# The __main__ block is for local testing only. Gunicorn doesn't use it.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)