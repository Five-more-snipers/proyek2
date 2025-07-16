import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# --- 1. Initialize App and Load Model ---
app = Flask(__name__)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', '1752031553')

try:
    model = tf.keras.models.load_model(MODEL_DIR)
    print(f"Model loaded successfully from: {MODEL_DIR}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Define Features and Helper Function ---
NUMERIC_FEATURES = [
    'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'sleep_hours', 'mental_health_rating', 'exercise_frequency'
]

def transformed_name(key):
    return f"{key}_xf"

# --- 3. Define Main Interactive Route ---
@app.route("/", methods=["GET", "POST"])
def interactive_predict():
    """
    Handle both displaying the form (GET) and processing the prediction (POST).
    """
    prediction_result = None

    if request.method == "POST":
        if not model:
            # You can pass an error message to the template if you want
            return "Model is not loaded, cannot make a prediction.", 500
            
        try:
            # Create a dictionary to hold the input from the form
            input_data = {}
            for key in NUMERIC_FEATURES:
                input_data[key] = float(request.form[key])

            # Preprocess data to match the model's expected input format
            processed_data = {
                transformed_name(key): tf.constant([[value]]) 
                for key, value in input_data.items()
            }
            
            # Make prediction
            prediction = model.predict(processed_data)
            predicted_score = prediction[0][0]
            
            # Format the result to show 2 decimal places
            prediction_result = f"{predicted_score:.2f}"

        except Exception as e:
            # You can also pass this error to the template
            return f"An error occurred: {e}", 400

    # Render the HTML page.
    # If it's a POST request, prediction_result will have a value.
    # If it's a GET request, prediction_result will be None.
    return render_template("index.html", prediction=prediction_result)

# --- 4. Run the App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)