import os
import tensorflow as tf
from flask import Flask, request, render_template
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', '1752031553')

try:
    loaded_model = tf.saved_model.load(MODEL_DIR)
    
    predictor = loaded_model.signatures['serving_default']
    
    print(f"TensorFlow SavedModel's 'serving_default' signature loaded successfully.")

except Exception as e:
    print(f"Error loading model or signature: {e}")
    predictor = None

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
        if not predictor:
            return "Model predictor is not loaded, cannot make a prediction.", 500
            
        try:
            input_data = {}
            for key in NUMERIC_FEATURES:
                input_data[key] = float(request.form[key])

            processed_data = {
                transformed_name(key): tf.constant([[value]], dtype=tf.float32) 
                for key, value in input_data.items()
            }
            
            prediction_dict = predictor(**processed_data)
            
            predicted_tensor = prediction_dict['dense_3']
            
            predicted_score = predicted_tensor.numpy()[0][0]
            
            prediction_result = f"{predicted_score:.2f}"

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return f"An error occurred during prediction: {e}", 400

    return render_template("index.html", prediction=prediction_result)

# --- 5. Run the App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)