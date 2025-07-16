import os
import tensorflow as tf
from flask import Flask, request, render_template
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__)
metrics = PrometheusMetrics(app)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', '1752031553')


try:
    # Keras 3 requires using TFSMLayer to load a legacy SavedModel format.
    model = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint='serving_default')
    print(f"TensorFlow SavedModel loaded successfully as a TFSMLayer from: {MODEL_DIR}")
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
                transformed_name(key): tf.constant([[value]], dtype=tf.float32) 
                for key, value in input_data.items()
            }
            

            prediction_dict = model(processed_data)
            
            predicted_tensor = prediction_dict['output_0']
            
            predicted_score = predicted_tensor.numpy()[0][0]
            
            prediction_result = f"{predicted_score:.2f}"

        except Exception as e:
            return f"An error occurred during prediction: {e}", 400

    return render_template("index.html", prediction=prediction_result)

# --- 5. Run the App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)