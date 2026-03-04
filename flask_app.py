from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import base64, json, os, io, cv2
from PIL import Image

app = Flask(__name__)

IMG_SIZE = 224
MODEL_PATH = "models/best_model.h5"
CONFIDENCE_THRESHOLD = 60
CLASS_NAMES = ["circle", "rectangle", "triangle"]

# Load model once at startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready.")

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    img_bytes = file.read()

    # ── Preprocess for model ──────────────────
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(pil_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ── Inference ────────────────────────────
    prediction = model.predict(img_array)
    probs = prediction[0] * 100
    confidence = float(np.max(probs))
    predicted_class = CLASS_NAMES[int(np.argmax(probs))]

    # ── DIP: Canny Edge Detection ─────────────
    img_cv = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    _, edge_buf = cv2.imencode(".png", edges)
    edge_b64 = base64.b64encode(edge_buf).decode("utf-8")

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": {
            name: round(float(p), 2)
            for name, p in zip(CLASS_NAMES, probs)
        },
        "low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "edge_image": edge_b64,
    })


@app.route("/metrics")
def metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found. Run eval.py first."}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
