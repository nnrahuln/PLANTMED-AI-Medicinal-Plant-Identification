from flask import Flask, request, jsonify, render_template
import base64
import io
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_FILE = os.path.join(BASE_DIR, "class_indices.pkl")
WEIGHTS_FILE = os.path.join(BASE_DIR, "model_weights.weights.h5")
EXCEL_FILE = os.path.join(BASE_DIR, "sci -123.xlsx")


def build_model(num_classes):
    """Rebuild the architecture used when the supplied weights were trained."""
    base_model = keras.applications.InceptionV3(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    return keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


with open(CLASS_FILE, "rb") as f:
    class_indices = pickle.load(f)

num_classes = len(class_indices)
idx_to_class = {int(v): k for k, v in class_indices.items()}

model = build_model(num_classes)
model.build((None, 224, 224, 3))
model.load_weights(WEIGHTS_FILE)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# The spreadsheet is the source of plant metadata shown after prediction.
df = pd.read_excel(EXCEL_FILE)


def prepare_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    resized = img.resize((224, 224))
    arr = keras_image.img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    return img, preprocess_input(arr)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": num_classes})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        file_bytes = file.read()
        original, image_array = prepare_image(file_bytes)
        predictions = model.predict(image_array, verbose=0)[0]

        top3_idx = np.argsort(predictions)[::-1][:3]
        top3 = [
            {
                "name": idx_to_class[int(idx)],
                "confidence": round(float(predictions[idx]) * 100, 2),
            }
            for idx in top3_idx
        ]

        predicted_idx = int(top3_idx[0])
        predicted_name = idx_to_class[predicted_idx]
        confidence = round(float(predictions[predicted_idx]) * 100, 2)

        details = df[df["Scientific_name"].astype(str) == predicted_name]
        if details.empty:
            return jsonify({"error": f"Plant details not found for {predicted_name}"}), 404

        d = details.iloc[0]
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=85)

        return jsonify({
            "plant_name": predicted_name,
            "confidence": confidence,
            "top3": top3,
            "kannada_name": str(d.get("Kannada Name", "N/A")),
            "parts_used": str(d.get("Parts_used", "N/A")),
            "uses": str(d.get("Uses", "N/A")),
            "grown_area": str(d.get("Grown_Area", "N/A")),
            "preparation_en": str(d.get("Preparation_method", "N/A")),
            "preparation_kn": str(d.get("Preperation method(Kannada)", "N/A")),
            "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
