from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import tensorflow as tf
import os
import traceback
import cv2
from PIL import Image

app = Flask(__name__)
CORS(app)

VOICE_MODEL_PATH = "cnn_pd_voice_model (1).h5"
SPIRAL_MODEL_PATH = "parkinson_disease_detection.h5"

voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH)
spiral_model = tf.keras.models.load_model(SPIRAL_MODEL_PATH)

print(" Models loaded successfully!")
print("Voice model input shape:", voice_model.input_shape)
print("Spiral model input shape:", spiral_model.input_shape)


def preprocess_audio(file_path):
    """Preprocess audio to match training data (for voice model)."""
    target_duration = 2.0
    y_audio, sr = librosa.load(file_path, sr=None)

    target_len = int(sr * target_duration)
    if len(y_audio) < target_len:
        y_audio = np.pad(y_audio, (0, target_len - len(y_audio)), 'constant')
    else:
        y_audio = y_audio[:target_len]

 
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=n_mels, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    expected_time_steps = 32  
    mel_spec_resized = cv2.resize(mel_spec_db, (expected_time_steps, n_mels))

    input_data = mel_spec_resized[np.newaxis, ..., np.newaxis].astype("float32")
    return input_data


@app.route("/predict", methods=["POST"])
def predict_voice():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        print(f"🎵 Received audio file: {file.filename}")

        target_duration = 2.0
        y_audio, sr = librosa.load(file_path, sr=None)
        target_len = int(sr * target_duration)
        if len(y_audio) < target_len:
            y_audio = np.pad(y_audio, (0, target_len - len(y_audio)), 'constant')
        else:
            y_audio = y_audio[:target_len]

        n_mels = 128
        expected_time_steps = 32 
        mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=n_mels, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    
        import cv2
        mel_spec_resized = cv2.resize(mel_spec_db, (expected_time_steps, n_mels))

        input_data = mel_spec_resized[np.newaxis, ..., np.newaxis].astype("float32")  # (1, 128, 32, 1)

        print(" Mel shape:", input_data.shape)

        prediction = voice_model.predict(input_data)
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_class])

        if predicted_class == 0:
            result = "Healthy"
            is_affected = False
        else:
            result = "Parkinson's Affected"
            is_affected = True

        print(f" Voice Prediction: {result} ({confidence*100:.2f}%)")

        return jsonify({
            "status": "success",
            "type": "voice",
            "result": result,
            "confidence": confidence * 100,
            "isAffected": is_affected
        })

    except Exception as e:
        print("Full error trace:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route("/predict_spiral", methods=["POST"])
def predict_spiral():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        print(f"Received spiral drawing: {file.filename}")

        img = Image.open(file_path).convert("L")
        target_size = spiral_model.input_shape[1:3]  
        img = img.resize(target_size)

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        print("Processed spiral image shape:", img_array.shape)

        predictions = spiral_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[0, predicted_class_index])

        label = "Parkinson's Affected" if predicted_class_index == 1 else "Healthy"
        is_affected = bool(predicted_class_index == 1)  # ✅ convert np.bool_ → bool

        print(f"✅ Spiral Prediction: {label} ({confidence * 100:.2f}%)")

        return jsonify({
            "status": "success",
            "type": "spiral",
            "result": label,
            "confidence": confidence * 100,
            "isAffected": is_affected
        })

    except Exception as e:
        print("Full error trace:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Parkinson Detection API running successfully!"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
