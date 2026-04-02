from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 Load your trained model
model = load_model("tomatoV2.h5")

# Helper function to preprocess image
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        uploaded_image = request.files["Image"]
        if uploaded_image:
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_image.filename)
            uploaded_image.save(filepath)
            return redirect(url_for("result", filename=uploaded_image.filename))
    return render_template("index.html")

@app.route("/result")
def result():
    filename = request.args.get("filename")
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Prepare image and predict
    img_array = prepare_image(filepath)
    prediction_probs = model.predict(img_array)
    class_names = ['Early Blight', 'Late Blight', 'Healthy']
    predicted_index = np.argmax(prediction_probs)
    predicted_class = class_names[predicted_index]
    confidence = float(prediction_probs[0][predicted_index])

    # Healthy case
    if predicted_class == "Healthy":
        prediction_text = "No disease found"
        return render_template("result.html", filename=filename, prediction=prediction_text, confidence=round(confidence*100,2))

    # Disease case
    prediction_text = f"Disease: {predicted_class}"

    # Treatment and causes
    if predicted_class == "Late Blight":
        t1 = "Remove and destroy infected plants immediately."
        t2 = "Use fungicides like copper sprays, chlorothalonil, or metalaxyl-based products."
        t3 = "Plant resistant or tolerant varieties."
        t4 = "Avoid overhead watering; use drip irrigation instead."
        t5 = "Ensure proper spacing and ventilation to reduce humidity around plants."
        c1 = "Fungal-like pathogen that thrives in cool, wet weather (15–20°C)."
        c2 = "Spores spread via wind, rain, or contaminated equipment."
        c3 = "Overhead irrigation that keeps foliage wet."
        c4 = "Infected seed tubers or transplants."
        c5 = "Poor field drainage or waterlogged soil."
    else:  # Early Blight
        t1 = "Remove infected leaves and plant debris to reduce spores."
        t2 = "Apply fungicides like chlorothalonil, mancozeb, or copper-based sprays."
        t3 = "Use disease-resistant tomato or potato varieties."
        t4 = "Improve air circulation by proper spacing and pruning."
        t5 = "Practice crop rotation (avoid planting tomatoes or potatoes in the same spot every year)."
        c1 = "Fungal spores in soil or plant debris."
        c2 = "Warm temperatures (25–30°C) combined with high humidity."
        c3 = "Poor crop rotation (planting solanaceous crops repeatedly)."
        c4 = "Overhead watering that wets leaves frequently."
        c5 = "Weak or stressed plants due to nutrient deficiencies or poor soil."

    return render_template(
        "result.html",
        filename=filename,
        prediction=prediction_text,
        confidence=round(confidence*100,2),
        t1=t1, t2=t2, t3=t3, t4=t4, t5=t5,
        c1=c1, c2=c2, c3=c3, c4=c4, c5=c5
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
