from flask import Flask, render_template, request, jsonify, redirect, session
import os
import sys
import json
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import joblib
from dotenv import load_dotenv
from google import genai

# ---------------- LOAD ENV ----------------
load_dotenv()

API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
gemini_model = "gemini-2.5-flash"

# ---------------- APP INIT ----------------
app = Flask(__name__)
app.secret_key = "krishi_mitra_secret_key"

# ---------------- ML MODELS PATHS ----------------
ML_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'ml_models')

CROP_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'crop_model.pkl')
DISEASE_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'disease_model.h5')
CLASS_PATH = os.path.join(ML_MODELS_DIR, 'class_indices.json')
RISK_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'risk_model.pkl')

# ---------------- CROP MODEL ----------------
sys.path.insert(0, ML_MODELS_DIR)
try:
    from crop_recommendation import recommend_crop, train_model
except ImportError:
    print("Error: crop_recommendation.py not found in ml_models folder")
    sys.exit(1)

if not os.path.exists(CROP_MODEL_PATH):
    print("Crop model not found. Training new model...")
    train_model()

# ---------------- DISEASE MODEL ----------------
if not os.path.exists(DISEASE_MODEL_PATH):
    print(f"Error: {DISEASE_MODEL_PATH} not found. Please provide the disease model.")
    sys.exit(1)

disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

if not os.path.exists(CLASS_PATH):
    print(f"Error: {CLASS_PATH} not found. Please provide class_indices.json.")
    sys.exit(1)

with open(CLASS_PATH, encoding='utf-8') as f:
    idx_to_class = {v: k for k, v in json.load(f).items()}

# ---------------- RISK MODEL ----------------
if not os.path.exists(RISK_MODEL_PATH):
    print(f"Error: {RISK_MODEL_PATH} not found. Please provide risk_model.pkl.")
    sys.exit(1)

risk_model = joblib.load(RISK_MODEL_PATH)

# ---------------- REMEDIES ----------------
REMEDIES = {
    'Tomato___Early_blight': 'Apply copper fungicide.',
    'Tomato___Late_blight': 'Apply Mancozeb spray.',
    'Potato___Early_blight': 'Use chlorothalonil fungicide.',
    'Potato___Late_blight': 'Use Metalaxyl fungicide.',
    'Tomato___healthy': 'Plant is healthy.'
}

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- WEATHER FUNCTION ----------------
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()

    if res.status_code != 200:
        return {"error": data.get("message", "Error fetching weather")}

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind": data["wind"]["speed"],
        "condition": data["weather"][0]["main"]
    }

# ---------------- RISK PREDICTION ----------------
def predict_risks(temp, humidity, rainfall, wind):
    pred = risk_model.predict([[temp, humidity, rainfall, wind]])[0]
    labels = ["heat_stress", "low_moisture", "flood_risk", "fungal_risk", "cold_stress"]
    return [labels[i] for i in range(len(labels)) if pred[i] == 1]

# ---------------- ADVISORY ----------------
def generate_advisory(risks, weather):
    if not risks:
        return "मौसम सामान्य है, फसल की नियमित देखभाल करें।"

    risks_text = ", ".join(risks)
    prompt = f"""
आप एक कृषि विशेषज्ञ हैं।

तापमान: {weather['temperature']}°C
आर्द्रता: {weather['humidity']}%
हवा: {weather['wind']} m/s

जोखिम: {risks_text}

2-3 छोटे सुझाव दें (सरल हिंदी में)
"""
    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )
        output = response.text.strip().replace("\n", "<br>")
        return output
    except:
        return "सलाह प्राप्त करने में समस्या।"

# ---------------- LOGIN ----------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['farmer_id'] == "Krishi" and request.form['password'] == "1234":
            session['user'] = "Krishi"
            return redirect('/home')
        return render_template('login.html', error="Invalid Credentials")
    return render_template('login.html')

# ---------------- HOME ----------------
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')

# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

# ---------------- CROP ----------------
@app.route('/crop-recommendation')
def crop_page():
    return render_template('crop_recommendation.html')

@app.route('/api/recommend-crop', methods=['POST'])
def api_crop():
    data = request.get_json()
    result = recommend_crop(
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    )
    return jsonify({'crop': result})

# ---------------- DISEASE ----------------
@app.route('/disease-detection', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        file = request.files['file']
        os.makedirs('static/uploads', exist_ok=True)
        path = os.path.join('static/uploads', file.filename)
        file.save(path)

        preds = disease_model.predict(preprocess_image(path))[0]
        idx = np.argmax(preds)
        disease = idx_to_class[idx]
        confidence = f"{preds[idx]*100:.1f}%"
        remedy = REMEDIES.get(disease, "Consult expert")

        return render_template('disease_result.html',
                               disease=disease,
                               confidence=confidence,
                               remedy=remedy,
                               image_path=path)
    return render_template('disease.html')

# ---------------- WEATHER ----------------
@app.route('/weather-advisory', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city = request.form.get('city')
        weather = get_weather(city)
        if "error" in weather:
            return render_template('weather.html', error=weather["error"])

        rainfall = 0
        risks = predict_risks(weather["temperature"], weather["humidity"], rainfall, weather["wind"])
        advisory = generate_advisory(risks, weather)

        return render_template('weather.html',
                               weather=weather,
                               risks=risks,
                               advisory=advisory)
    return render_template('weather.html')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)