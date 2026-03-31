import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def train_model():
    data = pd.DataFrame({
        'N': [90, 85, 60, 45, 70, 80, 65, 75, 50, 55,
              88, 77, 66, 59, 92, 73, 61, 84, 69, 58],
        'P': [42, 58, 55, 35, 40, 45, 50, 48, 30, 38,
              41, 44, 52, 36, 47, 49, 33, 46, 39, 31],
        'K': [43, 41, 44, 40, 42, 39, 45, 37, 35, 36,
              43, 40, 46, 38, 44, 41, 34, 42, 39, 32],
        'temperature': [20, 25, 30, 35, 22, 28, 26, 24, 32, 27,
                        23, 29, 31, 34, 21, 26, 33, 22, 30, 28],
        'humidity': [80, 70, 60, 50, 75, 65, 68, 72, 55, 58,
                     82, 67, 63, 52, 85, 69, 57, 74, 61, 59],
        'ph': [6.5, 6.0, 7.0, 6.8, 6.2, 6.7, 6.3, 6.9, 5.8, 6.1,
               6.4, 7.1, 6.6, 6.0, 6.5, 6.3, 5.9, 6.8, 6.2, 6.7],
        'rainfall': [200, 150, 100, 80, 180, 140, 120, 160, 90, 110,
                     210, 130, 95, 85, 220, 170, 75, 190, 105, 115],
        'label': [
            'rice', 'wheat', 'maize', 'cotton', 'barley',
            'sugarcane', 'millet', 'jute', 'groundnut', 'soybean',
            'gram', 'lentil', 'banana', 'apple', 'mango',
            'orange', 'papaya', 'chili', 'tomato', 'potato'
        ]
    })

    X = data[FEATURES]
    y = data['label']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()

    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    model = load_model()

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=FEATURES)

    prediction = model.predict(input_data)
    return prediction[0]