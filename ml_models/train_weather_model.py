import pandas as pd
import numpy as np
import joblib

print("STARTING SCRIPT 🚀")

# ---------------- CREATE DATA ----------------
data = []

for _ in range(3000):
    temp = np.random.uniform(5, 45)        # Temperature (°C)
    humidity = np.random.uniform(10, 100)  # %
    rainfall = np.random.uniform(0, 300)   # mm
    wind = np.random.uniform(0, 50)        # m/s

    # ---------------- RULE-BASED LABELS ----------------
    heat_stress = 1 if temp > 35 else 0
    low_moisture = 1 if humidity < 30 and rainfall < 20 else 0
    flood_risk = 1 if rainfall > 200 else 0
    fungal_risk = 1 if humidity > 80 and temp > 20 else 0
    cold_stress = 1 if temp < 10 else 0

    data.append([
        temp, humidity, rainfall, wind,
        heat_stress, low_moisture, flood_risk, fungal_risk, cold_stress
    ])

# ---------------- DATAFRAME ----------------
df = pd.DataFrame(data, columns=[
    "temp", "humidity", "rainfall", "wind",
    "heat_stress", "low_moisture", "flood_risk", "fungal_risk", "cold_stress"
])

print("Dataset created ✅")

# ---------------- SPLIT DATA ----------------
from sklearn.model_selection import train_test_split

X = df[["temp", "humidity", "rainfall", "wind"]]
y = df[["heat_stress", "low_moisture", "flood_risk", "fungal_risk", "cold_stress"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Data split done ✅")

# ---------------- TRAIN MODEL ----------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model training completed ✅")

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "risk_model.pkl")

print("Model saved as risk_model.pkl ✅")

# ---------------- TEST ACCURACY ----------------
from sklearn.metrics import accuracy_score

pred = model.predict(X_test)

# Calculate accuracy for each label
accuracies = []
for i in range(y_test.shape[1]):
    acc = accuracy_score(y_test.iloc[:, i], pred[:, i])
    accuracies.append(acc)

print("Individual Accuracies:", accuracies)
print("Average Accuracy:", sum(accuracies) / len(accuracies))

print("FINISHED ✅")