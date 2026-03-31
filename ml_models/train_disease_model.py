import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json

# ✅ ALL CLASSES (matching your remedies)
classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Miner',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',

    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',

    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',

    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',

    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',

    'Peach___Bacterial_spot',
    'Peach___healthy',

    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',

    'Squash___Powdery_mildew',

    'Soybean___healthy',
    'Raspberry___healthy',

    'Orange___Haunglongbing_(Citrus_greening)',

    'Blueberry___healthy'
]

# ✅ Create class index mapping
class_indices = {cls: i for i, cls in enumerate(classes)}

# ✅ Save JSON
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

print("✅ class_indices.json created")

# ✅ Dummy dataset (for testing only)
num_samples = 300
X = np.random.rand(num_samples, 224, 224, 3)
y = np.random.randint(0, len(classes), num_samples)

# ✅ CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])

# ✅ Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Train
print("🚀 Training model...")
model.fit(X, y, epochs=5)

# ✅ Save model
model.save('disease_model.h5')

print("🎉 Model saved as disease_model.h5")