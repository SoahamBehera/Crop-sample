"""
Create a simple disease detection model for testing purposes
This creates a basic CNN model trained on mock data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("🔨 Creating Disease Detection Model...")

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Disease classes
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Rice___Brown_Spot', 'Rice___Leaf_Blast', 'Rice___Neck_Blast', 'Rice___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
    'Wheat___Brown_rust', 'Wheat___Healthy', 'Wheat___Yellow_rust'
]

# Create a simple CNN model
# Create a Dual-Head CNN model (Classification + Regression)
inputs = layers.Input(shape=(224, 224, 3))

# Shared Feature Extraction Layers
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

flatten = layers.Flatten()(x)
dense_shared = layers.Dense(256, activation='relu')(flatten)
dense_shared = layers.Dropout(0.5)(dense_shared)

# Head 1: Disease Classification (Multi-class)
class_branch = layers.Dense(128, activation='relu')(dense_shared)
class_branch = layers.Dropout(0.5)(class_branch)
class_output = layers.Dense(len(DISEASE_CLASSES), activation='softmax', name='disease_output')(class_branch)

# Head 2: Severity/Affected Area Regression (Single value)
reg_branch = layers.Dense(64, activation='relu')(dense_shared)
reg_branch = layers.Dropout(0.3)(reg_branch)
# Output scaled 0-1 (representing percentage), or linear
reg_output = layers.Dense(1, activation='linear', name='severity_output')(reg_branch)

# Define Model
model = keras.Model(inputs=inputs, outputs=[class_output, reg_output])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={'disease_output': 'categorical_crossentropy', 'severity_output': 'mse'},
    metrics={'disease_output': 'accuracy', 'severity_output': 'mae'}
)

print(f"✅ Model created with {len(DISEASE_CLASSES)} classes")
print("\nModel Architecture:")
model.summary()

# Train on dummy data to initialize weights
print("\n🔄 Training model on dummy data for initialization...")
X_train = np.random.randn(100, 224, 224, 3).astype('float32') / 255.0
y_class = keras.utils.to_categorical(np.random.randint(0, len(DISEASE_CLASSES), 100), len(DISEASE_CLASSES))
y_sev = np.random.uniform(0, 100, 100)  # Severity 0-100

model.fit(
    X_train, {'disease_output': y_class, 'severity_output': y_sev},
    epochs=2,
    batch_size=10,
    verbose=1,
    validation_split=0.2
)

# Save the model
model_path = 'models/plant_disease_model.h5'
model.save(model_path)
print(f"\n✅ Disease detection model saved to: {model_path}")

# Also save class names for reference
import json
class_info = {
    'num_classes': len(DISEASE_CLASSES),
    'classes': DISEASE_CLASSES
}
with open('models/disease_classes.json', 'w') as f:
    json.dump(class_info, f, indent=2)

print("✅ Disease classes info saved to: models/disease_classes.json")
