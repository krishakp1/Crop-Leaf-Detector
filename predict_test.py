from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model("crop_model.h5")

# Get class labels in the same order used during training
class_names = sorted(os.listdir("PlantVillage"))

# Prediction function
def predict_crop(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB").resize((64, 64))  # convert to RGB just in case
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        return f"Prediction: {predicted_class} ({confidence}%)"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"
