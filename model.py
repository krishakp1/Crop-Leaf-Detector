from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("crop_model.h5")

# Set this to match your class folders inside PlantVillage/
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___healthy', 'Potato___Late_blight', 'Tomato__Target_Spot','Tomato__Tomato_mosaic_virus',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_healthy','Tomato_Late_blight','Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite']  # Edit this list based on your folders

def predict_crop_from_leaf(image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = round(100 * np.max(prediction), 2)
    return f"{class_names[class_index]} ({confidence}%)"
