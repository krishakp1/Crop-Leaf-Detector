import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

def format_class(class_name):
    if "___" in class_name:
        crop, disease = class_name.split("___")
        disease = disease.replace("_", " ")
    else:
        crop, disease = class_name, "Unknown"
    return crop, disease

# Load model
model = load_model("crop_model.h5")

# Set this to match your class folders inside PlantVillage/
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___healthy', 'Potato___Late_blight', 'Tomato__Target_Spot',
               'Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot',
               'Tomato_Early_blight','Tomato_healthy','Tomato_Late_blight',
               'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite'] 

# Disease details (add as needed)
disease_info = {
   'Pepper__bell___Bacterial_spot': {
        'description': 'Caused by Xanthomonas campestris, this disease forms dark, water-soaked spots on leaves and fruit.',
        'treatment': 'Remove infected leaves immediately. Use copper-based bactericides. Avoid overhead irrigation and handle plants carefully to reduce spread. Use certified disease-free seeds.'
    },
    'Pepper__bell___healthy': {
        'description': 'Healthy pepper leaves are bright green, with no visible lesions or spots.',
        'treatment': 'Regularly monitor crop for early symptoms. Maintain proper spacing for airflow. Apply balanced fertilizers and avoid overwatering.'
    },
    'Potato___Early_blight': {
        'description': 'Caused by Alternaria solani, seen as dark concentric ring spots on older leaves.',
        'treatment': 'Use fungicides like mancozeb or chlorothalonil. Rotate with non-host crops (like cereals). Remove plant debris after harvest. Ensure good drainage to reduce humidity.'
    },
    'Potato___Late_blight': {
        'description': 'Caused by Phytophthora infestans, it creates water-soaked lesions that turn brown with white fungal growth under leaves.',
        'treatment': 'Spray systemic fungicides (e.g., metalaxyl). Remove and destroy infected plants. Practice strict crop rotation. Use resistant potato varieties.'
    },
    'Potato___healthy': {
        'description': 'Lush, green foliage without discoloration or spots.',
        'treatment': 'Continue preventive care with proper fertilization and irrigation. Maintain weed control and pest monitoring.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Caused by Xanthomonas spp., with small dark spots on leaves that eventually become necrotic.',
        'treatment': 'Avoid working with wet plants. Apply copper-based bactericides weekly. Use resistant tomato varieties. Sanitize tools and remove crop residues.'
    },
    'Tomato___Early_blight': {
        'description': 'Dark brown spots with concentric rings, mainly on older leaves. Caused by Alternaria solani.',
        'treatment': 'Apply protective fungicides like chlorothalonil. Remove lower infected leaves early. Use disease-free seeds and rotate crops.'
    },
    'Tomato___Late_blight': {
        'description': 'Rapidly expanding, greasy lesions on leaves and fruit, with white mold on undersides. Caused by Phytophthora infestans.',
        'treatment': 'Apply systemic fungicides immediately. Uproot and destroy infected plants. Avoid overhead watering. Choose resistant varieties if available.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Caused by Fulvia fulva, forms yellow spots on top of leaves and olive-green mold underneath.',
        'treatment': 'Use fungicides like mancozeb or copper oxychloride. Improve greenhouse ventilation. Space plants adequately.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Small, circular dark brown spots with grayish centers. Caused by Septoria lycopersici.',
        'treatment': 'Remove infected lower leaves promptly. Apply fungicides regularly. Avoid overhead irrigation and rotate crops.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Mites cause yellow speckling on leaves, eventually turning brown and dropping. Webbing may be visible.',
        'treatment': 'Spray miticides or insecticidal soap. Encourage natural predators like ladybugs. Avoid plant stress by regular watering.'
    },
    'Tomato___Target_Spot': {
        'description': 'Circular spots with light brown centers and dark margins. Caused by Corynespora cassiicola.',
        'treatment': 'Use fungicides such as azoxystrobin. Remove affected leaves. Avoid wet foliage through drip irrigation.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Virus spread by whiteflies; causes yellowing, curling of leaves, and stunted growth.',
        'treatment': 'Control whiteflies using insecticides. Use virus-resistant seed varieties. Remove infected plants early.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Mottled yellow-green leaves, distorted growth, and reduced yield. Virus spread by contact.',
        'treatment': 'Sanitize hands and tools frequently. Remove and destroy infected plants. Use resistant varieties if available.'
    },
    'Tomato___healthy': {
        'description': 'Bright green leaves, upright growth, no visible spots or curling.',
        'treatment': 'Keep checking for pest symptoms regularly. Maintain balanced fertilization and irrigation.'
    },
    # You can add all others as needed...
}

# Predict function
def predict_crop_from_leaf(pil_image):
    image = pil_image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(predictions[0]) * 100, 2)
    return predicted_class, confidence, predictions[0]

# Streamlit UI
st.set_page_config(page_title="Crop Leaf Detector", page_icon="🍃")
st.title("🍃 PlantVillage Crop Leaf Disease Detector")
st.write("Upload a leaf image to detect its crop and possible disease.")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    all_probs = model.predict(img)[0] 
    pred = model.predict(img)
    class_index = np.argmax(pred)
    confidence = np.max(pred)

    # Map class index to label
    predicted_class = class_names[class_index]
    crop, disease = format_class(predicted_class)  # Make sure this function exists

    st.success(f"🌿 Crop: {crop}")
    st.error(f"🦠 Disease/Cause: {disease}")
    st.info(f"📈 Confidence: {confidence * 100:.2f}%")


    # Show disease info
    if predicted_class in disease_info:
        st.markdown("### 📋 Disease Info")
        st.write(f"**Description:** {disease_info[predicted_class]['description']}")
        st.write(f"**Treatment / Advice:** {disease_info[predicted_class]['treatment']}")
    else:
        st.info("No detailed information available for this class.")

    st.write("🔢 Prediction vector length:", len(all_probs))
    
 
    st.bar_chart(all_probs)
    # Plot confidence bar chart
    
    
    # Get top prediction
    predicted_index = np.argmax(all_probs)
  
    confidence = all_probs[predicted_index]

    # Display prediction
    st.success(f"🪴 **Detected Crop**: `{predicted_class}` with `{confidence*100:.2f}%` confidence.")

    # Crop-wise confidence
    crop_conf = {}

    for i, name in enumerate(class_names):
        crop, _ = format_class(name)
        crop_conf[crop] = crop_conf.get(crop, 0) + pred[0][i]

    st.subheader("🌿 Crop-level Confidence Chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(crop_conf.keys(), crop_conf.values(), color='green')
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence per Crop")
    ax.set_xticks(range(len(crop_conf)))
    ax.set_xticklabels(crop_conf.keys(), rotation=45, ha='right')  # ✅ Key fix: rotate and align
    fig.tight_layout()
    st.pyplot(fig)
    
    # Disease-level chart
    cause_conf = {}

    for i, name in enumerate(class_names):
        _, cause = format_class(name)
        cause_conf[cause] = cause_conf.get(cause, 0) + pred[0][i]

    st.subheader("Cause/Disease-level Confidence Chart")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(cause_conf.keys(), cause_conf.values(), color='orange')
    ax2.set_ylabel("Confidence")
    ax2.set_xticklabels(cause_conf.keys(), rotation=45)
    st.pyplot(fig2)
