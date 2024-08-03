import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

# URL to the model file stored on Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=18IlaJv3-K45Bhi3Dk-8Mx1mtUcTJvGwg"

@st.cache_resource
def load_model():
    try:
        # Download the model file from the URL
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # Raise an error on bad status

        # Save the model to a temporary file
        temp_model_path = "model_temp.h5"
        with open(temp_model_path, "wb") as f:
            f.write(response.content)

        # Load the model from the saved file
        model = tf.keras.models.load_model(temp_model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(image, model, categories):
    if model is None:
        return "Model is not loaded"
    
    image = Image.open(image)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)
    if confidence < 0.5:
        return "Not in system"
    return categories[pred_class]

st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
