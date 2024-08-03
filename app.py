import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# URL to the model file stored on Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=18IlaJv3-K45Bhi3Dk-8Mx1mtUcTJvGwg"


@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Define the file name and cache directory
        file_name = "flower_model.h5"
        cache_dir = os.path.expanduser("~/.streamlit/")

        # Use tf.keras.utils.get_file to download and cache the model
        model_path = tf.keras.utils.get_file(
            fname=file_name,
            origin=MODEL_URL,
            cache_dir=cache_dir,
            cache_subdir='.'
        )

        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']  # Update with your actual class names

def predict_flower(image, model, categories):
    if model is None:
        return "Model is not loaded"
    try:
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
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Prediction error"

st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
