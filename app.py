import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# URL to the model file stored on Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=18IlaJv3-K45Bhi3Dk-8Mx1mtUcTJvGwg"

@st.cache(allow_output_mutation=True)
def load_model():
    # Fetch the model file from Google Drive
    response = requests.get(MODEL_URL)
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

model = load_model()

# Define the flower categories
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(image, model, categories):
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

# Streamlit UI
st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
