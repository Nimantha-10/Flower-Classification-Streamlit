import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# URL to the model file stored on Dropbox
MODEL_URL = "https://www.dropbox.com/scl/fi/t527snher97bzrw4g1tas/flower_model.h5?rlkey=d3ltlw10hnso9qlfnefrnckdd&st=m11eaw66&dl=1"

@st.cache(allow_output_mutation=True)
def load_model():
    response = requests.get(MODEL_URL)
    model = tf.keras.models.load_model(BytesIO(response.content))
    return model

model = load_model()

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

st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
