import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile

# URL to the model file stored on Google Drive
MODEL_URL = "https://drive.google.com/uc?export=download&id=1--BsZo7orLcfLT7YP82w-j0qX8Y7GtS5"

@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Download the model file
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # Check for HTTP request errors

        # Save the model file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file.flush()  # Ensure content is written to disk

            # Load the model from the temporary file
            model = tf.keras.models.load_model(temp_file.name)
        
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

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
        return f"Error in prediction: {e}"

st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
