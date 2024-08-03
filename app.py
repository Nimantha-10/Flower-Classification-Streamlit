import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import tempfile

# URL to the model file stored on Dropbox
MODEL_URL = "https://www.dropbox.com/scl/fi/t527snher97bzrw4g1tas/flower_model.h5?rlkey=d3ltlw10hnso9qlfnefrnckdd&st=m11eaw66&dl=1"

@st.cache(allow_output_mutation=True)
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            temp_file.seek(0)
            model = tf.keras.models.load_model(temp_file.name)
            return model
    else:
        st.error("Error loading model from Dropbox.")
        return None

model = load_model()

if model is None:
    st.stop()

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(image, model, categories):
    try:
        # Open the image file
        image = Image.open(image)
        
        # Resize image to the expected input shape (128, 128)
        image = image.resize((128, 128))  # Change this if your model expects a different size
        
        # Convert image to numpy array and normalize
        image = np.array(image) / 255.0
        
        # Check if the image has 3 channels, if not, convert to RGB
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB if necessary
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Perform prediction
        predictions = model.predict(image)
        
        # Get the predicted class
        pred_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Handle low confidence cases
        if confidence < 0.5:
            return "Not in system"
        
        return categories[pred_class]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error during prediction"

st.title("Flower Classification")
st.header("Upload a picture of a flower to identify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
