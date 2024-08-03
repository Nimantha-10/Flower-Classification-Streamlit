import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Function to load model from BytesIO
def load_model_from_bytes(byte_stream):
    try:
        # Load model directly from BytesIO stream
        return tf.keras.models.load_model(io.BytesIO(byte_stream))
    except Exception as e:
        st.error(f"Error loading model from bytes: {e}")
        return None

st.title("Flower Classification")
st.header("Upload your flower model and a picture of a flower to identify it.")

# Upload model file
model_file = st.file_uploader("Upload a Keras model file...", type=["h5"])

if model_file is not None:
    # Read the file as bytes and load the model
    model_bytes = model_file.read()
    model = load_model_from_bytes(model_bytes)
    if model:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load the model. Please check the file and try again.")
else:
    model = None
    st.info("Please upload a Keras model file to proceed.")

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

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button("Predict"):
        flower_name = predict_flower(uploaded_file, model, categories)
        st.write(f"The predicted flower is: {flower_name}")
