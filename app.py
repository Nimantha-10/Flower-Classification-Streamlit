import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO

# Function to download file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(f"{URL}&id={file_id}", stream=True)
    
    # Get the confirmation token if present
    confirm_token = get_confirm_token(response)
    if confirm_token:
        response = session.get(f"{URL}&confirm={confirm_token}&id={file_id}", stream=True)
    
    # Save the file
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('confirm'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download the model from Google Drive
file_id = '1--BsZo7orLcfLT7YP82w-j0qX8Y7GtS5'
output = 'flower_model.h5'
download_file_from_google_drive(file_id, output)

# Check if file exists and is not empty
import os
if not os.path.exists(output) or os.path.getsize(output) == 0:
    st.error("Model file not downloaded correctly or is empty.")
    st.stop()

# Load the model
try:
    model = tf.keras.models.load_model(output)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define flower classes (replace with your actual class names)
class_names = ['rose', 'tulip', 'sunflower', 'daisy', 'dandelion']

# Define a function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to the input size your model expects
    img = np.array(img) / 255.0     # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define a function to predict the flower type and similar images
def predict_flower(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = class_names[np.argmax(predictions)]
    # For similar images, you need to implement your logic here
    similar_images = [f"URL of similar image {i+1}" for i in range(5)]
    return predicted_class, similar_images

# Streamlit app layout
st.title("Flower Recognition App")
st.write("Upload an image of a flower to get the prediction and similar images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    predicted_class, similar_images = predict_flower(image)
    st.write(f"Predicted Flower: {predicted_class}")
    st.write("Similar Images:")
    for img_url in similar_images:
        st.image(img_url, width=150)  # Replace with the actual way you get images
