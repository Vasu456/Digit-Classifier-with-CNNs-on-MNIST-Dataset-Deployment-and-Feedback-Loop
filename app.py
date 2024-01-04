# app.py

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('trained_model.h5')

# Streamlit app code
def main():
    st.title('MNIST Digit Classification App')

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.text("Uploaded Image")
        st.image(image, width=100)

        # Preprocess the image for model prediction
        img = np.array(image)
        
        # Convert to grayscale if needed
        img = np.mean(img, axis=-1, keepdims=True) if img.shape[-1] == 3 else img
        
        # Resize image to (28, 28)
        img = cv2.resize(img, (28, 28))

        # Flatten the image
        img = img.flatten()

        # Reshape for model prediction
        img = img.reshape((1, 784))  # Assuming input shape (784,) for flattened image
        img = img / 255.0  # Normalize pixel values

        # Make predictions using the loaded model
        prediction = model.predict(img)

        # Display the prediction
        st.write(f'Model Prediction: {np.argmax(prediction)}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
