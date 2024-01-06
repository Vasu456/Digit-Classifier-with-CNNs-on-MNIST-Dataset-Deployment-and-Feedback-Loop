# app.py

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('cnn_model.h5')

# Streamlit app code
def main():
    st.title('MNIST Digit Classification App')

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # # Display the uploaded image
        # image = Image.open(uploaded_file)
        # st.text("Uploaded Image")
        # st.image(image, width=100)

        # # Preprocess the image for model prediction
        # img = np.array(image)
        
        # # Convert to grayscale if needed
        # img = np.mean(img, axis=-1, keepdims=True) if img.shape[-1] == 3 else img
        
        # # Resize image to (28, 28)
        # img = cv2.resize(img, (28, 28))

        # # Flatten the image
        # img = img.flatten()

        # # Reshape for model prediction
        # img = img.reshape((1, 784))  # Assuming input shape (784,) for flattened image
        # img = img / 255.0  # Normalize pixel values

        # # Make predictions using the loaded model
        # prediction = model.predict(img)
        img = Image.open(uploaded_file)
        st.image(img, width=100)
        img = np.array(img.resize((28, 28)))
        img = img / 255.0
        img = img.reshape((-1, 28, 28, 1))

        # Display the uploaded image
        #st.image(img, caption="Uploaded Image", use_column_width=True)

        # Model prediction
        prediction = model.predict(img)
        st.write(f"Prediction: {np.argmax(prediction)}")

        # Display the prediction
        #st.write(f'Model Prediction: {np.argmax(prediction)}')

         # Feedback interface
        user_feedback = st.radio("Was the prediction correct?", ("Correct", "Incorrect"))

    # Store incorrect predictions
        if user_feedback == "Incorrect":
            correct_label = st.selectbox("Select the correct label:", list(range(10)))
            #img.save(f'incorrect_predictions/{digit}_actual_{uploaded_file.name}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
