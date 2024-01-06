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
        img = Image.open(uploaded_file)
        st.image(img, width=100)
        img = np.array(img.resize((28, 28)))
        img = img / 255.0
        img = img.reshape((-1, 28, 28, 1))

        # Model prediction
        prediction = model.predict(img)
        st.write(f"Prediction: {np.argmax(prediction)}")


         # Feedback interface
        user_feedback = st.radio("Was the prediction correct?", ("Correct", "Incorrect"))

        # Store incorrect predictions
        if user_feedback == "Incorrect":
            correct_label = st.selectbox("Select the correct label:", list(range(10)))
            #img.save(f'incorrect_predictions/{digit}_actual_{uploaded_file.name}')

# Run the Streamlit app
if __name__ == '__main__':
    main()
