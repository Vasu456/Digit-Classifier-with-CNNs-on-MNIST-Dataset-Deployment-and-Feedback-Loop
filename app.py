# app.py
import os
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
        
        # Save incorrect predictions
        if user_feedback == "Incorrect":
            correct_label = st.selectbox("Select the correct label:", list(range(10)))

            # Create the "incorrect_predictions" folder if it doesn't exist
            save_path = 'incorrect_predictions'
            os.makedirs(save_path, exist_ok=True)

            # Save incorrect prediction
            #Image.fromarray((img.squeeze() * 255).astype('uint8')).save(f'{save_path}/{correct_label}_actual_{np.argmax(prediction)}_predicted_{uploaded_file.name}')
            Image.fromarray((img.squeeze() * 255).astype('uint8')).save(f'{save_path}/{correct_label}_actual_{correct_label}_predicted_{np.argmax(prediction)}_{uploaded_file.name}')
            st.write("Incorrect prediction saved!")

# Run the Streamlit app
if __name__ == '__main__':
    main()
