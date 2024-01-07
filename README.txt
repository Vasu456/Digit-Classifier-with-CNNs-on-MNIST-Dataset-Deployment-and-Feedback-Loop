Digit Classifier with CNNs on MNIST Dataset: Deployment and Feedback Loop

Folder Contents:
1. code_cnn.ipynb
2. app.py
3. retrain.ipynb
4. incorrect_predictions
5. cnn_model.h5
6. Requirements.txt
7. README.txt

app.py: The Streamlit application (app.py) allows users to upload images for digit classification using a pre-trained CNN model. It features a feedback interface to store incorrect predictions for later fine-tuning.

code_cnn.ipynb: The Jupyter notebook (code_cnn.ipynb) outlines the training process for a Convolutional Neural Network (CNN) on the MNIST dataset. It includes data preprocessing, model building, training, and evaluation, saving the model as cnn_model.h5.

retrain.ipynb: The Jupyter notebook (retrain.ipynb) retrieves incorrectly predicted images, extends the training set, fine-tunes the CNN model, and saves the fine-tuned model.

requirements.txt: The requirements.txt file lists necessary Python packages required to run the model and the Streamlit app.

incorrect_predictions: It contains images of predictions that were deemed incorrect and saved due to user feedback. This folder serves as a repository for storing input images, along with their predicted and actual labels, to facilitate the fine-tuning of the model based on user feedback.

Readme.txt: The Readme.txt provides concise instructions for running the entire system, from installing prerequisites to model training, Streamlit deployment, user feedback, and model fine-tuning.


Instructions to run code:

Ensure Python is installed on your system.
Download the MNIST dataset and place it in the "input" folder.

Install required libraries using the command 'pip install -r requirements.txt'.

Model Training (model_cnn.ipynb):
Open model_cnn.ipynb and run the cells sequentially.
This notebook handles data preprocessing, builds and compiles the CNN model, trains it on the MNIST dataset, and saves the trained model as cnn_model.h5.

Streamlit Deployment (app.py):
Open a terminal in the project directory.
Run the Streamlit app using: 'streamlit run app.py'.
Access the displayed link in your web browser. The Streamlit app allows users to upload images, see real-time predictions, and interact with the model.

User Feedback Loop:
Use the Streamlit app to upload images for predictions.
Utilize the feedback interface to indicate the correctness of model predictions.
Incorrect predictions are automatically stored in the "incorrect_predictions" folder.

Model Fine-tuning (retrain.ipynb):
This script retrieves stored incorrect predictions, extends the training set, fine-tunes the model, and gets saved in cnn_model.h5.
An automated job scheduler ensures that the fine-tuning process runs automatically at specified intervals, enhancing the model's accuracy over time.


I have automated the fine-tuning process using a cron job on my laptop. During the initial model deployment, an incorrect prediction was identified where the model mistakenly predicted '0' as '9'. I saved this misclassified image to a incorrect_predictions folder for incorrect predictions. With the scheduled cron job, the retrain.ipynb script ran automatically at the specified intervals, retrieving the stored incorrect prediction, extending the training set, and fine-tuning the model weights accordingly. Consequently, the next time I ran the model, it correctly predicted the digit, showcasing the effectiveness of the automated fine-tuning process in improving model accuracy over time.

Evaluation before retrain: loss: 0.0643 - accuracy: 0.9874
Evaluation after retrain:  loss: 0.0118 - accuracy: 0.9976

