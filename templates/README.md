Breast Cancer Prediction Web Application
This project is a simple web application built using Flask to predict whether breast cancer is benign or malignant based on features extracted from breast cancer cell data. The machine learning model used in this project is trained using a dataset of breast cancer samples and is saved as a .pkl file. The application uses this model to make predictions based on user input.

Technologies Used:
Flask: A lightweight Python web framework for building web applications.
Scikit-learn: A library for machine learning algorithms and tools.
Pickle: For serializing the trained machine learning model.
HTML: For the frontend form to collect input from the user.
Project Structure:
php
Copy
Breast_Cancer_Prediction/
│
├── app.py                  # Main Flask application
├── train_model.py          # Script to train the model
├── breast_cancer_model.pkl # Pickled machine learning model
├── templates/
│   ├── index.html          # Frontend form to take user input
│   └── result.html         # Displays the result of the prediction
├── static/                 # Contains static files like CSS, JS (if any)
│
├── breast_cancer_data.csv  # The dataset (CSV file with features and labels)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
Prerequisites:
Before running the application, ensure that you have the following installed on your local machine:

Python 3.x (preferably version 3.8 or higher)
Virtual Environment (recommended)
Setting up the Environment:
Create a Virtual Environment (if you don't have one):

bash
Copy
python -m venv venv
Activate the Virtual Environment:

On Windows:
bash
Copy
venv\Scripts\activate
On macOS/Linux:
bash
Copy
source venv/bin/activate
Install Dependencies: After activating the virtual environment, install the required packages using the requirements.txt file:

bash
Copy
pip install -r requirements.txt
Alternatively, you can install the dependencies manually by running:

bash
Copy
pip install flask scikit-learn pandas numpy
Getting Started:
Step 1: Train the Model
To begin, you need to train the machine learning model using the train_model.py script. This script will:

Load the breast cancer dataset (CSV file).
Preprocess the data (e.g., handle missing values, normalize/standardize features).
Train a Random Forest Classifier.
Save the trained model to a breast_cancer_model.pkl file using Pickle.
Run the following command in your terminal:

bash
Copy
python train_model.py
This will generate the breast_cancer_model.pkl file, which will be used by the web application for making predictions.

Step 2: Running the Application
Start the Flask application: After training the model, you can run the app.py file to start the Flask web server:

bash
Copy
python app.py
The server will start running on http://127.0.0.1:5000/.

Navigate to the application in your web browser: Open your web browser and go to the URL:

cpp
Copy
http://127.0.0.1:5000/
Input the data:

Enter the values for each feature (e.g., radius mean, texture mean, etc.) in the form.
Click "Submit" to send the data to the server.
View the Prediction:

After submission, the result (either "Benign" or "Malignant") will be displayed on the result page.
Detailed Explanation of Files:
app.py:
This file contains the Flask web application code. It handles:

Routing: The home page (/) renders a form for the user to input data.
Form Submission: When the form is submitted, it processes the data, passes it to the model for prediction, and displays the result.
train_model.py:
This script trains the machine learning model using a breast cancer dataset (breast_cancer_data.csv). It:

Loads the dataset.
Preprocesses it (removes unnecessary columns, splits the data into features and target labels).
Trains a Random Forest Classifier model.
Saves the trained model as breast_cancer_model.pkl.
breast_cancer_model.pkl:
This is the serialized machine learning model file that stores the trained Random Forest model. It is used by app.py to make predictions on new data.

index.html:
This HTML file contains the form that collects the features from the user (like radius, texture, area, etc.) and sends the data to the Flask backend.

result.html:
This HTML file displays the result after the prediction is made. It shows whether the breast cancer is classified as "Benign" or "Malignant."

Model Description:
The machine learning model used for predicting breast cancer is a Random Forest Classifier, which is an ensemble learning method. It is used to classify breast cancer into two categories:

Benign (Non-cancerous)
Malignant (Cancerous)
The model was trained using a dataset containing several features related to the cell characteristics of breast cancer samples. The features include radius, texture, smoothness, area, and others.

How to Contribute:
Fork the repository.
Create a new branch for your changes.
Make your changes and test them thoroughly.
Submit a pull request describing your changes.
Common Issues and Troubleshooting:
KeyError or ValueError in train_model.py:

Ensure the dataset is properly loaded and no features are missing or incorrectly named.
_pickle.UnpicklingError in app.py:

Ensure that the breast_cancer_model.pkl file exists and is in the correct directory.
Check that the model was trained and saved correctly with train_model.py.
Model expects 32 features but only 4 are provided:

Make sure that the user input form (index.html) contains all 32 features required by the model.
400 Bad Request Error:

Check if all input fields in the form are filled out correctly. Missing or invalid input values can lead to this error.
License:
This project is licensed under the MIT License - see the LICENSE file for details.

radius_se: 2.4
texture_se: 1.3
perimeter_se: 20.1
area_se: 140.3
smoothness_se: 0.03
compactness_se: 0.05
concavity_se: 0.06
concave_points_se: 0.02
symmetry_se: 0.08
fractal_dimension_se: 0.05
radius_worst: 22.5
texture_worst: 20.0
perimeter_worst: 145.6
area_worst: 1240.1
smoothness_worst: 0.08
compactness_worst: 0.14
concavity_worst: 0.18
concave_points_worst: 0.12
symmetry_worst: 0.26
fractal_dimension_worst: 0.08
