# # 
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import pickle

# # Load dataset (example, modify according to your dataset)
# data = pd.read_csv('breast_cancer_data.csv')

# # Assume 'label' is the target column, modify based on your data
# X = data.drop('label', axis=1)
# y = data['label']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Save the trained model
# with open('breast_cancer_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# print("Model saved as breast_cancer_model.pkl")

# breast_cancer_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # To save the trained model

# Load the dataset
data = pd.read_csv('breast_cancer_data.csv')  # Replace with your actual dataset file name

# Print first few rows of the dataset (to check)
print(data.head())

# Clean data (if necessary)
# Remove unwanted columns, e.g., 'id' or 'Unnamed: 32'
data = data.drop(columns=['id', 'Unnamed: 32'])

# Convert 'diagnosis' column from M/B to binary (1 for malignant, 0 for benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Define features (X) and target (y)
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'breast_cancer_model.pkl')
print("Model has been trained and saved as 'breast_cancer_model.pkl'")
