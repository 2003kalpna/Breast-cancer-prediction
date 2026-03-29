# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import pickle

# # Load the dataset
# data = pd.read_csv('breast_cancer_data.csv')

# # Set the target column name to 'diagnosis'
# target_column = 'diagnosis'  # This is the correct column to predict

# # Split the data into features (X) and target (y)
# X = data.drop(columns=[target_column])  # Features (all columns except target)
# y = data[target_column]  # Target column (what you want to predict)

# # Optionally, you can convert the target labels to numeric (if needed)
# # For example, if diagnosis contains 'M' and 'B', convert them to 1 and 0
# y = y.map({'M': 1, 'B': 0})  # M = malignant, B = benign

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize and train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Make predictions and evaluate the model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Model accuracy: {accuracy * 100:.2f}%')

# # Save the trained model using pickle
# with open('breast_cancer_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle

# # Load the dataset
# data = pd.read_csv('breast_cancer_data.csv')

# # Check the column names to ensure we're using the right column for the target
# print("Dataset Columns:", data.columns)

# # Ensure that you are using the correct target column name from the dataset
# # For example, if your target column is "diagnosis" instead of "target", update accordingly
# # Let's assume your target column is called 'diagnosis' and features are other columns
# X = data.drop(columns=["diagnosis"])  # Drop the target column (correct if the column name is 'diagnosis')
# y = data["diagnosis"]  # Replace with the correct target column name if it's different

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Save the trained model using pickle
# with open('breast_cancer_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# print("Model has been saved as 'breast_cancer_model.pkl'")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('breast_cancer_data.csv')

# Convert 'diagnosis' column to numeric
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Drop irrelevant columns (id and Unnamed: 32)
data = data.drop(columns=['id', 'Unnamed: 32'])

# Fill missing values with mean of columns
data.fillna(data.mean(), inplace=True)

# Prepare features (X) and target (y)
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model
joblib.dump(model, 'breast_cancer_model.pkl')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

# Save the Confusion Matrix Image
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Feature Importance Chart
importances = model.feature_importances_
features = X.columns

# Sort Features by Importance
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Breast Cancer Prediction")

# Save Feature Importance Image
plt.savefig("feature_importance.png")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)

# Show Graph (Automatically Open hoga)
plt.title("Confusion Matrix")
plt.show()

