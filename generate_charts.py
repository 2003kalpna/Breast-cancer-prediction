import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load Trained Model
model = joblib.load('breast_cancer_model.pkl')

# Load Data
data = pd.read_csv('breast_cancer_data.csv')

# Convert 'diagnosis' column to numeric
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Drop unnecessary columns
data = data.drop(columns=['id', 'Unnamed: 32'])

# Features & Target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Predictions
y_pred = model.predict(X)

# 📌  Confusion Matrix Graph
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

#  Feature Importance Chart (Kaunse Features Important Hai)
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance, y=features, palette="viridis")
plt.title("Feature Importance in Breast Cancer Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.savefig("feature_importance.png")
plt.show()

# Prediction Distribution Chart (Cancer vs Non-Cancer Count)
plt.figure(figsize=(5, 5))
sns.countplot(x=y_pred, palette="coolwarm")
plt.title("Predicted Cancer vs Non-Cancer Cases")
plt.xlabel("Prediction (0 = No Cancer, 1 = Cancer)")
plt.ylabel("Count")
plt.savefig("prediction_distribution.png")
plt.show()
