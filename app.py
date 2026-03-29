import joblib
from flask import Flask, request, render_template

# Create Flask app
app = Flask(__name__)

# Load the pre-trained model (Ensure this file exists in your project folder)
model = joblib.load('breast_cancer_model.pkl')  # Make sure the model file is in the same directory or provide the correct path

# Feature names - make sure these match the ones used when you trained the model
feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", 
    "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points",
    "mean_symmetry", "mean_fractal_dimension", "radius_se", "texture_se", 
    "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", 
    "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", 
    "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", 
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", 
    "fractal_dimension_worst"
]

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract feature values from the form (Ensure the form matches these feature names)
            features = [float(request.form.get(feature)) for feature in feature_names]

            # Predict using the model
            prediction = model.predict([features])

            # Interpret the result (1 -> Malignant, 0 -> Benign)
            if prediction == 1:
                result = "Malignant"
            else:
                result = "Benign"
            
            # Return result to the user
            return render_template('index.html', prediction_text=f'The result is: {result}')
        except Exception as e:
            return f"Error: {e}"

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

import joblib

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

# Print the number of features the model expects (for RandomForestClassifier)
print(f'Number of features expected by the model: {model.n_features_in_}')
