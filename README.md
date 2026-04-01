# 🩺 Breast Cancer Prediction Web Application

This project is a **Flask-based web application** that predicts whether breast cancer is  
**🟢 Benign (Non-cancerous)** or **🔴 Malignant (Cancerous)** using machine learning.

The model is trained on a dataset of breast cancer features and saved as a `.pkl` file,  
which is used by the application to make real-time predictions.

---

## 🚀 Technologies Used

- 💻 **Flask** – Backend web framework  
- 🧠 **Scikit-learn** – Machine learning model  
- 💾 **Pickle** – Model serialization  
- 📊 **Pandas & NumPy** – Data processing  
- 🎨 **HTML** – Frontend form  

---

## 📂 Project Structure
Breast_Cancer_Prediction/
│
├── app.py
├── train_model.py
├── breast_cancer_model.pkl
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── static/
├── breast_cancer_data.csv
├── requirements.txt
└── README.md

---

## ⚙️ Prerequisites

- Python 3.x (3.8 or higher recommended)  
- Virtual Environment (recommended)  

---

## 🧩 Setup Environment

### 1️⃣ Create Virtual Environment


### 3️⃣ Install Dependencies

---

## 🧠 Step 1: Train the Model


✔ Loads dataset  
✔ Preprocesses data  
✔ Trains Random Forest model  
✔ Saves model as `breast_cancer_model.pkl`  

---

## ▶️ Step 2: Run the Application

🌐 Open in browser:

---

## 🔍 How It Works

1. User enters feature values  
2. Data is sent to Flask backend  
3. Model processes input  
4. Prediction is displayed:

- 🟢 **Benign**
- 🔴 **Malignant**

---

## 📄 File Explanation

### 🔹 app.py
- Handles routing  
- Takes user input  
- Sends data to model  
- Displays prediction  

### 🔹 train_model.py
- Loads dataset  
- Preprocesses data  
- Trains model  
- Saves `.pkl` file  

### 🔹 breast_cancer_model.pkl
- Trained model used for prediction  

### 🔹 index.html
- User input form  

### 🔹 result.html
- Shows prediction result  

---

## 🤖 Model Description

- Algorithm: **Random Forest Classifier**  
- Type: Classification  
- Output:
  - 🟢 Benign  
  - 🔴 Malignant  

---

## ⚠️ Common Issues & Fixes

❌ Model not loading → Check `.pkl` file  
❌ Feature mismatch → Provide all inputs  
❌ 400 Error → Fill all fields correctly  
❌ KeyError → Check dataset columns  

---

## 📌 Sample Input Values
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


---

## 💡 Future Improvements

- 🎨 Improve UI design  
- 🌐 Deploy online  
- 📊 Show model accuracy  
- ✅ Add input validation  

---

## 📜 License

MIT License  

---

## 🙌 Author

**Kalpna Nimesh**  
B.Tech Computer Science Engineering  

---

⭐ If you like this project, give it a star!
