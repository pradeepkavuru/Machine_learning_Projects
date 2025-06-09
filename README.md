# 🩺 Health Disease Prediction System

This project is a Machine Learning-powered web application that predicts possible health conditions based on user inputs like demographics, vital signs, and symptoms. It also suggests relevant medications and diet recommendations.

Built using **Python**, **Streamlit**, and **scikit-learn / XGBoost**, this app provides fast, interpretable predictions for educational or preliminary diagnostic use.

---

## 🚀 Features

- ✅ Predicts health condition based on input data
- 💊 Recommends medications for predicted disease
- 🥗 Suggests a nutrition/diet plan
- 📊 Real-time prediction via Streamlit UI
- 📁 Trained on a structured dataset with features like symptoms, age, gender, BP, etc.

---

## 📂 Input Features

- **Demographics**: Age, Gender  
- **Vitals**: Blood Pressure, Cholesterol, Blood Sugar, Heart Rate  
- **Symptoms**: Selected from a predefined list (multi-select)

---

## 🧠 Model Info

- Algorithm: `XGBoostClassifier` (or other model like RandomForest)
- Input encoding includes:
  - One-hot encoded symptoms
  - Label encoded gender, disease, medication, and nutrition
- Model trained on labeled health dataset with 90%+ accuracy

---

## 💡 How It Works

1. User selects their **gender**, inputs **vital signs** and **symptoms**.
2. The data is transformed into a feature vector.
3. The model predicts the **disease**.
4. Based on the prediction, a second model (or logic) suggests:
   - Medications
   - Diet plan
5. Results are shown in the app with interpretation.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn / XGBoost
- Pandas, NumPy
- Pickle / Joblib for model persistence

---
