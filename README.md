# Edunet-Foundation-AICTE-Internship
Water Quality Prediction Project 
Water Quality Prediction System
This project predicts the concentration of water pollutants such as O₂, NO₃, NO₂, SO₄, PO₄, and Cl using machine learning. It uses a trained RandomForestRegressor wrapped in MultiOutputRegressor, and provides a user-friendly Streamlit web interface for real-time predictions based on Year and Station ID.

🚀 Features
✅ Machine Learning model trained using historical pollution data.

✅ Predicts multiple pollutants at once.

✅ Streamlit app for clean and interactive UI.

✅ Input validation to handle user errors.

✅ Bar chart visualization of predicted results.

✅ Download predictions as CSV.

✅ Logging of prediction history.

✅ Modular code structure with reusable components.

🛠 Tech Stack
Python 3.8+

Pandas, NumPy, Scikit-learn

Matplotlib

Streamlit

Joblib

📁 Project Structure
bash
Copy
Edit
├── app.py                      # Streamlit application script
├── Pollution_Model.pkl         # Trained ML model
├── Model_Columns.pkl           # Feature columns used during training (https://drive.google.com/file/d/1Y3XKfaC8IQzAa9zBMV4ZQR1Yz9JXJQzF/view?usp=drive_link)
├── WaterQualityPrediction.ipynb  # Model training notebook
├── prediction_log.txt          #  Prediction logs
└── model_log.txt               #  Dependencies list
🔍 How It Works
Model Training (Jupyter Notebook):

Dataset is cleaned and prepared.

Model is trained using MultiOutputRegressor + RandomForestRegressor.

Model and feature columns are saved using joblib.

Prediction Interface (Streamlit):

User inputs: Year and Station ID.

Data is one-hot encoded and aligned with model schema.

Predictions are displayed with values and bar charts.

Option to download results as CSV.
