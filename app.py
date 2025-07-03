# # Import all the necessary libraries
# import pandas as pd
# import numpy as np
# import joblib
# import pickle
# import streamlit as st

# # Load the model and structure
# # model = joblib.load("Pollution_Model.pkl")
# # model_cols = joblib.load("Model_Columns.pkl")
# with open("Pollution_Model.pkl", "rb") as f:
#     model = joblib.load(f)

# with open("Model_Columns.pkl", "rb") as f:
#     model_cols = joblib.load(f)


# # Let's create an User interface
# st.title("Water Pollutants Predictor")
# st.write("Predict the water pollutants based on Year and Station ID")

# # User inputs
# year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
# station_id = st.text_input("Enter Station ID", value='1')

# # To encode and then predict
# if st.button('Predict'):
#     if not station_id:
#         st.warning('Please enter the station ID')
#     else:
#         # Prepare the input
#         input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
#         input_encoded = pd.get_dummies(input_df, columns=['id'])

#         # Align with model cols
#         for col in model_cols:
#             if col not in input_encoded.columns:
#                 input_encoded[col] = 0
#         input_encoded = input_encoded[model_cols]

#         # Predict
#         predicted_pollutants = model.predict(input_encoded)[0]
#         pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

#         st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
#         predicted_values = {}
#         for p, val in zip(pollutants, predicted_pollutants):
#             st.write(f'{p}:{val:.2f}')
# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import datetime

# Load the model and structure
with open("Pollution_Model.pkl", "rb") as f:
    model = joblib.load(f)

with open("Model_Columns.pkl", "rb") as f:
    model_cols = joblib.load(f)

# Let's create a User Interface
st.title("💧 Water Pollutants Predictor")
st.write("🔍 Predict the water pollutants based on Year and Station ID")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# To encode and then predict
if st.button('Predict'):
    if not station_id:
        st.warning('⚠️ Please enter the station ID')
    elif not station_id.strip().isalnum():
        st.error("❌ Invalid Station ID. Please enter a valid alphanumeric ID.")
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        # Show results
        st.subheader(f"📊 Predicted pollutant levels for Station '{station_id}' in {year_input}:")
        predicted_values = {}
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f"**{p}**: {val:.2f}")
            predicted_values[p] = val

        # Bar Chart Visualization
        fig, ax = plt.subplots()
        ax.bar(pollutants, predicted_pollutants, color='skyblue')
        ax.set_ylabel("Concentration")
        ax.set_title("Predicted Water Pollutant Levels")
        st.pyplot(fig)

        # Download Prediction as CSV
        pred_df = pd.DataFrame({'Pollutant': pollutants, 'Predicted Value': predicted_pollutants})
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Prediction as CSV", data=csv, file_name=f"prediction_{station_id}_{year_input}.csv", mime="text/csv")

        # Log prediction (for developers)
        with open("prediction_log.txt", "a") as log_file:
            log_file.write(f"\n[{datetime.datetime.now()}] Station: {station_id}, Year: {year_input}, Predicted: {predicted_pollutants.tolist()}")
