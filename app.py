import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained models
mlp_model = joblib.load("model/mlp_model.pkl")  # Ensure you save your trained model
scaler = joblib.load("model/scaler.pkl")

# Sample dataset (this is just an example, replace it with your actual dataset)
sample_data = {
    "GRE_Score": [320, 310, 300, 330],
    "TOEFL_Score": [110, 105, 100, 115],
    "University_Rating": [4, 3, 2, 5],
    "SOP": [4.5, 4.0, 3.0, 5.0],
    "LOR": [4.5, 3.5, 2.5, 5.0],
    "CGPA": [9.65, 8.87, 8.00, 8.67],
    "Research": [1, 1, 0, 1]
}
sample_df = pd.DataFrame(sample_data)

# Streamlit title and instructions
st.title("Admission Prediction App")
#st.write("Choose a model and enter the details below to predict admission chances.")

# Model selection dropdown
#model_choice = st.selectbox("Choose a model", ["MLP Model", "Scaler"])
model_choice = "MLP_Model"

# Display a sample of the dataset
st.write("Sample dataset:")
st.dataframe(sample_df)

# Input fields
gre = st.number_input("GRE Score (0 to 340)", min_value=0, max_value=340, step=1)
toefl = st.number_input("TOEFL Score (0 to 120)", min_value=0, max_value=120, step=1)
university_rating = st.number_input("University Rating (1 to 5)", min_value=1, max_value=5, step=1)
sop = st.number_input("Statement of Purpose (SOP) Score (0 to 5)", min_value=0.0, max_value=5.0, step=0.1)
lor = st.number_input("Letter of Recommendation (LOR) Score (0 to 5)", min_value=0.0, max_value=5.0, step=0.1)
cgpa = st.number_input("CGPA (0 to 10)", min_value=0.0, max_value=10.0, step=0.1)
research = st.selectbox("Research Experience", options=["Yes", "No"])

# Convert 'Research' to numerical values (0 for No, 1 for Yes)
research_numeric = 1 if research == "Yes" else 0

# Prepare input data (ensure columns match the order of the training data)
input_data = pd.DataFrame([[gre, toefl, university_rating, sop, lor, cgpa, research_numeric]],
                          columns=["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"])

# Reorder columns to ensure they match the training data column order
input_data = input_data[["GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]]

# Handle model choice and make prediction
# Handle model choice and make prediction
if st.button("Predict"):
    # Add dummy Serial_No column to match the scaler's expected input
    input_data["Serial_No"] = 0  # Or any placeholder value

    # Reorder columns to match the scaler's original training order
    expected_columns = ["Serial_No", "GRE_Score", "TOEFL_Score", "University_Rating", "SOP", "LOR", "CGPA", "Research"]
    input_data = input_data[expected_columns]
    # Scale the input and use the model for prediction
    input_scaled = scaler.transform(input_data)
    prediction = mlp_model.predict(input_scaled)
    st.write(f"Predicted Admission Chance: {prediction[0]:.2f}")

   
