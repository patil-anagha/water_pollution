import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open("model_water.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocessing function (if needed)
def preprocess_input(data):
    # Add any preprocessing steps here
    return data

# Prediction function
def predict_potability(data):
    # Preprocess the input
    processed_data = preprocess_input(data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    # Convert numerical predictions to labels
    labels = ["Water Not Potable" if pred == 0 else "Water Potable" for pred in predictions]
    
    return labels


def main():
    # Set the title and sidebar
    st.title("Water Potability Prediction")
    st.sidebar.title("Options")

    # Add inputs for user to enter data
    column_name = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    input_data = {}
    for feature in column_name:
        input_data[feature] = st.sidebar.number_input(f"Enter {feature}")
    input_df = pd.DataFrame([input_data])

    # Display the user input
    st.subheader("User Input:")
    st.write(input_df)

    # Make predictions when the user clicks the "Predict" button
    if st.button("Predict"):
        predictions = predict_potability(input_df)
        st.subheader("Prediction:")
        st.write(predictions)

if __name__ == "__main__":
    main()
