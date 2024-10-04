import mlflow.pyfunc
import streamlit as st
from model_training import train_all_models, load_data
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

def load_model(model_name):
    # Load the model from the MLflow registry
    model_uri = f"models:/{model_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)

def main():
    st.title("House Price Prediction")

    # Load the dataset and train models
    data = load_data()
    best_model_name = train_all_models(data)

    # Load the best model from MLflow
    model = load_model(best_model_name)

    # User inputs for prediction
    area = st.number_input("Area (in square feet)", min_value=500, max_value=100000, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=5, step=1)
    stories = st.number_input("Stories", min_value=1, max_value=10, step=1)
    mainroad = st.selectbox("Main Road Access", ("Yes", "No"))
    guestroom = st.selectbox("Guestroom", ("Yes", "No"))
    basement = st.selectbox("Basement", ("Yes", "No"))
    hotwaterheating = st.selectbox("Hot Water Heating", ("Yes", "No"))
    airconditioning = st.selectbox("Air Conditioning", ("Yes", "No"))
    parking = st.number_input("Parking Spaces", min_value=0, max_value=4, step=1)
    prefarea = st.selectbox("Preferred Area", ("Yes", "No"))
    furnishingstatus = st.selectbox("Furnishing Status", ("furnished", "semi-furnished", "unfurnished"))

    # Convert categorical inputs to binary values
    mainroad = 1 if mainroad == "Yes" else 0
    guestroom = 1 if guestroom == "Yes" else 0
    basement = 1 if basement == "Yes" else 0
    hotwaterheating = 1 if hotwaterheating == "Yes" else 0
    airconditioning = 1 if airconditioning == "Yes" else 0
    prefarea = 1 if prefarea == "Yes" else 0


    furnishingstatus_encoded = [0, 0]  # [semi-furnished, unfurnished]
    if furnishingstatus == "semi-furnished":
        furnishingstatus_encoded = [1, 0]
    elif furnishingstatus == "unfurnished":
        furnishingstatus_encoded = [0, 1]

    if st.button("Predict"):
        # Prepare input data
        input_data = [[area, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement,
                       hotwaterheating, airconditioning, prefarea] + furnishingstatus_encoded]

        # Predict the price using the model
        prediction = model.predict(input_data)

        # Display the predicted price
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()



