# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:01:36 2024

@author: balak
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pkl

# Streamlit App Title
st.title("Predictive Maintenance App with Model Training")
st.sidebar.title("Navigation")

# Menu for Navigation
menu = st.sidebar.radio("Choose a section:", ["Train Model", "Manual Input Prediction"])

# Define required features
required_features = [
    'cycle', 'T2_FanInletTemp_°R', 'T24_LPCOutletTemp_°R', 'T30_HPCOutletTemp_°R',
    'T50_LPTOutletTemp_°R', 'P2_FanInletPressure_psia', 'P15_BypassDuctPressure_psia',
    'P30_HPCOutletPressure_psia', 'Nf_FanSpeed_rpm', 'Nc_CoreSpeed_rpm', 
    'epr_EnginePressureRatio', 'Ps30_HPCOutletStaticPressure_psia', 
    'phi_FuelFlowRatio_pps_psi', 'NRf_CorrectedFanSpeed_rpm', 
    'NRc_CorrectedCoreSpeed_rpm', 'BPR_BypassRatio', 'farB_BurnerFuelAirRatio', 
    'htBleed_BleedEnthalpy', 'Nf_dmd_DemandedFanSpeed_rpm', 
    'PCNfR_dmd_DemandedCorrectedFanSpeed_rpm', 'W31_HPTCoolantBleed_lbm_s', 
    'W32_LPTCoolantBleed_lbm_s'
]

# Function to preprocess data
def preprocess_data(data, features):
    scaler = MinMaxScaler()
    X = data[features]
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Train Model Section
# Train Model Section
if menu == "Train Model":
    st.write("Upload your dataset to train a model and predict Remaining Useful Life (RUL) of engines.")
    
    uploaded_file = st.file_uploader("Upload a CSV file for training", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            # Check for missing features
            missing_features = [feature for feature in required_features if feature not in data.columns]
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
            else:
                st.success("Dataset is ready for training.")
                
                # Train the model
                if st.button("Train Model"):
                    data['RUL'] = data.groupby('id')['cycle'].transform('max') - data['cycle']
                    X = data[required_features]
                    y = data['RUL']
                    
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Save the trained model and scaler
                    with open("trained_model.pkl", "wb") as f:
                        pkl.dump((model, scaler), f)
                    
                    st.write("Model Training Completed!")
                    st.write(f"Mean Squared Error: {mse:.2f}")
                    st.write(f"R-squared: {r2:.2f}")
                    st.success("Model saved for prediction.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Manual Input Prediction Section
elif menu == "Manual Input Prediction":
    st.write("Enter the engine parameters below to predict the Remaining Useful Life (RUL).")
    
    # Load the saved model and scaler
    try:
        with open("trained_model.pkl", "rb") as f:
            model, scaler = pkl.load(f)
        
        # Create input fields for each required feature
        input_data = []
        for feature in required_features:
            value = st.number_input(f"Enter {feature}:", value=0.0)
            input_data.append(value)
        
        # Predict RUL
        if st.button("Predict RUL"):
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)  # Scale the input
            prediction = model.predict(input_scaled)
            st.write(f"Predicted Remaining Useful Life (RUL): {prediction[0]:.2f} cycles")
    
    except FileNotFoundError:
        st.error("Model not found. Please train the model in the 'Train Model' section first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
