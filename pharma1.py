import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pkl
import cv2
import tempfile
from ultralytics import YOLO
import cv2   
import os

# Streamlit App Title
st.title("Predictive Maintenance App with Model Training")
st.sidebar.title("Navigation")

# Menu for Navigation
menu = st.sidebar.radio("Choose a section:", ["Train Model", "Manual Input Prediction", "Real-Time Monitoring"])

# Define required features
required_features = [
    'cycle', 'T2_FanInletTemp_\u00b0R', 'T24_LPCOutletTemp_\u00b0R', 'T30_HPCOutletTemp_\u00b0R',
    'T50_LPTOutletTemp_\u00b0R', 'P2_FanInletPressure_psia', 'P15_BypassDuctPressure_psia',
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
                    
                    model = RandomForestRegressor(n_estimators=200, random_state=42)
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
        
elif menu == "Real-Time Monitoring":
    # Load YOLO model
    try:
        model = YOLO("fire.pt")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")

    st.title("Real Time Monitoring App (Video)")

    # Upload a video
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        temp_dir = tempfile.TemporaryDirectory()
        input_path = os.path.join(temp_dir.name, "input_video.mp4")
        output_path = os.path.join(temp_dir.name, "output_video.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_video.read())

        # Display the uploaded video
        st.video(input_path, format="video/mp4", start_time=0)

        try:
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("Failed to open video. Please check the file format.")

            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for output video (H.264)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create the output video writer
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            st.text("Processing video... Please wait.")

            
            skip_frames = 1
            frame_count = 0  

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames that are not multiples of 'skip_frames'
                if frame_count % skip_frames != 0:
                    frame_count += 1
                    continue

                # Perform detection on the frame using YOLO
                results = model(frame)

                # Extract the annotated frame from the results
                annotated_frame = results[0].plot()  # This should be a valid numpy array

                # Ensure the annotated frame is in BGR format and uint8
                if annotated_frame is not None:
                    if annotated_frame.shape[2] == 4:  # If the frame has alpha channel
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGBA2BGR)

                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = np.uint8(annotated_frame)

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                frame_count += 1

            cap.release()
            out.release()

            # Display the processed video
            st.success("Video processed successfully!")
            st.video(output_path, format="video/mp4")
            st.text("Fire is detected..")

        except Exception as e:
            st.error(f"An error occurred while processing the video: {e}")

        finally:
            # Cleanup temporary directory
            temp_dir.cleanup()
