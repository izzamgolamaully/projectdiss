#dashboard.py
import streamlit as st
import requests

st.set_page_config(page_title="Solar Fault Detection", layout="centered")
st.title("Solar Panel Fault Detection Dashboard")

# --- Upload Image ---
st.header("Step 1: Upload Solar Panel Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# --- Enter Sensor Data ---
st.header("Step 2: Enter Sensor Data")
power = st.number_input("Power Output (W)", value=250.0)
temp = st.number_input("Temperature (°C)", value=25.0)
irradiance = st.number_input("Irradiance (W/m²)", value=800.0)

# --- Submit to API ---
if st.button("Run Fault Detection"):
    if uploaded_file is not None:
        files = {'image': uploaded_file}
        data = {'data': str([power, temp, irradiance])}

        with st.spinner("Processing..."):
            try:
                response = requests.post("http://127.0.0.1:5000/predict-combined", files=files, data=data)
            except Exception as e:
                st.error(f"Could not connect to the Flask server: {e}")
                st.stop()

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Received")

            # --- LSTM Results ---
            st.subheader("LSTM (Sensor-Based Fault Detection)")
            st.write(f"**Fault Probability:** {result['lstm_fault_probability']:.2f}")
            st.write(f"**Alert:** {result['lstm_alert']}")

            # --- YOLOv8 Results ---
            st.subheader("YOLOv8 (Image-Based Detection)")
            st.write(f"**Alert:** {result['yolo_alert']}")
            st.write("**Detections:**")
            for det in result['yolo_detections']:
                st.write(f"- {det['label']} ({det['confidence']:.2f})")

            # --- Annotated Image Display ---
            st.subheader("Annotated YOLO Output")
            if 'image_url' in result:
                image_url = f"http://127.0.0.1:5000{result['image_url']}"
                st.write("Image URL:", image_url) #DEBUGGING
                try:
                    image_response = requests.get(image_url)
                    if image_response.status_code == 200:
                        st.image(image_response.content, caption="Detected Defects", use_column_width=True)
                    else:
                        st.warning("Annotated image could not be loaded from the server.")
                except Exception as e:
                    st.error(f"Error retrieving image: {e}")
            else:
                st.warning(" No image returned by the server.")
        else:
            st.error(f" Server returned error: {response.status_code}\n{response.text}")
    else:
        st.warning("Please upload an image before running the prediction.")
