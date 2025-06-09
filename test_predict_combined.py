import requests
import os 

# Endpoint for the combined prediction
url = "http://127.0.0.1:5000/predict-combined"

# Path to the solar panel image
image_path = r"C:\Project\Electrical (33).jpg"  

assert os.path.exists(image_path), f"Image file {image_path} does not exist."

# Simulated sensor input: [power, temperature, irradiance]
sensor_data = [310.2, 42.5, 880.0]

# Prepare the form payload
with open(image_path, "rb") as img_file:
    files = {'image': img_file}
    data = {'data': str(sensor_data)}  # Send data as string

    # Send the request
    print("Sending request to /predict-combined...")
    response = requests.post(url, files=files, data=data)

# Display the result
print("Status Code:", response.status_code)

try:
    print("Response JSON:")
    print(response.json())
except Exception:
    print("Raw Response Text:")
    print(response.text)
