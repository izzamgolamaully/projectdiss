import requests
import os

# API endpoint
url = "http://127.0.0.1:5000/detect-image"

# Path to your test image
image_path = r"C:\Project\Bird (41).jpg"  

assert os.path.exists(image_path), f"Image file {image_path} does not exist."

# Send POST request with image file
with open(image_path, "rb") as img_file:
    files = {"image": img_file}
    response = requests.post(url, files=files)

# Print results
print("Status Code:", response.status_code)

try:
    print("Response JSON:", response.json())
except Exception:
    print("Raw Response Text:", response.text)
