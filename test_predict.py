import requests

url = "http://127.0.0.1:5000/predict"
sensor_data = [310.5, 45.2, 900.1]  # Example sensor input

response = requests.post(url, json={"data": sensor_data})

print("Status Code:", response.status_code)

try:
    print("Response JSON:", response.json())
except Exception:
    print("Raw Response Text:", response.text)
