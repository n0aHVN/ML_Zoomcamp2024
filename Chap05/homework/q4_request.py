import requests
# url = "http://127.0.0.1:8080/predict"
url = "http://localhost:8080/predict"
client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()
print(response)