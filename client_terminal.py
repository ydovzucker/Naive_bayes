import requests

# 1. Get input from user
sample = {}
while True:
    key = input("Enter feature name (or 'done' to finish): ")
    if key.lower() == 'done':
        break
    value = input(f"Enter value for '{key}': ")
    sample[key] = value

# 2. Send request to FastAPI server
response = requests.post("http://127.0.0.1:8000/predict", json=sample)

# 3. Show result
print("Prediction from server:", response.json()["prediction"])