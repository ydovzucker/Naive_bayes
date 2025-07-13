import subprocess
import time
import requests
import pandas as pd
import os
import sys

# === 1. Start the FastAPI server in background ===
print("🚀 Starting FastAPI server...")
python_path = sys.executable

server_process = subprocess.Popen(
    [python_path, "-m", "uvicorn", "api_server:app", "--reload"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# === 2. Wait a few seconds for server to start ===
time.sleep(2)  # adjust if needed

# === 3. Ask user for CSV and target ===
csv_path = input("Enter the path to your CSV file: ")
while not os.path.exists(csv_path):
    csv_path = input("❌ File not found. Try again: ")

# === 4. Load the data and show columns ===
df = pd.read_csv(csv_path)
print("\n📊 Column names in your data:")
print(list(df.columns))

target_column = input("\n🧠 Enter the name of the target (label) column from the list above: ")

# === 5. Convert and extract features ===
data_dict = df.to_dict(orient="records")
features = [col for col in df.columns if col != target_column]

# === 5. Train ===
print("📤 Sending training data to API...")
train_payload = {
    "data": data_dict,
    "target_column": target_column
}
train_response = requests.post("http://127.0.0.1:8000/train", json=train_payload)
print("✅ Train response:", train_response.json())

# === 6. Get user input for prediction ===
print("\n🧠 Enter values for prediction:")

sample = {}
for feature in features:
    valid_options = df[feature].dropna().unique().tolist()
    print(f"\nOptions for '{feature}': {valid_options}")

    while True:
        value = input(f"Enter value for '{feature}': ").strip()
        if value in valid_options:
            sample[feature] = value
            break
        else:
            print(f"❌ Invalid value. Please choose from: {valid_options}")

# === 7. Predict ===
predict_response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print("🧾 Full prediction response:", predict_response.json())
print("\n🔮 Prediction:", predict_response.json()["prediction"])

# === 8. Clean up: stop the server ===
server_process.terminate()
print("\n🛑 FastAPI server terminated.")