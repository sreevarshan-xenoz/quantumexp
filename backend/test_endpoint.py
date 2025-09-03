import requests
import json

try:
    # Test the enhanced dataset preview endpoint
    url = "http://localhost:8000/enhanced_dataset_preview"
    payload = {
        "datasetName": "iris_binary",
        "nSamples": 100,
        "noiseLevel": 0.1
    }
    
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Endpoint is working!")
    else:
        print("❌ Endpoint returned an error")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server. Make sure the backend is running on port 8000")
except Exception as e:
    print(f"❌ Error: {e}")