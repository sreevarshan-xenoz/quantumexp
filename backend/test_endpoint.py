import requests
import json

datasets_to_test = [
    "iris_binary",
    "wine_binary", 
    "breast_cancer",
    "digits_binary",
    "circles",
    "moons",
    "blobs",
    "classification",
    "spiral",
    "xor",
    "gaussian_quantum"
]

try:
    for dataset_name in datasets_to_test:
        print(f"\n🧪 Testing dataset: {dataset_name}")
        
        # Test the enhanced dataset preview endpoint
        url = "http://localhost:8000/enhanced_dataset_preview"
        payload = {
            "datasetName": dataset_name,
            "nSamples": 100,
            "noiseLevel": 0.1
        }
        
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Check if plots are present
                plots = data.get('plots', {})
                if plots:
                    print(f"✅ {dataset_name}: Plots generated successfully")
                    print(f"   Available plots: {list(plots.keys())}")
                else:
                    print(f"❌ {dataset_name}: No plots found in response")
                    print(f"   Response keys: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"❌ {dataset_name}: Invalid JSON response")
                print(f"   Response: {response.text[:200]}...")
        else:
            print(f"❌ {dataset_name}: Endpoint returned error")
            print(f"   Response: {response.text[:200]}...")
        
        print("-" * 50)
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server. Make sure the backend is running on port 8000")
except Exception as e:
    print(f"❌ Error: {e}")