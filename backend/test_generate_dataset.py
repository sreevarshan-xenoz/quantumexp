import requests
import json

# Test the generate_dataset endpoint with iris_binary
try:
    response = requests.post('http://localhost:8000/generate_dataset', 
                           json={'datasetType': 'iris_binary', 'noiseLevel': 0.1, 'sampleSize': 100})
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Response keys: {list(data.keys())}')
        print(f'Data shape: {len(data.get("data", []))} samples')
        print(f'Labels shape: {len(data.get("labels", []))} labels')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Exception: {e}')