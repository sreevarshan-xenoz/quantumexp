import requests
import json

# Test the enhanced_dataset_preview endpoint with iris_binary
try:
    response = requests.post('http://localhost:8000/enhanced_dataset_preview', 
                           json={'datasetType': 'iris_binary', 'noiseLevel': 0.1, 'sampleSize': 100})
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Response keys: {list(data.keys())}')
        print(f'Dataset info: {list(data.get("dataset_info", {}).keys())}')
        print(f'Plots keys: {list(data.get("plots", {}).keys())}')
        print(f'Data shape: {len(data.get("data", []))} samples')
        print(f'Labels shape: {len(data.get("labels", []))} labels')
        
        # Check if plots have data
        plots = data.get('plots', {})
        for plot_name, plot_data in plots.items():
            print(f'{plot_name}: {len(plot_data)} characters' if plot_data else f'{plot_name}: empty')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Exception: {e}')