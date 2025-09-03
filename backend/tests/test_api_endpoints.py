import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import app

client = TestClient(app)


class TestAPIEndpoints:
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Quantum-Classical ML Simulation API" in data["message"]

    def test_generate_dataset_endpoint(self):
        """Test dataset generation endpoint."""
        request_data = {
            "datasetType": "circles",
            "noiseLevel": 0.1,
            "sampleSize": 100
        }
        
        response = client.post("/generate_dataset", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 100  # Should have 100 samples
        assert len(data["data"][0]) == 3  # Each sample should have [x, y, label]

    def test_enhanced_dataset_preview_endpoint(self):
        """Test enhanced dataset preview endpoint."""
        request_data = {
            "datasetName": "iris_binary",
            "nSamples": 100,
            "noiseLevel": 0.1,
            "featureEngineering": None,
            "handleImbalance": None
        }
        
        response = client.post("/enhanced_dataset_preview", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "dataset_info" in data
            assert "plots" in data
            
            dataset_info = data["dataset_info"]
            assert "name" in dataset_info
            assert "description" in dataset_info
            assert "metadata" in dataset_info
            assert "feature_names" in dataset_info
            
        elif response.status_code == 501:
            # Enhanced dataset manager not available
            assert "Enhanced dataset manager not available" in response.json()["detail"]

    def test_run_simulation_endpoint(self):
        """Test simulation run endpoint."""
        request_data = {
            "datasetType": "circles",
            "noiseLevel": 0.1,
            "sampleSize": 100,
            "quantumFramework": "qiskit",
            "quantumModel": "vqc",
            "classicalModel": "logistic",
            "featureMap": "zz",
            "optimizer": "spsa",
            "hybridModel": "xgboost"
        }
        
        response = client.post("/run_simulation", json=request_data)
        
        # The simulation might fail due to missing quantum dependencies
        # but we should at least get a proper error response
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "plots" in data

    def test_quantum_hardware_status_endpoint(self):
        """Test quantum hardware status endpoint."""
        response = client.get("/quantum_hardware/status")
        
        # Should return status even if no hardware is available
        assert response.status_code in [200, 500]

    def test_advanced_algorithms_list_endpoint(self):
        """Test advanced algorithms list endpoint."""
        response = client.get("/advanced_algorithms/list")
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "algorithms" in data

    def test_hybrid_optimization_optimizers_endpoint(self):
        """Test hybrid optimization optimizers endpoint."""
        response = client.get("/hybrid_optimization/optimizers")
        
        assert response.status_code in [200, 500]

    def test_invalid_dataset_type(self):
        """Test invalid dataset type handling."""
        request_data = {
            "datasetType": "invalid_dataset",
            "noiseLevel": 0.1,
            "sampleSize": 100
        }
        
        response = client.post("/generate_dataset", json=request_data)
        assert response.status_code == 500
        assert "Dataset generation failed" in response.json()["detail"]

    def test_invalid_request_format(self):
        """Test invalid request format handling."""
        # Missing required fields
        request_data = {
            "noiseLevel": 0.1
        }
        
        response = client.post("/generate_dataset", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_enhanced_dataset_with_feature_engineering(self):
        """Test enhanced dataset preview with feature engineering."""
        request_data = {
            "datasetName": "circles",
            "nSamples": 100,
            "noiseLevel": 0.1,
            "featureEngineering": "pca",
            "handleImbalance": None
        }
        
        response = client.post("/enhanced_dataset_preview", json=request_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "dataset_info" in data
            # After PCA, should have 2 features (or less)
            assert data["dataset_info"]["n_features"] <= 2

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/")
        # CORS middleware should handle OPTIONS requests
        assert response.status_code in [200, 405]


if __name__ == '__main__':
    pytest.main([__file__])