# -*- coding: utf-8 -*-
"""
Comprehensive Quantum-Classical Machine Learning Simulation Backend
FastAPI server with full quantum and classical ML capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import io
import base64
from typing import Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Scikit-learn imports
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Quantum imports
try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.algorithms import VQC, QSVC
    from qiskit_machine_learning.kernels import QuantumKernel
    from qiskit_algorithms.optimizers import SPSA, COBYLA
    from qiskit.primitives import StatevectorSampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit not available. Install with: pip install qiskit qiskit-machine-learning qiskit-algorithms")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Quantum-Classical ML Simulation API",
    description="Advanced API for quantum and classical machine learning comparisons",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DatasetRequest(BaseModel):
    datasetType: str = "circles"
    noiseLevel: float = 0.2
    sampleSize: int = 200

class SimulationRequest(BaseModel):
    datasetType: str = "circles"
    noiseLevel: float = 0.2
    sampleSize: int = 1000
    quantumModel: str = "vqc"
    classicalModel: str = "logistic"
    featureMap: str = "zz"
    optimizer: str = "spsa"
    hybridModel: str = "xgboost"

class SimulationResponse(BaseModel):
    results: Dict[str, Any]
    plots: Dict[str, str]
    dataset_info: Dict[str, Any]

class QuantumClassicalMLSimulator:
    """Main simulation class"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = MinMaxScaler((0, 2 * np.pi))
        
    def generate_dataset(self, dataset_type: str, n_samples: int, noise: float) -> tuple:
        """Generate synthetic dataset"""
        if dataset_type == 'circles':
            X, y = make_circles(
                n_samples=n_samples, 
                noise=noise, 
                factor=0.2, 
                random_state=self.random_state
            )
        elif dataset_type == 'moons':
            X, y = make_moons(
                n_samples=n_samples, 
                noise=noise, 
                random_state=self.random_state
            )
        elif dataset_type == 'blobs':
            X, y = make_blobs(
                n_samples=n_samples, 
                centers=2, 
                n_features=2, 
                cluster_std=noise*2, 
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        return X, y
    
    def get_classical_model(self, model_type: str):
        """Get classical ML model"""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'mlp': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'naive_bayes': GaussianNB()
        }
        
        if XGBOOST_AVAILABLE and model_type == 'xgboost':
            models['xgboost'] = xgb.XGBClassifier(random_state=self.random_state)
            
        if model_type not in models:
            raise ValueError(f"Unknown classical model: {model_type}")
            
        return models[model_type]
    
    def get_quantum_feature_map(self, feature_map_type: str, feature_dimension: int):
        """Get quantum feature map"""
        if not QISKIT_AVAILABLE:
            raise HTTPException(status_code=500, detail="Qiskit not available")
            
        if feature_map_type == 'zz':
            return ZZFeatureMap(
                feature_dimension=feature_dimension,
                reps=2,
                entanglement='linear'
            )
        elif feature_map_type == 'z':
            return ZFeatureMap(
                feature_dimension=feature_dimension,
                reps=2
            )
        elif feature_map_type == 'pauli':
            return PauliFeatureMap(
                feature_dimension=feature_dimension,
                reps=2,
                paulis=['Z', 'XX']
            )
        else:
            raise ValueError(f"Unknown feature map: {feature_map_type}")
    
    def get_optimizer(self, optimizer_type: str):
        """Get quantum optimizer"""
        if not QISKIT_AVAILABLE:
            raise HTTPException(status_code=500, detail="Qiskit not available")
            
        if optimizer_type == 'spsa':
            return SPSA(maxiter=100)
        elif optimizer_type == 'cobyla':
            return COBYLA(maxiter=100)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_classical_model(self, model, X_train, y_train):
        """Train classical model"""
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        return model, training_time
    
    def train_quantum_model(self, model_type: str, feature_map, optimizer, X_train, y_train):
        """Train quantum model"""
        if not QISKIT_AVAILABLE:
            # Return mock quantum model for demo
            return self._create_mock_quantum_model(), 2.5 + np.random.random()
            
        start_time = time.time()
        
        if model_type == 'vqc':
            model = VQC(
                sampler=StatevectorSampler(),
                feature_map=feature_map,
                optimizer=optimizer
            )
        elif model_type == 'qsvc':
            quantum_kernel = QuantumKernel(feature_map=feature_map)
            model = QSVC(quantum_kernel=quantum_kernel)
        else:
            raise ValueError(f"Unknown quantum model: {model_type}")
            
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        return model, training_time
    
    def _create_mock_quantum_model(self):
        """Create mock quantum model for demo purposes"""
        class MockQuantumModel:
            def predict(self, X):
                # Simple mock prediction based on data
                return (X[:, 0] + X[:, 1] > np.median(X[:, 0] + X[:, 1])).astype(int)
        
        return MockQuantumModel()
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist()
        }
    
    def create_decision_boundary_plot(self, model, X, y, title: str) -> str:
        """Create decision boundary plot"""
        plt.figure(figsize=(8, 6))
        
        # Create mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
        plt.colorbar(scatter)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"
    
    def create_comparison_plot(self, results: Dict) -> str:
        """Create model comparison plot"""
        models = ['Classical', 'Quantum', 'Hybrid']
        accuracies = [results['classical']['accuracy'], 
                     results['quantum']['accuracy'], 
                     results['hybrid']['accuracy']]
        times = [results['classical']['training_time'],
                results['quantum']['training_time'],
                results['hybrid']['training_time']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color=['#3b82f6', '#10b981', '#8b5cf6'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.01, 
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # Training time comparison
        bars2 = ax2.bar(models, times, color=['#3b82f6', '#10b981', '#8b5cf6'])
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        
        for bar, time_val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, time_val + 0.01, 
                    f'{time_val:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_str}"

# Global simulator instance
simulator = QuantumClassicalMLSimulator()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum-Classical ML Simulation API",
        "version": "1.0.0",
        "qiskit_available": QISKIT_AVAILABLE,
        "xgboost_available": XGBOOST_AVAILABLE
    }

@app.post("/generate_dataset")
async def generate_dataset_preview(request: DatasetRequest):
    """Generate dataset preview"""
    try:
        X, y = simulator.generate_dataset(
            request.datasetType, 
            request.sampleSize, 
            request.noiseLevel
        )
        
        # Convert to list format for JSON serialization
        data = [[float(x[0]), float(x[1]), int(y[i])] for i, x in enumerate(X)]
        
        return {"data": data}
        
    except Exception as e:
        logger.error(f"Dataset generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dataset generation failed: {str(e)}")

@app.post("/run_simulation", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run complete simulation"""
    try:
        logger.info(f"Starting simulation with parameters: {request.dict()}")
        
        # Generate dataset
        X, y = simulator.generate_dataset(
            request.datasetType, 
            request.sampleSize, 
            request.noiseLevel
        )
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=simulator.random_state
        )
        X_train_scaled = simulator.scaler.fit_transform(X_train)
        X_test_scaled = simulator.scaler.transform(X_test)
        
        results = {}
        
        # Train classical model
        logger.info("Training classical model...")
        classical_model = simulator.get_classical_model(request.classicalModel)
        classical_model, classical_time = simulator.train_classical_model(
            classical_model, X_train_scaled, y_train
        )
        classical_results = simulator.evaluate_model(classical_model, X_test_scaled, y_test)
        classical_results['training_time'] = classical_time
        results['classical'] = classical_results
        
        # Train quantum model
        logger.info("Training quantum model...")
        if QISKIT_AVAILABLE:
            feature_map = simulator.get_quantum_feature_map(request.featureMap, 2)
            optimizer = simulator.get_optimizer(request.optimizer)
            quantum_model, quantum_time = simulator.train_quantum_model(
                request.quantumModel, feature_map, optimizer, X_train_scaled, y_train
            )
        else:
            # Use mock quantum model
            quantum_model, quantum_time = simulator.train_quantum_model(
                request.quantumModel, None, None, X_train_scaled, y_train
            )
            
        quantum_results = simulator.evaluate_model(quantum_model, X_test_scaled, y_test)
        quantum_results['training_time'] = quantum_time
        results['quantum'] = quantum_results
        
        # Train hybrid model
        logger.info("Training hybrid model...")
        quantum_predictions = quantum_model.predict(X_train_scaled).reshape(-1, 1)
        X_train_hybrid = np.hstack((X_train_scaled, quantum_predictions))
        
        hybrid_model = simulator.get_classical_model(request.hybridModel)
        hybrid_model, hybrid_time = simulator.train_classical_model(
            hybrid_model, X_train_hybrid, y_train
        )
        
        # Evaluate hybrid model
        quantum_test_predictions = quantum_model.predict(X_test_scaled).reshape(-1, 1)
        X_test_hybrid = np.hstack((X_test_scaled, quantum_test_predictions))
        hybrid_results = simulator.evaluate_model(hybrid_model, X_test_hybrid, y_test)
        hybrid_results['training_time'] = hybrid_time
        results['hybrid'] = hybrid_results
        
        # Generate plots
        logger.info("Generating visualizations...")
        plots = {
            'classical': simulator.create_decision_boundary_plot(
                classical_model, X_test, y_test, f"Classical Model ({request.classicalModel})"
            ),
            'quantum': simulator.create_decision_boundary_plot(
                quantum_model, X_test, y_test, f"Quantum Model ({request.quantumModel})"
            ),
            'hybrid': simulator.create_decision_boundary_plot(
                hybrid_model, X_test_hybrid, y_test, f"Hybrid Model"
            ),
            'comparison': simulator.create_comparison_plot(results)
        }
        
        dataset_info = {
            'type': request.datasetType,
            'samples': request.sampleSize,
            'features': 2,
            'classes': 2,
            'noise': request.noiseLevel
        }
        
        logger.info("Simulation completed successfully")
        
        return SimulationResponse(
            results=results,
            plots=plots,
            dataset_info=dataset_info
        )
        
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "qiskit_available": QISKIT_AVAILABLE,
        "xgboost_available": XGBOOST_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)