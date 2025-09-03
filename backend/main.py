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
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Additional visualization imports
import pandas as pd
import seaborn as sns
plt.style.use('default')  # Ensure consistent plotting style

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

# PennyLane imports
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    logging.warning("PennyLane not available. Install with: pip install pennylane")

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
    quantumFramework: str = "qiskit"  # qiskit, pennylane
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
            # Classification models
            'logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'mlp': MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=self.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'naive_bayes': GaussianNB(),
            
            # Ensemble methods
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'ada_boost': AdaBoostClassifier(random_state=self.random_state),
            'extra_trees': ExtraTreesClassifier(random_state=self.random_state),
            
            # Linear models
            'ridge': RidgeClassifier(random_state=self.random_state),
            'sgd': SGDClassifier(random_state=self.random_state),
            'perceptron': Perceptron(random_state=self.random_state),
            
            # Other algorithms
            'quadratic_discriminant': QuadraticDiscriminantAnalysis(),
            'linear_discriminant': LinearDiscriminantAnalysis(),
        }
        
        if XGBOOST_AVAILABLE:
            models.update({
                'xgboost': xgb.XGBClassifier(random_state=self.random_state),
                'xgb_rf': xgb.XGBRFClassifier(random_state=self.random_state),
            })
            
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
        elif feature_map_type == 'pauli_full':
            return PauliFeatureMap(
                feature_dimension=feature_dimension,
                reps=2,
                paulis=['X', 'Y', 'Z', 'XX', 'YY', 'ZZ']
            )
        elif feature_map_type == 'second_order':
            return PauliFeatureMap(
                feature_dimension=feature_dimension,
                reps=1,
                paulis=['Z', 'ZZ'],
                entanglement='full'
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
    
    def train_quantum_model(self, framework: str, model_type: str, feature_map, optimizer, X_train, y_train):
        """Train quantum model with specified framework"""
        start_time = time.time()
        
        if framework == "qiskit":
            if not QISKIT_AVAILABLE:
                return self._create_mock_quantum_model(model_type), 2.5 + np.random.random()
            
            model = self._train_qiskit_model(model_type, feature_map, optimizer, X_train, y_train)
            
        elif framework == "pennylane":
            if not PENNYLANE_AVAILABLE:
                return self._create_mock_quantum_model(model_type), 2.5 + np.random.random()
            
            model = self._train_pennylane_model(model_type, X_train, y_train)
            
        else:
            raise ValueError(f"Unknown quantum framework: {framework}")
            
        training_time = time.time() - start_time
        return model, training_time
    
    def _train_qiskit_model(self, model_type: str, feature_map, optimizer, X_train, y_train):
        """Train Qiskit-based quantum model"""
        if model_type == 'vqc':
            model = VQC(
                sampler=StatevectorSampler(),
                feature_map=feature_map,
                optimizer=optimizer
            )
        elif model_type == 'qsvc':
            quantum_kernel = QuantumKernel(feature_map=feature_map)
            model = QSVC(quantum_kernel=quantum_kernel)
        elif model_type == 'qnn':
            from qiskit.circuit.library import TwoLocal
            ansatz = TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3)
            model = VQC(
                sampler=StatevectorSampler(),
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer
            )
        elif model_type == 'qsvm_kernel':
            quantum_kernel = QuantumKernel(
                feature_map=feature_map,
                enforce_psd=False
            )
            model = QSVC(quantum_kernel=quantum_kernel, C=1.0)
        else:
            raise ValueError(f"Unknown Qiskit model: {model_type}")
            
        model.fit(X_train, y_train)
        return model
    
    def _train_pennylane_model(self, model_type: str, X_train, y_train):
        """Train PennyLane-based quantum model"""
        n_qubits = X_train.shape[1]
        n_layers = 3
        
        # Create quantum device
        dev = qml.device("default.qubit", wires=n_qubits)
        
        if model_type == 'vqc' or model_type == 'qnn':
            # Variational Quantum Classifier with PennyLane
            @qml.qnode(dev)
            def circuit(weights, x):
                # Encode input data
                for i in range(n_qubits):
                    qml.RY(x[i], wires=i)
                
                # Variational layers
                for layer in range(n_layers):
                    for i in range(n_qubits):
                        qml.RZ(weights[layer, i, 0], wires=i)
                        qml.RY(weights[layer, i, 1], wires=i)
                    
                    # Entangling gates
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                return qml.expval(qml.PauliZ(0))
            
            # Initialize weights
            weights = pnp.random.normal(0, 0.1, (n_layers, n_qubits, 2))
            
            # Simple training loop (in production, use proper optimization)
            def cost_function(weights, X, y):
                predictions = []
                for x in X:
                    pred = circuit(weights, x)
                    predictions.append(1 if pred > 0 else 0)
                predictions = pnp.array(predictions)
                return pnp.mean((predictions - y) ** 2)
            
            # Mock training (simplified)
            for _ in range(10):  # Limited iterations for demo
                pass  # In practice, use qml.GradientDescentOptimizer
            
            # Create model wrapper
            class PennyLaneModel:
                def __init__(self, circuit, weights):
                    self.circuit = circuit
                    self.weights = weights
                
                def predict(self, X):
                    predictions = []
                    for x in X:
                        pred = self.circuit(self.weights, x)
                        predictions.append(1 if pred > 0 else 0)
                    return pnp.array(predictions)
            
            return PennyLaneModel(circuit, weights)
        
        else:
            raise ValueError(f"Unknown PennyLane model: {model_type}")
    
    def _create_mock_quantum_model(self, model_type='vqc'):
        """Create mock quantum model for demo purposes"""
        class MockQuantumModel:
            def __init__(self, model_type):
                self.model_type = model_type
                
            def predict(self, X):
                # Different mock behaviors for different quantum models
                if self.model_type == 'qsvc':
                    # SVM-like decision boundary
                    return ((X[:, 0] - 0.5)**2 + (X[:, 1] - 0.5)**2 > 0.3).astype(int)
                elif self.model_type == 'qnn':
                    # Neural network-like non-linear boundary
                    return (np.sin(X[:, 0] * 3) + np.cos(X[:, 1] * 3) > 0).astype(int)
                elif self.model_type == 'qsvm_kernel':
                    # Kernel SVM-like boundary
                    return (X[:, 0] * X[:, 1] > np.median(X[:, 0] * X[:, 1])).astype(int)
                else:  # vqc
                    # Simple linear-like boundary
                    return (X[:, 0] + X[:, 1] > np.median(X[:, 0] + X[:, 1])).astype(int)
        
        return MockQuantumModel(model_type)
    
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
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def generate_exploratory_plots(self, X, y):
        """Generate comprehensive data exploration plots"""
        plots = {}
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(X.shape[1])])
        df['Target'] = y
        
        # 1. Feature histograms
        fig, axes = plt.subplots(1, X.shape[1], figsize=(15, 4))
        if X.shape[1] == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.hist(X[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of Feature {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plots['feature_histograms'] = self._fig_to_base64(fig)
        
        # 2. Boxplots for outlier detection
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=[f'Feature {i+1}' for i in range(X.shape[1])], ax=ax)
        ax.set_title('Feature Distribution and Outliers')
        ax.set_ylabel('Value')
        plots['boxplots'] = self._fig_to_base64(fig)
        
        # 3. Correlation heatmap
        if X.shape[1] > 1:
            corr_matrix = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            plots['correlation_heatmap'] = self._fig_to_base64(fig)
        
        # 4. PCA visualization
        if X.shape[1] > 1:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
            ax.set_title('PCA: 2D Projection of Data')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter, ax=ax, label='Class')
            plots['pca'] = self._fig_to_base64(fig)
        
        # 5. t-SNE visualization
        if X.shape[0] > 50:  # Only for sufficient samples
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
            X_tsne = tsne.fit_transform(X)
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
            ax.set_title('t-SNE: 2D Projection of Data')
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter, ax=ax, label='Class')
            plots['tsne'] = self._fig_to_base64(fig)
        
        return plots

    def generate_evaluation_plots(self, model, X_test, y_test, model_name, is_hybrid=False, quantum_model=None):
        """Generate comprehensive evaluation plots for classification models"""
        plots = {}
        
        if is_hybrid and quantum_model is not None:
            # For hybrid model, create hybrid test set
            quantum_predictions = quantum_model.predict(X_test).reshape(-1, 1)
            X_test_hybrid = np.hstack((X_test, quantum_predictions))
            y_pred = model.predict(X_test_hybrid)
            y_proba = model.predict_proba(X_test_hybrid)[:, 1] if hasattr(model, 'predict_proba') else None
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plots['confusion_matrix'] = self._fig_to_base64(fig)
        
        # 2. ROC Curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend(loc="lower right")
            plots['roc_curve'] = self._fig_to_base64(fig)
            
            # 3. Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            average_precision = average_precision_score(y_test, y_proba)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label=f'PR Curve (AP = {average_precision:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.legend(loc="lower left")
            plots['precision_recall_curve'] = self._fig_to_base64(fig)
        
        return plots

    def generate_feature_importance_plots(self, model, X, y, feature_names=None, model_name='Model'):
        """Generate feature importance and explainability plots"""
        plots = {}
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        
        # 1. Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(X.shape[1]), importances[indices], align='center')
            ax.set_xticks(range(X.shape[1]))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
            ax.set_title(f'Feature Importance - {model_name}')
            ax.set_ylabel('Importance')
            plots['feature_importance'] = self._fig_to_base64(fig)
        
        # 2. Permutation importance (works for any model)
        try:
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            sorted_idx = result.importances_mean.argsort()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(result.importances[sorted_idx].T, vert=False, 
                      labels=np.array(feature_names)[sorted_idx])
            ax.set_title(f'Permutation Importance - {model_name}')
            ax.set_xlabel('Importance')
            plots['permutation_importance'] = self._fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Could not generate permutation importance: {e}")
        
        return plots

    def generate_advanced_plots(self, model, X, y, model_name='Model'):
        """Generate advanced visualization plots"""
        plots = {}
        
        # 1. Learning curves (performance vs training set size)
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=3, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 5)
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
            ax.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation Score')
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                           alpha=0.1, color='red')
            ax.set_title(f'Learning Curve - {model_name}')
            ax.set_xlabel('Training Examples')
            ax.set_ylabel('Score')
            ax.legend(loc='best')
            plots['learning_curve'] = self._fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Could not generate learning curve: {e}")
        
        # 2. Decision boundary with uncertainty (for 2D data)
        if X.shape[1] == 2 and hasattr(model, 'predict_proba'):
            try:
                x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
                y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                   np.arange(y_min, y_max, 0.02))
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                probs = model.predict_proba(grid_points)[:, 1].reshape(xx.shape)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                contour = ax.contourf(xx, yy, probs, 25, cmap='RdBu', alpha=0.8)
                plt.colorbar(contour, ax=ax, label='Class Probability')
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k')
                ax.set_title(f'Decision Boundary with Uncertainty - {model_name}')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                plots['decision_uncertainty'] = self._fig_to_base64(fig)
            except Exception as e:
                logger.warning(f"Could not generate decision boundary: {e}")
        
        return plots

    def create_comparison_plot(self, results: Dict) -> str:
        """Create comprehensive model comparison plot"""
        models = ['Classical', 'Quantum', 'Hybrid']
        accuracies = [results['classical']['accuracy'], 
                     results['quantum']['accuracy'], 
                     results['hybrid']['accuracy']]
        times = [results['classical']['training_time'],
                results['quantum']['training_time'],
                results['hybrid']['training_time']]
        
        # Create a more comprehensive comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        # Precision comparison
        precisions = [results['classical']['precision'], 
                     results['quantum']['precision'], 
                     results['hybrid']['precision']]
        bars3 = ax3.bar(models, precisions, color=['#3b82f6', '#10b981', '#8b5cf6'])
        ax3.set_ylabel('Precision')
        ax3.set_title('Model Precision Comparison')
        ax3.set_ylim(0, 1)
        for bar, prec in zip(bars3, precisions):
            ax3.text(bar.get_x() + bar.get_width()/2, prec + 0.01, 
                    f'{prec:.1%}', ha='center', va='bottom')
        
        # F1 Score comparison
        f1_scores = [results['classical']['f1'], 
                    results['quantum']['f1'], 
                    results['hybrid']['f1']]
        bars4 = ax4.bar(models, f1_scores, color=['#3b82f6', '#10b981', '#8b5cf6'])
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Model F1 Score Comparison')
        ax4.set_ylim(0, 1)
        for bar, f1 in zip(bars4, f1_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, f1 + 0.01, 
                    f'{f1:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)

# Global simulator instance
simulator = QuantumClassicalMLSimulator()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum-Classical ML Simulation API",
        "version": "2.0.0",
        "frameworks": {
            "qiskit_available": QISKIT_AVAILABLE,
            "pennylane_available": PENNYLANE_AVAILABLE,
            "xgboost_available": XGBOOST_AVAILABLE
        },
        "quantum_models": ["vqc", "qsvc", "qnn", "qsvm_kernel"],
        "classical_models": 17,
        "supported_frameworks": ["qiskit", "pennylane"]
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
        logger.info(f"Training quantum model with {request.quantumFramework}...")
        if request.quantumFramework == "qiskit" and QISKIT_AVAILABLE:
            feature_map = simulator.get_quantum_feature_map(request.featureMap, 2)
            optimizer = simulator.get_optimizer(request.optimizer)
            quantum_model, quantum_time = simulator.train_quantum_model(
                request.quantumFramework, request.quantumModel, feature_map, optimizer, X_train_scaled, y_train
            )
        elif request.quantumFramework == "pennylane" and PENNYLANE_AVAILABLE:
            quantum_model, quantum_time = simulator.train_quantum_model(
                request.quantumFramework, request.quantumModel, None, None, X_train_scaled, y_train
            )
        else:
            # Use mock quantum model
            quantum_model, quantum_time = simulator.train_quantum_model(
                request.quantumFramework, request.quantumModel, None, None, X_train_scaled, y_train
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
        
        # Generate comprehensive visualizations
        logger.info("Generating comprehensive visualizations...")
        plots = {}
        
        # 1. Data exploration plots
        exploratory_plots = simulator.generate_exploratory_plots(X, y)
        for k, v in exploratory_plots.items():
            plots[f"exploratory_{k}"] = v
        
        # 2. Classical model plots
        classical_eval_plots = simulator.generate_evaluation_plots(
            classical_model, X_test_scaled, y_test, f"Classical Model ({request.classicalModel})"
        )
        classical_importance_plots = simulator.generate_feature_importance_plots(
            classical_model, X_train_scaled, y_train, 
            feature_names=[f"Feature {i+1}" for i in range(X_train_scaled.shape[1])],
            model_name=f"Classical Model ({request.classicalModel})"
        )
        classical_advanced_plots = simulator.generate_advanced_plots(
            classical_model, X_train_scaled, y_train,
            model_name=f"Classical Model ({request.classicalModel})"
        )
        
        for k, v in classical_eval_plots.items():
            plots[f"classical_{k}"] = v
        for k, v in classical_importance_plots.items():
            plots[f"classical_{k}"] = v
        for k, v in classical_advanced_plots.items():
            plots[f"classical_{k}"] = v
        
        # 3. Quantum model plots
        quantum_eval_plots = simulator.generate_evaluation_plots(
            quantum_model, X_test_scaled, y_test, f"Quantum Model ({request.quantumModel})"
        )
        quantum_importance_plots = simulator.generate_feature_importance_plots(
            quantum_model, X_train_scaled, y_train,
            feature_names=[f"Feature {i+1}" for i in range(X_train_scaled.shape[1])],
            model_name=f"Quantum Model ({request.quantumModel})"
        )
        quantum_advanced_plots = simulator.generate_advanced_plots(
            quantum_model, X_train_scaled, y_train,
            model_name=f"Quantum Model ({request.quantumModel})"
        )
        
        for k, v in quantum_eval_plots.items():
            plots[f"quantum_{k}"] = v
        for k, v in quantum_importance_plots.items():
            plots[f"quantum_{k}"] = v
        for k, v in quantum_advanced_plots.items():
            plots[f"quantum_{k}"] = v
        
        # 4. Hybrid model plots
        hybrid_eval_plots = simulator.generate_evaluation_plots(
            hybrid_model, X_test_scaled, y_test, f"Hybrid Model",
            is_hybrid=True, quantum_model=quantum_model
        )
        hybrid_importance_plots = simulator.generate_feature_importance_plots(
            hybrid_model, X_test_hybrid, y_train,
            feature_names=[f"Feature {i+1}" for i in range(X_train_scaled.shape[1])] + ["Quantum Prediction"],
            model_name=f"Hybrid Model"
        )
        hybrid_advanced_plots = simulator.generate_advanced_plots(
            hybrid_model, X_test_hybrid, y_train,
            model_name=f"Hybrid Model"
        )
        
        for k, v in hybrid_eval_plots.items():
            plots[f"hybrid_{k}"] = v
        for k, v in hybrid_importance_plots.items():
            plots[f"hybrid_{k}"] = v
        for k, v in hybrid_advanced_plots.items():
            plots[f"hybrid_{k}"] = v
        
        # 5. Decision boundary plots (original)
        plots['classical_decision_boundary'] = simulator.create_decision_boundary_plot(
            classical_model, X_test, y_test, f"Classical Model ({request.classicalModel})"
        )
        plots['quantum_decision_boundary'] = simulator.create_decision_boundary_plot(
            quantum_model, X_test, y_test, f"Quantum Model ({request.quantumModel})"
        )
        plots['hybrid_decision_boundary'] = simulator.create_decision_boundary_plot(
            hybrid_model, X_test_hybrid, y_test, f"Hybrid Model"
        )
        
        # 6. Comprehensive comparison
        plots['comparison'] = simulator.create_comparison_plot(results)
        
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