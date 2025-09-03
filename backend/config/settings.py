"""
Configuration settings for the quantum-classical ML platform.
"""

import os
from typing import Dict, Any, List
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    app_name: str = "Quantum-Classical ML Simulation API"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # Dataset settings
    max_dataset_size: int = Field(default=10000, env="MAX_DATASET_SIZE")
    default_test_size: float = Field(default=0.2, env="DEFAULT_TEST_SIZE")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    
    # Quantum settings
    quantum_backend: str = Field(default="qasm_simulator", env="QUANTUM_BACKEND")
    max_qubits: int = Field(default=10, env="MAX_QUBITS")
    max_shots: int = Field(default=1024, env="MAX_SHOTS")
    
    # Model settings
    max_training_time: int = Field(default=300, env="MAX_TRAINING_TIME")  # seconds
    default_cv_folds: int = Field(default=5, env="DEFAULT_CV_FOLDS")
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="quantum_ml.log", env="LOG_FILE")
    
    # Cache settings
    enable_cache: bool = Field(default=True, env="ENABLE_CACHE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # Security settings
    api_key: str = Field(default="", env="API_KEY")
    rate_limit: int = Field(default=100, env="RATE_LIMIT")  # requests per minute
    
    # Feature flags
    enable_quantum_hardware: bool = Field(default=False, env="ENABLE_QUANTUM_HARDWARE")
    enable_advanced_algorithms: bool = Field(default=True, env="ENABLE_ADVANCED_ALGORITHMS")
    enable_federated_learning: bool = Field(default=True, env="ENABLE_FEDERATED_LEARNING")
    enable_transfer_learning: bool = Field(default=True, env="ENABLE_TRANSFER_LEARNING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class DatasetConfig:
    """Configuration for dataset management."""
    
    SYNTHETIC_DATASETS = {
        'circles': {
            'name': 'Concentric Circles',
            'description': 'Two concentric circles with adjustable noise and separation',
            'parameters': ['n_samples', 'noise', 'factor'],
            'default_params': {'n_samples': 1000, 'noise': 0.1, 'factor': 0.5}
        },
        'moons': {
            'name': 'Interleaving Moons',
            'description': 'Two interleaving half circles with adjustable noise',
            'parameters': ['n_samples', 'noise'],
            'default_params': {'n_samples': 1000, 'noise': 0.1}
        },
        'blobs': {
            'name': 'Gaussian Blobs',
            'description': 'Isotropic Gaussian blobs with adjustable parameters',
            'parameters': ['n_samples', 'centers', 'cluster_std'],
            'default_params': {'n_samples': 1000, 'centers': 3, 'cluster_std': 1.0}
        },
        'spiral': {
            'name': 'Spiral',
            'description': 'Spiral dataset with adjustable noise',
            'parameters': ['n_samples', 'noise'],
            'default_params': {'n_samples': 1000, 'noise': 0.1}
        },
        'xor': {
            'name': 'XOR Problem',
            'description': 'XOR dataset with adjustable noise',
            'parameters': ['n_samples', 'noise'],
            'default_params': {'n_samples': 1000, 'noise': 0.1}
        }
    }
    
    REAL_DATASETS = {
        'iris_binary': {
            'name': 'Iris (Binary)',
            'description': 'Iris dataset converted to binary classification (Setosa vs others)',
            'source': 'sklearn',
            'features': 4,
            'samples': 150
        },
        'wine_binary': {
            'name': 'Wine (Binary)',
            'description': 'Wine dataset converted to binary classification',
            'source': 'sklearn',
            'features': 13,
            'samples': 178
        },
        'breast_cancer': {
            'name': 'Breast Cancer',
            'description': 'Breast Cancer Wisconsin dataset',
            'source': 'sklearn',
            'features': 30,
            'samples': 569
        },
        'digits_binary': {
            'name': 'Digits (Binary)',
            'description': 'Digits dataset converted to binary classification (Even vs Odd)',
            'source': 'sklearn',
            'features': 64,
            'samples': 1797
        }
    }


class ModelConfig:
    """Configuration for model management."""
    
    CLASSICAL_MODELS = {
        'logistic': {
            'name': 'Logistic Regression',
            'description': 'Linear model for binary classification',
            'hyperparameters': ['C', 'penalty', 'solver'],
            'default_params': {'max_iter': 1000, 'random_state': 42}
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees',
            'hyperparameters': ['n_estimators', 'max_depth', 'min_samples_split'],
            'default_params': {'n_estimators': 100, 'random_state': 42}
        },
        'svm': {
            'name': 'Support Vector Machine',
            'description': 'Support Vector Machine with RBF kernel',
            'hyperparameters': ['C', 'gamma', 'kernel'],
            'default_params': {'kernel': 'rbf', 'random_state': 42}
        },
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Gradient boosting framework',
            'hyperparameters': ['n_estimators', 'learning_rate', 'max_depth'],
            'default_params': {'random_state': 42}
        }
    }
    
    QUANTUM_MODELS = {
        'vqc': {
            'name': 'Variational Quantum Classifier',
            'description': 'Quantum classifier using variational circuits',
            'hyperparameters': ['feature_map', 'optimizer', 'max_iter'],
            'default_params': {'max_iter': 100}
        },
        'qsvc': {
            'name': 'Quantum Support Vector Classifier',
            'description': 'SVM with quantum kernel',
            'hyperparameters': ['feature_map', 'C'],
            'default_params': {'C': 1.0}
        }
    }
    
    FEATURE_MAPS = {
        'zz': {
            'name': 'ZZ Feature Map',
            'description': 'Feature map with ZZ interactions',
            'parameters': ['feature_dimension', 'reps']
        },
        'z': {
            'name': 'Z Feature Map',
            'description': 'Feature map with Z rotations',
            'parameters': ['feature_dimension', 'reps']
        },
        'pauli': {
            'name': 'Pauli Feature Map',
            'description': 'Feature map with Pauli rotations',
            'parameters': ['feature_dimension', 'reps', 'paulis']
        }
    }
    
    OPTIMIZERS = {
        'spsa': {
            'name': 'SPSA',
            'description': 'Simultaneous Perturbation Stochastic Approximation',
            'parameters': ['maxiter', 'learning_rate']
        },
        'cobyla': {
            'name': 'COBYLA',
            'description': 'Constrained Optimization BY Linear Approximation',
            'parameters': ['maxiter', 'tol']
        },
        'adam': {
            'name': 'ADAM',
            'description': 'Adaptive Moment Estimation',
            'parameters': ['maxiter', 'lr', 'beta_1', 'beta_2']
        }
    }


# Global settings instance
settings = Settings()

# Configuration dictionaries
DATASET_CONFIG = DatasetConfig()
MODEL_CONFIG = ModelConfig()