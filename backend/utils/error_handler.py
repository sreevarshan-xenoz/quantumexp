"""
Enhanced error handling utilities for the quantum-classical ML platform.
"""

import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import HTTPException
import numpy as np


class QuantumMLError(Exception):
    """Base exception class for quantum ML platform errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "QUANTUM_ML_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class DatasetError(QuantumMLError):
    """Exception raised for dataset-related errors."""
    
    def __init__(self, message: str, dataset_name: str = None, **kwargs):
        super().__init__(message, "DATASET_ERROR", {"dataset_name": dataset_name, **kwargs})


class ModelError(QuantumMLError):
    """Exception raised for model-related errors."""
    
    def __init__(self, message: str, model_type: str = None, **kwargs):
        super().__init__(message, "MODEL_ERROR", {"model_type": model_type, **kwargs})


class QuantumHardwareError(QuantumMLError):
    """Exception raised for quantum hardware-related errors."""
    
    def __init__(self, message: str, provider: str = None, **kwargs):
        super().__init__(message, "QUANTUM_HARDWARE_ERROR", {"provider": provider, **kwargs})


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, logger_name: str = "quantum_ml"):
        self.logger = logging.getLogger(logger_name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def handle_error(self, error: Exception, context: str = None) -> HTTPException:
        """
        Handle errors and convert them to appropriate HTTP exceptions.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            HTTPException: Appropriate HTTP exception for the error
        """
        error_id = self._generate_error_id()
        
        # Log the error with full traceback
        self.logger.error(
            f"Error {error_id} in {context or 'unknown context'}: {str(error)}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        # Handle specific error types
        if isinstance(error, QuantumMLError):
            return self._handle_quantum_ml_error(error, error_id)
        elif isinstance(error, ValueError):
            return HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid input",
                    "message": str(error),
                    "error_id": error_id,
                    "context": context
                }
            )
        elif isinstance(error, ImportError):
            return HTTPException(
                status_code=501,
                detail={
                    "error": "Missing dependency",
                    "message": f"Required package not available: {str(error)}",
                    "error_id": error_id,
                    "context": context
                }
            )
        elif isinstance(error, FileNotFoundError):
            return HTTPException(
                status_code=404,
                detail={
                    "error": "Resource not found",
                    "message": str(error),
                    "error_id": error_id,
                    "context": context
                }
            )
        else:
            # Generic error handling
            return HTTPException(
                status_code=500,
                detail={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "error_id": error_id,
                    "context": context
                }
            )
    
    def _handle_quantum_ml_error(self, error: QuantumMLError, error_id: str) -> HTTPException:
        """Handle specific quantum ML errors."""
        status_codes = {
            "DATASET_ERROR": 400,
            "MODEL_ERROR": 422,
            "QUANTUM_HARDWARE_ERROR": 503
        }
        
        status_code = status_codes.get(error.error_code, 500)
        
        return HTTPException(
            status_code=status_code,
            detail={
                "error": error.error_code,
                "message": error.message,
                "details": error.details,
                "error_id": error_id
            }
        )
    
    def _generate_error_id(self) -> str:
        """Generate a unique error ID for tracking."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def validate_dataset_parameters(self, dataset_name: str, n_samples: int, noise: float):
        """Validate dataset parameters."""
        if n_samples <= 0:
            raise DatasetError("Sample size must be positive", dataset_name=dataset_name)
        
        if n_samples > 10000:
            raise DatasetError("Sample size too large (max 10000)", dataset_name=dataset_name)
        
        if not 0 <= noise <= 1:
            raise DatasetError("Noise level must be between 0 and 1", dataset_name=dataset_name)
    
    def validate_model_parameters(self, model_type: str, parameters: Dict[str, Any]):
        """Validate model parameters."""
        if not model_type:
            raise ModelError("Model type cannot be empty")
        
        # Add specific validation for different model types
        if model_type == "vqc" and "feature_map" not in parameters:
            raise ModelError("VQC requires feature_map parameter", model_type=model_type)
    
    def safe_array_conversion(self, data: Any, context: str = "data conversion") -> np.ndarray:
        """Safely convert data to numpy array with error handling."""
        try:
            if isinstance(data, np.ndarray):
                return data
            
            array = np.array(data)
            
            # Check for invalid values
            if np.any(np.isnan(array)):
                raise ValueError("Data contains NaN values")
            
            if np.any(np.isinf(array)):
                raise ValueError("Data contains infinite values")
            
            return array
            
        except Exception as e:
            raise DatasetError(f"Failed to convert data to array: {str(e)}", context=context)
    
    def log_performance_metrics(self, operation: str, duration: float, **kwargs):
        """Log performance metrics for monitoring."""
        self.logger.info(
            f"Performance: {operation} completed in {duration:.3f}s"
            + (f" - {kwargs}" if kwargs else "")
        )


# Global error handler instance
error_handler = ErrorHandler()


def handle_api_error(context: str = None):
    """Decorator for handling API errors."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise error_handler.handle_error(e, context or func.__name__)
        return wrapper
    return decorator