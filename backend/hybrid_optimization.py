"""
Quantum-Classical Hybrid Optimization Module
Advanced optimization techniques combining quantum and classical approaches
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optimization imports
try:
    from scipy.optimize import minimize, differential_evolution, basinhopping
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Quantum computing imports
try:
    from qiskit import QuantumCircuit
    from qiskit.algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, SLSQP
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

logger = logging.getLogger(__name__)

class HybridOptimizer(ABC):
    """Abstract base class for hybrid quantum-classical optimizers"""
    
    def __init__(self, quantum_component: Any, classical_component: Any):
        self.quantum_component = quantum_component
        self.classical_component = classical_component
        self.optimization_history = []
        self.best_params = None
        self.best_value = float('inf')
        self.convergence_data = []
    
    @abstractmethod
    def optimize(self, objective_function: Callable, initial_params: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run hybrid optimization"""
        pass
    
    def _log_iteration(self, params: np.ndarray, value: float, iteration: int, method: str):
        """Log optimization iteration"""
        self.optimization_history.append({
            'iteration': iteration,
            'parameters': params.copy(),
            'value': value,
            'method': method,
            'timestamp': time.time()
        })
        
        if value < self.best_value:
            self.best_value = value
            self.best_params = params.copy()

class ParameterShiftOptimizer(HybridOptimizer):
    """
    Parameter Shift Rule Optimizer
    Uses quantum parameter shift rule for gradient estimation
    """
    
    def __init__(self, quantum_circuit: QuantumCircuit, shift_value: float = np.pi/2):
        super().__init__(quantum_circuit, None)
        self.shift_value = shift_value
        self.gradient_history = []
    
    def compute_gradient(self, objective_function: Callable, params: np.ndarray) -> np.ndarray:
        """Compute gradient using parameter shift rule"""
        gradient = np.zeros_like(params)
        
        for i in range(len(params)):
            # Forward shift
            params_plus = params.copy()
            params_plus[i] += self.shift_value
            value_plus = objective_function(params_plus)
            
            # Backward shift
            params_minus = params.copy()
            params_minus[i] -= self.shift_value
            value_minus = objective_function(params_minus)
            
            # Parameter shift rule
            gradient[i] = (value_plus - value_minus) / 2
        
        self.gradient_history.append(gradient.copy())
        return gradient
    
    def optimize(self, objective_function: Callable, initial_params: np.ndarray, 
                max_iter: int = 100, learning_rate: float = 0.01, 
                tolerance: float = 1e-6) -> Dict[str, Any]:
        """Run parameter shift optimization"""
        
        params = initial_params.copy()
        start_time = time.time()
        
        for iteration in range(max_iter):
            # Compute current value
            current_value = objective_function(params)
            self._log_iteration(params, current_value, iteration, 'parameter_shift')
            
            # Compute gradient
            gradient = self.compute_gradient(objective_function, params)
            
            # Update parameters
            params = params - learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(gradient) < tolerance:
                logger.info(f"Parameter shift converged at iteration {iteration}")
                break
        
        execution_time = time.time() - start_time
        
        return {
            'optimizer': 'ParameterShift',
            'optimal_parameters': self.best_params,
            'optimal_value': self.best_value,
            'iterations': len(self.optimization_history),
            'execution_time': execution_time,
            'convergence_data': self.optimization_history,
            'gradient_norms': [np.linalg.norm(g) for g in self.gradient_history],
            'success': True
        }

class MultiObjectiveOptimizer(HybridOptimizer):
    """
    Multi-Objective Quantum-Classical Optimizer
    Optimizes multiple objectives simultaneously using Pareto optimization
    """
    
    def __init__(self, quantum_component: Any, classical_component: Any):
        super().__init__(quantum_component, classical_component)
        self.pareto_front = []
        self.objective_weights = None
    
    def evaluate_objectives(self, params: np.ndarray, objective_functions: List[Callable]) -> np.ndarray:
        """Evaluate multiple objective functions"""
        objectives = np.array([f(params) for f in objective_functions])
        return objectives
    
    def is_pareto_optimal(self, objectives: np.ndarray, pareto_front: List[np.ndarray]) -> bool:
        """Check if solution is Pareto optimal"""
        for front_point in pareto_front:
            if np.all(front_point <= objectives) and np.any(front_point < objectives):
                return False
        return True
    
    def update_pareto_front(self, params: np.ndarray, objectives: np.ndarray):
        """Update Pareto front with new solution"""
        if self.is_pareto_optimal(objectives, [pf['objectives'] for pf in self.pareto_front]):
            # Remove dominated solutions
            self.pareto_front = [
                pf for pf in self.pareto_front 
                if not (np.all(objectives <= pf['objectives']) and np.any(objectives < pf['objectives']))
            ]
            
            # Add new solution
            self.pareto_front.append({
                'parameters': params.copy(),
                'objectives': objectives.copy()
            })
    
    def optimize(self, objective_functions: List[Callable], initial_params: np.ndarray,
                weights: Optional[np.ndarray] = None, max_iter: int = 100) -> Dict[str, Any]:
        """Run multi-objective optimization"""
        
        if weights is None:
            weights = np.ones(len(objective_functions)) / len(objective_functions)
        self.objective_weights = weights
        
        # Weighted sum approach
        def weighted_objective(params):
            objectives = self.evaluate_objectives(params, objective_functions)
            return np.dot(weights, objectives)
        
        start_time = time.time()
        
        # Use multiple starting points for diversity
        num_starts = min(10, max_iter // 10)
        best_results = []
        
        for start in range(num_starts):
            # Random perturbation of initial parameters
            perturbed_params = initial_params + np.random.normal(0, 0.1, size=initial_params.shape)
            
            # Run optimization from this starting point
            result = minimize(
                weighted_objective,
                perturbed_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter // num_starts}
            )
            
            if result.success:
                objectives = self.evaluate_objectives(result.x, objective_functions)
                self.update_pareto_front(result.x, objectives)
                best_results.append({
                    'parameters': result.x,
                    'objectives': objectives,
                    'weighted_value': result.fun
                })
        
        execution_time = time.time() - start_time
        
        return {
            'optimizer': 'MultiObjective',
            'pareto_front': self.pareto_front,
            'best_solutions': best_results,
            'num_objectives': len(objective_functions),
            'weights': weights,
            'execution_time': execution_time,
            'success': len(self.pareto_front) > 0
        }

class BayesianQuantumOptimizer(HybridOptimizer):
    """
    Bayesian Optimization for Quantum Circuits
    Uses Gaussian Process to model quantum objective function
    """
    
    def __init__(self, quantum_component: Any, acquisition_function: str = 'expected_improvement'):
        super().__init__(quantum_component, None)
        self.acquisition_function = acquisition_function
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
    
    def initialize_gp(self, bounds: List[Tuple[float, float]]):
        """Initialize Gaussian Process model"""
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
    
    def acquisition_expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function"""
        if len(self.y_observed) == 0:
            return np.ones(X.shape[0])
        
        mu, sigma = self.gp_model.predict(X.reshape(-1, len(X)), return_std=True)
        mu = mu.flatten()
        sigma = sigma.flatten()
        
        f_best = np.min(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = f_best - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]],
                n_initial: int = 5, max_iter: int = 50) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        
        if not SCIPY_AVAILABLE:
            raise RuntimeError("SciPy not available for Bayesian optimization")
        
        self.initialize_gp(bounds)
        start_time = time.time()
        
        # Initial random sampling
        for _ in range(n_initial):
            params = np.array([
                np.random.uniform(bound[0], bound[1]) for bound in bounds
            ])
            value = objective_function(params)
            
            self.X_observed.append(params)
            self.y_observed.append(value)
            self._log_iteration(params, value, len(self.X_observed), 'bayesian_initial')
        
        # Bayesian optimization loop
        for iteration in range(max_iter - n_initial):
            # Fit GP model
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            self.gp_model.fit(X_array, y_array)
            
            # Find next point to evaluate
            def neg_acquisition(x):
                return -self.acquisition_expected_improvement(x)
            
            # Multiple random starts for acquisition optimization
            best_acq_value = float('inf')
            best_next_params = None
            
            for _ in range(10):
                x0 = np.array([
                    np.random.uniform(bound[0], bound[1]) for bound in bounds
                ])
                
                result = minimize(
                    neg_acquisition,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_acq_value:
                    best_acq_value = result.fun
                    best_next_params = result.x
            
            if best_next_params is not None:
                # Evaluate objective at new point
                value = objective_function(best_next_params)
                
                self.X_observed.append(best_next_params)
                self.y_observed.append(value)
                self._log_iteration(best_next_params, value, 
                                 len(self.X_observed), 'bayesian_optimization')
        
        execution_time = time.time() - start_time
        
        return {
            'optimizer': 'BayesianOptimization',
            'optimal_parameters': self.best_params,
            'optimal_value': self.best_value,
            'total_evaluations': len(self.X_observed),
            'execution_time': execution_time,
            'convergence_data': self.optimization_history,
            'gp_model_trained': self.gp_model is not None,
            'success': True
        }

class ParallelHybridOptimizer(HybridOptimizer):
    """
    Parallel Hybrid Optimizer
    Runs multiple optimization strategies in parallel
    """
    
    def __init__(self, quantum_component: Any, classical_component: Any, max_workers: int = 4):
        super().__init__(quantum_component, classical_component)
        self.max_workers = max_workers
        self.parallel_results = []
    
    def run_optimizer(self, optimizer_config: Dict[str, Any], objective_function: Callable,
                     initial_params: np.ndarray) -> Dict[str, Any]:
        """Run a single optimizer configuration"""
        optimizer_type = optimizer_config['type']
        optimizer_params = optimizer_config.get('params', {})
        
        try:
            if optimizer_type == 'parameter_shift':
                optimizer = ParameterShiftOptimizer(self.quantum_component)
                result = optimizer.optimize(objective_function, initial_params, **optimizer_params)
            
            elif optimizer_type == 'bayesian':
                optimizer = BayesianQuantumOptimizer(self.quantum_component)
                bounds = optimizer_params.get('bounds', [(-np.pi, np.pi)] * len(initial_params))
                result = optimizer.optimize(objective_function, bounds, **optimizer_params)
            
            elif optimizer_type == 'scipy':
                method = optimizer_params.get('method', 'L-BFGS-B')
                result = minimize(
                    objective_function,
                    initial_params,
                    method=method,
                    options={'maxiter': optimizer_params.get('max_iter', 100)}
                )
                result = {
                    'optimizer': f'SciPy_{method}',
                    'optimal_parameters': result.x,
                    'optimal_value': result.fun,
                    'success': result.success,
                    'iterations': result.nit if hasattr(result, 'nit') else 0
                }
            
            else:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
            result['config'] = optimizer_config
            return result
            
        except Exception as e:
            logger.error(f"Optimizer {optimizer_type} failed: {e}")
            return {
                'optimizer': optimizer_type,
                'success': False,
                'error': str(e),
                'config': optimizer_config
            }
    
    def optimize(self, objective_function: Callable, initial_params: np.ndarray,
                optimizer_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multiple optimizers in parallel"""
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all optimization tasks
            future_to_config = {
                executor.submit(self.run_optimizer, config, objective_function, initial_params): config
                for config in optimizer_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    self.parallel_results.append(result)
                    
                    # Update best result
                    if result.get('success', False) and 'optimal_value' in result:
                        if result['optimal_value'] < self.best_value:
                            self.best_value = result['optimal_value']
                            self.best_params = result['optimal_parameters']
                            
                except Exception as e:
                    logger.error(f"Parallel optimization task failed: {e}")
                    self.parallel_results.append({
                        'optimizer': config.get('type', 'unknown'),
                        'success': False,
                        'error': str(e),
                        'config': config
                    })
        
        execution_time = time.time() - start_time
        
        # Sort results by performance
        successful_results = [r for r in self.parallel_results if r.get('success', False)]
        successful_results.sort(key=lambda x: x.get('optimal_value', float('inf')))
        
        return {
            'optimizer': 'ParallelHybrid',
            'optimal_parameters': self.best_params,
            'optimal_value': self.best_value,
            'all_results': self.parallel_results,
            'successful_results': successful_results,
            'best_optimizer': successful_results[0]['optimizer'] if successful_results else None,
            'total_optimizers': len(optimizer_configs),
            'successful_optimizers': len(successful_results),
            'execution_time': execution_time,
            'success': len(successful_results) > 0
        }

class HybridOptimizationManager:
    """Manager class for hybrid quantum-classical optimization"""
    
    def __init__(self):
        self.optimizers = {
            'parameter_shift': ParameterShiftOptimizer,
            'multi_objective': MultiObjectiveOptimizer,
            'bayesian': BayesianQuantumOptimizer,
            'parallel': ParallelHybridOptimizer
        }
        self.optimization_history = {}
    
    def create_optimizer(self, optimizer_type: str, **kwargs) -> HybridOptimizer:
        """Create an optimizer instance"""
        if optimizer_type not in self.optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = self.optimizers[optimizer_type]
        return optimizer_class(**kwargs)
    
    def run_optimization(self, optimizer_type: str, objective_function: Callable,
                        initial_params: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Run optimization with specified optimizer"""
        
        optimizer = self.create_optimizer(optimizer_type, **kwargs)
        result = optimizer.optimize(objective_function, initial_params, **kwargs)
        
        # Store in history
        optimization_id = f"{optimizer_type}_{int(time.time())}"
        self.optimization_history[optimization_id] = result
        result['optimization_id'] = optimization_id
        
        return result
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history"""
        return self.optimization_history
    
    def compare_optimizers(self, objective_function: Callable, initial_params: np.ndarray,
                          optimizer_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple optimizers on the same problem"""
        
        results = {}
        start_time = time.time()
        
        for config in optimizer_configs:
            optimizer_type = config['type']
            optimizer_params = config.get('params', {})
            
            try:
                result = self.run_optimization(
                    optimizer_type, objective_function, initial_params, **optimizer_params
                )
                results[optimizer_type] = result
                
            except Exception as e:
                logger.error(f"Optimizer comparison failed for {optimizer_type}: {e}")
                results[optimizer_type] = {
                    'success': False,
                    'error': str(e)
                }
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_results:
            best_optimizer = min(successful_results.items(), 
                               key=lambda x: x[1].get('optimal_value', float('inf')))
            
            return {
                'comparison_results': results,
                'best_optimizer': best_optimizer[0],
                'best_value': best_optimizer[1].get('optimal_value'),
                'total_comparison_time': total_time,
                'successful_optimizers': len(successful_results),
                'total_optimizers': len(optimizer_configs)
            }
        else:
            return {
                'comparison_results': results,
                'best_optimizer': None,
                'total_comparison_time': total_time,
                'successful_optimizers': 0,
                'total_optimizers': len(optimizer_configs),
                'error': 'No optimizers succeeded'
            }

# Global hybrid optimization manager
hybrid_optimizer_manager = HybridOptimizationManager()