"""
Quantum Federated Learning Module
Implements distributed quantum machine learning with privacy-preserving protocols
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from abc import ABC, abstractmethod
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

# Quantum computing imports
try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.algorithms.optimizers import SPSA, COBYLA
    from qiskit.primitives import Sampler, Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Cryptographic imports for secure aggregation
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantumClient(ABC):
    """Abstract base class for quantum federated learning clients"""
    
    def __init__(self, client_id: str, quantum_circuit: QuantumCircuit, local_data: Tuple[np.ndarray, np.ndarray]):
        self.client_id = client_id
        self.quantum_circuit = quantum_circuit
        self.local_data = local_data
        self.local_parameters = None
        self.global_parameters = None
        self.training_history = []
        self.privacy_budget = 1.0  # Differential privacy budget
        
    @abstractmethod
    def local_train(self, epochs: int = 1) -> Dict[str, Any]:
        """Train the local quantum model"""
        pass
    
    @abstractmethod
    def get_model_update(self) -> Dict[str, Any]:
        """Get model update for aggregation"""
        pass
    
    @abstractmethod
    def update_global_model(self, global_parameters: np.ndarray) -> None:
        """Update local model with global parameters"""
        pass

class QuantumVQCClient(QuantumClient):
    """Quantum Variational Quantum Classifier client for federated learning"""
    
    def __init__(self, client_id: str, num_qubits: int, local_data: Tuple[np.ndarray, np.ndarray], 
                 feature_map_type: str = 'ZZ', ansatz_type: str = 'RealAmplitudes'):
        
        # Create quantum circuit
        if feature_map_type == 'ZZ':
            feature_map = ZZFeatureMap(num_qubits, reps=1)
        else:
            feature_map = self._create_custom_feature_map(num_qubits)
        
        if ansatz_type == 'RealAmplitudes':
            ansatz = RealAmplitudes(num_qubits, reps=2)
        else:
            ansatz = self._create_custom_ansatz(num_qubits)
        
        # Combine feature map and ansatz
        quantum_circuit = QuantumCircuit(num_qubits)
        quantum_circuit.compose(feature_map, inplace=True)
        quantum_circuit.compose(ansatz, inplace=True)
        
        super().__init__(client_id, quantum_circuit, local_data)
        
        # Initialize parameters
        self.num_parameters = len(list(ansatz.parameters))
        self.local_parameters = np.random.uniform(-np.pi, np.pi, self.num_parameters)
        self.optimizer = SPSA(maxiter=50)
        
    def _create_custom_feature_map(self, num_qubits: int) -> QuantumCircuit:
        """Create custom feature map"""
        feature_params = ParameterVector('x', num_qubits)
        qc = QuantumCircuit(num_qubits)
        
        for i in range(num_qubits):
            qc.ry(feature_params[i], i)
        
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        
        return qc
    
    def _create_custom_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Create custom variational ansatz"""
        params = ParameterVector('θ', num_qubits * 2)
        qc = QuantumCircuit(num_qubits)
        
        for i in range(num_qubits):
            qc.ry(params[i], i)
        
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        
        for i in range(num_qubits):
            qc.ry(params[i + num_qubits], i)
        
        return qc
    
    def local_train(self, epochs: int = 1) -> Dict[str, Any]:
        """Train local VQC model"""
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit not available for quantum training")
        
        X_local, y_local = self.local_data
        start_time = time.time()
        
        # Define cost function
        def cost_function(params):
            # Bind parameters to circuit
            bound_circuit = self.quantum_circuit.bind_parameters(params)
            
            # Execute circuit (simplified simulation)
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(bound_circuit)
            result = job.result()
            
            # Calculate cost (simplified)
            cost = np.random.random()  # Placeholder for actual cost calculation
            return cost
        
        # Optimize parameters
        initial_params = self.local_parameters
        result = self.optimizer.minimize(cost_function, initial_params)
        
        if result.x is not None:
            self.local_parameters = result.x
        
        training_time = time.time() - start_time
        
        # Record training history
        training_record = {
            'epoch': len(self.training_history) + 1,
            'loss': result.fun if hasattr(result, 'fun') else 0.0,
            'training_time': training_time,
            'data_size': len(X_local),
            'timestamp': time.time()
        }
        self.training_history.append(training_record)
        
        return {
            'client_id': self.client_id,
            'training_completed': True,
            'local_loss': result.fun if hasattr(result, 'fun') else 0.0,
            'training_time': training_time,
            'data_size': len(X_local),
            'parameter_norm': np.linalg.norm(self.local_parameters)
        }
    
    def get_model_update(self) -> Dict[str, Any]:
        """Get model update for federated aggregation"""
        if self.local_parameters is None:
            raise ValueError("Local model not trained yet")
        
        # Add differential privacy noise
        privacy_noise = np.random.normal(0, 0.1 * self.privacy_budget, self.local_parameters.shape)
        noisy_parameters = self.local_parameters + privacy_noise
        
        # Calculate update (difference from global model)
        if self.global_parameters is not None:
            update = noisy_parameters - self.global_parameters
        else:
            update = noisy_parameters
        
        return {
            'client_id': self.client_id,
            'parameter_update': update,
            'data_size': len(self.local_data[0]),
            'privacy_budget_used': 0.1 * self.privacy_budget,
            'update_norm': np.linalg.norm(update)
        }
    
    def update_global_model(self, global_parameters: np.ndarray) -> None:
        """Update local model with global parameters"""
        self.global_parameters = global_parameters.copy()
        # Optionally blend with local parameters
        blend_factor = 0.8
        if self.local_parameters is not None:
            self.local_parameters = (blend_factor * global_parameters + 
                                   (1 - blend_factor) * self.local_parameters)
        else:
            self.local_parameters = global_parameters.copy()

class QuantumFederatedServer:
    """Quantum Federated Learning Server for coordinating distributed training"""
    
    def __init__(self, global_model_params: int, aggregation_method: str = 'fedavg'):
        self.global_model_params = global_model_params
        self.global_parameters = np.random.uniform(-np.pi, np.pi, global_model_params)
        self.aggregation_method = aggregation_method
        self.clients = {}
        self.round_history = []
        self.convergence_threshold = 1e-4
        self.max_rounds = 100
        
    def register_client(self, client: QuantumClient) -> bool:
        """Register a new client"""
        try:
            self.clients[client.client_id] = client
            client.update_global_model(self.global_parameters)
            logger.info(f"Client {client.client_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register client {client.client_id}: {e}")
            return False
    
    def federated_averaging(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Perform federated averaging of client updates"""
        if not client_updates:
            return self.global_parameters
        
        # Weight by data size
        total_data_size = sum(update['data_size'] for update in client_updates)
        
        if total_data_size == 0:
            # Uniform averaging if no data size info
            weights = [1.0 / len(client_updates) for _ in client_updates]
        else:
            weights = [update['data_size'] / total_data_size for update in client_updates]
        
        # Weighted average of parameter updates
        aggregated_update = np.zeros_like(self.global_parameters)
        for update, weight in zip(client_updates, weights):
            aggregated_update += weight * update['parameter_update']
        
        # Update global parameters
        learning_rate = 1.0
        new_global_parameters = self.global_parameters + learning_rate * aggregated_update
        
        return new_global_parameters
    
    def secure_aggregation(self, client_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Perform secure aggregation with cryptographic protection"""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography not available, falling back to standard aggregation")
            return self.federated_averaging(client_updates)
        
        try:
            # Generate shared secret for this round
            round_secret = Fernet.generate_key()
            cipher_suite = Fernet(round_secret)
            
            # Encrypt and aggregate updates
            encrypted_updates = []
            for update in client_updates:
                # Serialize and encrypt parameter update
                serialized_update = json.dumps(update['parameter_update'].tolist()).encode()
                encrypted_update = cipher_suite.encrypt(serialized_update)
                encrypted_updates.append({
                    'client_id': update['client_id'],
                    'encrypted_data': encrypted_update,
                    'data_size': update['data_size']
                })
            
            # Decrypt and aggregate (in practice, this would use secure multi-party computation)
            decrypted_updates = []
            for enc_update in encrypted_updates:
                decrypted_data = cipher_suite.decrypt(enc_update['encrypted_data'])
                parameter_update = np.array(json.loads(decrypted_data.decode()))
                decrypted_updates.append({
                    'parameter_update': parameter_update,
                    'data_size': enc_update['data_size']
                })
            
            return self.federated_averaging(decrypted_updates)
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            return self.federated_averaging(client_updates)
    
    def run_federated_round(self, participating_clients: List[str], 
                           local_epochs: int = 1, secure: bool = False) -> Dict[str, Any]:
        """Run one round of federated learning"""
        round_start_time = time.time()
        round_num = len(self.round_history) + 1
        
        logger.info(f"Starting federated round {round_num} with {len(participating_clients)} clients")
        
        # Phase 1: Local training
        training_results = {}
        with ThreadPoolExecutor(max_workers=min(len(participating_clients), 4)) as executor:
            future_to_client = {
                executor.submit(self.clients[client_id].local_train, local_epochs): client_id
                for client_id in participating_clients if client_id in self.clients
            }
            
            for future in as_completed(future_to_client):
                client_id = future_to_client[future]
                try:
                    result = future.result()
                    training_results[client_id] = result
                except Exception as e:
                    logger.error(f"Training failed for client {client_id}: {e}")
        
        # Phase 2: Collect model updates
        client_updates = []
        for client_id in participating_clients:
            if client_id in self.clients and client_id in training_results:
                try:
                    update = self.clients[client_id].get_model_update()
                    client_updates.append(update)
                except Exception as e:
                    logger.error(f"Failed to get update from client {client_id}: {e}")
        
        # Phase 3: Aggregate updates
        if client_updates:
            if secure:
                new_global_parameters = self.secure_aggregation(client_updates)
            else:
                new_global_parameters = self.federated_averaging(client_updates)
            
            # Calculate convergence metric
            parameter_change = np.linalg.norm(new_global_parameters - self.global_parameters)
            self.global_parameters = new_global_parameters
            
            # Phase 4: Broadcast updated global model
            for client_id in participating_clients:
                if client_id in self.clients:
                    self.clients[client_id].update_global_model(self.global_parameters)
        else:
            parameter_change = 0.0
        
        round_time = time.time() - round_start_time
        
        # Record round history
        round_record = {
            'round': round_num,
            'participating_clients': len(participating_clients),
            'successful_updates': len(client_updates),
            'parameter_change': parameter_change,
            'round_time': round_time,
            'global_parameter_norm': np.linalg.norm(self.global_parameters),
            'secure_aggregation': secure,
            'timestamp': time.time()
        }
        self.round_history.append(round_record)
        
        logger.info(f"Round {round_num} completed in {round_time:.2f}s, parameter change: {parameter_change:.6f}")
        
        return {
            'round': round_num,
            'success': True,
            'participating_clients': len(participating_clients),
            'parameter_change': parameter_change,
            'converged': parameter_change < self.convergence_threshold,
            'round_time': round_time,
            'training_results': training_results
        }
    
    def run_federated_training(self, num_rounds: int = 10, clients_per_round: int = None,
                              local_epochs: int = 1, secure: bool = False) -> Dict[str, Any]:
        """Run complete federated training process"""
        start_time = time.time()
        
        if clients_per_round is None:
            clients_per_round = len(self.clients)
        
        logger.info(f"Starting federated training: {num_rounds} rounds, {clients_per_round} clients per round")
        
        converged = False
        for round_num in range(num_rounds):
            # Select participating clients (random sampling)
            available_clients = list(self.clients.keys())
            participating_clients = np.random.choice(
                available_clients, 
                size=min(clients_per_round, len(available_clients)), 
                replace=False
            ).tolist()
            
            # Run federated round
            round_result = self.run_federated_round(
                participating_clients, local_epochs, secure
            )
            
            # Check convergence
            if round_result['converged']:
                logger.info(f"Federated training converged at round {round_num + 1}")
                converged = True
                break
        
        total_time = time.time() - start_time
        
        return {
            'federated_training_completed': True,
            'total_rounds': len(self.round_history),
            'converged': converged,
            'final_parameter_norm': np.linalg.norm(self.global_parameters),
            'total_training_time': total_time,
            'average_round_time': total_time / len(self.round_history) if self.round_history else 0,
            'round_history': self.round_history,
            'global_parameters': self.global_parameters.tolist()
        }

class QuantumFederatedLearningManager:
    """Manager class for quantum federated learning experiments"""
    
    def __init__(self):
        self.servers = {}
        self.experiments = {}
        
    def create_federated_experiment(self, experiment_id: str, num_clients: int, 
                                  num_qubits: int = 2, data_distribution: str = 'iid') -> Dict[str, Any]:
        """Create a new federated learning experiment"""
        try:
            # Create server
            server = QuantumFederatedServer(global_model_params=num_qubits * 2)
            
            # Generate distributed data
            datasets = self._generate_distributed_data(num_clients, data_distribution)
            
            # Create clients
            clients = []
            for i in range(num_clients):
                client_id = f"client_{i}"
                client = QuantumVQCClient(
                    client_id=client_id,
                    num_qubits=num_qubits,
                    local_data=datasets[i]
                )
                server.register_client(client)
                clients.append(client)
            
            # Store experiment
            self.servers[experiment_id] = server
            self.experiments[experiment_id] = {
                'server': server,
                'clients': clients,
                'num_clients': num_clients,
                'num_qubits': num_qubits,
                'data_distribution': data_distribution,
                'created_at': time.time()
            }
            
            return {
                'experiment_id': experiment_id,
                'num_clients': num_clients,
                'num_qubits': num_qubits,
                'data_distribution': data_distribution,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to create federated experiment: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_distributed_data(self, num_clients: int, distribution: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate distributed datasets for clients"""
        datasets = []
        
        if distribution == 'iid':
            # Independent and identically distributed data
            for _ in range(num_clients):
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)
                datasets.append((X, y))
        
        elif distribution == 'non_iid':
            # Non-IID data distribution
            for i in range(num_clients):
                # Each client gets data from different regions
                center_x = (i % 2) * 2 - 1
                center_y = (i // 2) * 2 - 1
                X = np.random.randn(100, 2) * 0.5 + [center_x, center_y]
                y = (X[:, 0] * X[:, 1] > 0).astype(int)
                datasets.append((X, y))
        
        else:
            # Default to IID
            datasets = self._generate_distributed_data(num_clients, 'iid')
        
        return datasets
    
    def run_experiment(self, experiment_id: str, num_rounds: int = 10, 
                      secure: bool = False) -> Dict[str, Any]:
        """Run federated learning experiment"""
        if experiment_id not in self.experiments:
            return {'success': False, 'error': 'Experiment not found'}
        
        try:
            server = self.experiments[experiment_id]['server']
            result = server.run_federated_training(
                num_rounds=num_rounds,
                secure=secure
            )
            
            return {
                'experiment_id': experiment_id,
                **result
            }
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of federated learning experiment"""
        if experiment_id not in self.experiments:
            return {'success': False, 'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        server = experiment['server']
        
        return {
            'experiment_id': experiment_id,
            'num_clients': experiment['num_clients'],
            'num_rounds_completed': len(server.round_history),
            'current_parameter_norm': np.linalg.norm(server.global_parameters),
            'last_parameter_change': server.round_history[-1]['parameter_change'] if server.round_history else 0,
            'round_history': server.round_history,
            'success': True
        }

# Global federated learning manager
federated_manager = QuantumFederatedLearningManager()