import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';

const AdvancedQuantumAlgorithms = () => {
  const [availableAlgorithms, setAvailableAlgorithms] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [algorithmParams, setAlgorithmParams] = useState({});
  const [executionResults, setExecutionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);

  // Fetch available algorithms on component mount
  useEffect(() => {
    fetchAvailableAlgorithms();
  }, []);

  const fetchAvailableAlgorithms = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/advanced_algorithms/available');
      if (response.ok) {
        const data = await response.json();
        setAvailableAlgorithms(data);
        if (data.available_algorithms.length > 0) {
          setSelectedAlgorithm(data.available_algorithms[0]);
        }
      } else {
        console.error('Failed to fetch available algorithms');
      }
    } catch (error) {
      console.error('Error fetching algorithms:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeAlgorithm = async (algorithmType, specificParams = {}) => {
    setExecuting(true);
    setExecutionResults(null);
    
    try {
      let endpoint = '';
      let requestBody = {};

      switch (algorithmType) {
        case 'QAOA':
          endpoint = 'http://localhost:8000/advanced_algorithms/qaoa';
          requestBody = {
            num_qubits: algorithmParams.num_qubits || 4,
            problem_type: algorithmParams.problem_type || 'max_cut',
            p_layers: algorithmParams.p_layers || 1,
            optimizer: algorithmParams.optimizer || 'COBYLA',
            max_iter: algorithmParams.max_iter || 50,
            ...specificParams
          };
          break;

        case 'VQE':
          endpoint = 'http://localhost:8000/advanced_algorithms/vqe';
          requestBody = {
            num_qubits: algorithmParams.num_qubits || 2,
            molecule: algorithmParams.molecule || 'H2',
            ansatz_type: algorithmParams.ansatz_type || 'RealAmplitudes',
            optimizer: algorithmParams.optimizer || 'SPSA',
            max_iter: algorithmParams.max_iter || 50,
            ...specificParams
          };
          break;

        case 'QNN':
          endpoint = 'http://localhost:8000/advanced_algorithms/qnn_train';
          requestBody = {
            num_qubits: algorithmParams.num_qubits || 2,
            num_layers: algorithmParams.num_layers || 2,
            feature_map_type: algorithmParams.feature_map_type || 'ZZ',
            optimizer: algorithmParams.optimizer || 'SPSA',
            max_iter: algorithmParams.max_iter || 50,
            dataset_size: algorithmParams.dataset_size || 100,
            ...specificParams
          };
          break;

        case 'QFT':
          endpoint = 'http://localhost:8000/advanced_algorithms/qft';
          requestBody = {
            num_qubits: algorithmParams.num_qubits || 3,
            ...specificParams
          };
          break;

        default:
          throw new Error(`Unknown algorithm type: ${algorithmType}`);
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (response.ok) {
        const result = await response.json();
        setExecutionResults(result);
      } else {
        const error = await response.json();
        setExecutionResults({ success: false, error: Array.isArray(error.detail) ? error.detail.map(e => e.msg || e).join(', ') : error.detail });
      }
    } catch (error) {
      console.error('Error executing algorithm:', error);
      setExecutionResults({ success: false, error: error.message });
    } finally {
      setExecuting(false);
    }
  };

  const handleParamChange = (param, value) => {
    setAlgorithmParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const getAlgorithmIcon = (algorithm) => {
    const icons = {
      'QAOA': '🔄',
      'VQE': '⚛️',
      'QNN': '🧠',
      'QFT': '📊',
      'QPE': '📐'
    };
    return icons[algorithm] || '🔬';
  };

  const renderParameterControls = () => {
    if (!selectedAlgorithm || !availableAlgorithms) return null;

    const algorithmInfo = availableAlgorithms.algorithm_details[selectedAlgorithm];
    
    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Algorithm Parameters</h4>
        
        {/* Common parameters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Qubits
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={algorithmParams.num_qubits || 2}
              onChange={(e) => handleParamChange('num_qubits', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Iterations
            </label>
            <input
              type="number"
              min="10"
              max="500"
              value={algorithmParams.max_iter || 50}
              onChange={(e) => handleParamChange('max_iter', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Algorithm-specific parameters */}
        {selectedAlgorithm === 'QAOA' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Problem Type
              </label>
              <select
                value={algorithmParams.problem_type || 'max_cut'}
                onChange={(e) => handleParamChange('problem_type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="max_cut">Max Cut</option>
                <option value="portfolio">Portfolio Optimization</option>
                <option value="tsp">Traveling Salesman</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                QAOA Layers (p)
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={algorithmParams.p_layers || 1}
                onChange={(e) => handleParamChange('p_layers', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        )}

        {selectedAlgorithm === 'VQE' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Molecule
              </label>
              <select
                value={algorithmParams.molecule || 'H2'}
                onChange={(e) => handleParamChange('molecule', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="H2">H₂ (Hydrogen)</option>
                <option value="LiH">LiH (Lithium Hydride)</option>
                <option value="BeH2">BeH₂ (Beryllium Hydride)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Ansatz Type
              </label>
              <select
                value={algorithmParams.ansatz_type || 'RealAmplitudes'}
                onChange={(e) => handleParamChange('ansatz_type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="RealAmplitudes">Real Amplitudes</option>
                <option value="EfficientSU2">Efficient SU(2)</option>
                <option value="Custom">Custom</option>
              </select>
            </div>
          </div>
        )}

        {selectedAlgorithm === 'QNN' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Number of Layers
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={algorithmParams.num_layers || 2}
                onChange={(e) => handleParamChange('num_layers', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Feature Map
              </label>
              <select
                value={algorithmParams.feature_map_type || 'ZZ'}
                onChange={(e) => handleParamChange('feature_map_type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ZZ">ZZ Feature Map</option>
                <option value="Pauli">Pauli Feature Map</option>
                <option value="Custom">Custom</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dataset Size
              </label>
              <input
                type="number"
                min="50"
                max="1000"
                value={algorithmParams.dataset_size || 100}
                onChange={(e) => handleParamChange('dataset_size', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Optimizer
          </label>
          <select
            value={algorithmParams.optimizer || 'SPSA'}
            onChange={(e) => handleParamChange('optimizer', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="SPSA">SPSA</option>
            <option value="COBYLA">COBYLA</option>
            <option value="L_BFGS_B">L-BFGS-B</option>
            <option value="SLSQP">SLSQP</option>
          </select>
        </div>
      </div>
    );
  };

  const renderResults = () => {
    if (!executionResults) return null;

    if (!executionResults.success) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Execution failed: {typeof executionResults.error === 'string' ? executionResults.error : JSON.stringify(executionResults.error)}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Execution Results</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h5 className="font-medium text-green-800 mb-2">Algorithm Info</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Algorithm:</span> {executionResults.algorithm}</div>
              <div><span className="font-medium">Execution Time:</span> {executionResults.execution_time?.toFixed(3)}s</div>
              {executionResults.num_qubits && (
                <div><span className="font-medium">Qubits:</span> {executionResults.num_qubits}</div>
              )}
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h5 className="font-medium text-blue-800 mb-2">Results</h5>
            <div className="space-y-1 text-sm">
              {executionResults.optimal_value !== undefined && (
                <div><span className="font-medium">Optimal Value:</span> {executionResults.optimal_value.toFixed(6)}</div>
              )}
              {executionResults.optimal_energy !== undefined && (
                <div><span className="font-medium">Ground Energy:</span> {executionResults.optimal_energy.toFixed(6)}</div>
              )}
              {executionResults.estimated_phase !== undefined && (
                <div><span className="font-medium">Estimated Phase:</span> {executionResults.estimated_phase.toFixed(6)}</div>
              )}
              {executionResults.optimizer_evals && (
                <div><span className="font-medium">Optimizer Evaluations:</span> {executionResults.optimizer_evals}</div>
              )}
            </div>
          </div>
        </div>

        {executionResults.measurement_counts && (
          <div className="bg-gray-50 p-4 rounded-lg">
            <h5 className="font-medium text-gray-800 mb-2">Measurement Counts</h5>
            <div className="grid grid-cols-4 gap-2">
              {Object.entries(executionResults.measurement_counts).slice(0, 8).map(([state, count]) => (
                <div key={state} className="text-center p-2 bg-white rounded border">
                  <div className="font-mono text-sm">{state}</div>
                  <div className="font-bold text-blue-600">{count}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading advanced quantum algorithms...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🔬</span>
            Advanced Quantum Algorithms
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!availableAlgorithms ? (
            <Alert>
              <AlertDescription>
                Advanced quantum algorithms are not available. Please ensure the backend modules are installed.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-6">
              {/* Algorithm Selection */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Available Algorithms</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {availableAlgorithms.available_algorithms.map((algorithm) => {
                    const info = availableAlgorithms.algorithm_details[algorithm];
                    return (
                      <div
                        key={algorithm}
                        className={`border rounded-lg p-4 cursor-pointer transition-all ${
                          selectedAlgorithm === algorithm
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedAlgorithm(algorithm)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-2xl">{getAlgorithmIcon(algorithm)}</span>
                          <Badge variant={selectedAlgorithm === algorithm ? "primary" : "secondary"}>
                            {algorithm}
                          </Badge>
                        </div>
                        <h4 className="font-medium text-gray-800 mb-1">{info.name}</h4>
                        <p className="text-sm text-gray-600">{info.description}</p>
                        <div className="mt-2">
                          <div className="text-xs text-gray-500">Use Cases:</div>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {info.use_cases?.slice(0, 2).map((useCase, idx) => (
                              <span key={idx} className="text-xs bg-gray-100 px-2 py-1 rounded">
                                {useCase}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Parameter Controls */}
              {selectedAlgorithm && (
                <div className="border rounded-lg p-6">
                  {renderParameterControls()}
                  
                  <div className="mt-6 flex justify-center">
                    <Button
                      onClick={() => executeAlgorithm(selectedAlgorithm)}
                      disabled={executing}
                      size="lg"
                    >
                      {executing ? 'Executing...' : `Run ${selectedAlgorithm}`}
                    </Button>
                  </div>
                </div>
              )}

              {/* Results */}
              {executionResults && (
                <div className="border rounded-lg p-6">
                  {renderResults()}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default AdvancedQuantumAlgorithms;