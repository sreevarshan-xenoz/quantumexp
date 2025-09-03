import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';

const HybridOptimization = () => {
  const [availableOptimizers, setAvailableOptimizers] = useState(null);
  const [selectedOptimizer, setSelectedOptimizer] = useState('');
  const [optimizationParams, setOptimizationParams] = useState({});
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);

  // Fetch available optimizers on component mount
  useEffect(() => {
    fetchAvailableOptimizers();
  }, []);

  const fetchAvailableOptimizers = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/hybrid_optimization/available');
      if (response.ok) {
        const data = await response.json();
        setAvailableOptimizers(data);
        if (data.available_optimizers.length > 0) {
          setSelectedOptimizer(data.available_optimizers[0]);
        }
      } else {
        console.error('Failed to fetch available optimizers');
      }
    } catch (error) {
      console.error('Error fetching optimizers:', error);
    } finally {
      setLoading(false);
    }
  };

  const runOptimization = async () => {
    setExecuting(true);
    setOptimizationResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/hybrid_optimization/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          optimizer_type: selectedOptimizer,
          objective_type: optimizationParams.objective_type || 'quadratic',
          num_parameters: optimizationParams.num_parameters || 4,
          optimization_params: {
            max_iter: optimizationParams.max_iter || 50,
            learning_rate: optimizationParams.learning_rate || 0.01,
            tolerance: optimizationParams.tolerance || 1e-6,
            n_initial: optimizationParams.n_initial || 5
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        setOptimizationResults(result);
      } else {
        const error = await response.json();
        setOptimizationResults({ success: false, error: Array.isArray(error.detail) ? error.detail.map(e => e.msg || e).join(', ') : error.detail });
      }
    } catch (error) {
      console.error('Error running optimization:', error);
      setOptimizationResults({ success: false, error: error.message });
    } finally {
      setExecuting(false);
    }
  };

  const compareOptimizers = async () => {
    setExecuting(true);
    setComparisonResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/hybrid_optimization/compare', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          objective_type: optimizationParams.objective_type || 'quadratic',
          num_parameters: optimizationParams.num_parameters || 4,
          optimizers: ['parameter_shift', 'bayesian']
        })
      });

      if (response.ok) {
        const result = await response.json();
        setComparisonResults(result);
      } else {
        const error = await response.json();
        setComparisonResults({ success: false, error: Array.isArray(error.detail) ? error.detail.map(e => e.msg || e).join(', ') : error.detail });
      }
    } catch (error) {
      console.error('Error comparing optimizers:', error);
      setComparisonResults({ success: false, error: error.message });
    } finally {
      setExecuting(false);
    }
  };

  const handleParamChange = (param, value) => {
    setOptimizationParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const getOptimizerIcon = (optimizer) => {
    const icons = {
      'parameter_shift': '🎯',
      'multi_objective': '🎪',
      'bayesian': '🧠',
      'parallel': '⚡'
    };
    return icons[optimizer] || '🔧';
  };

  const renderParameterControls = () => {
    if (!selectedOptimizer || !availableOptimizers) return null;

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Optimization Parameters</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Objective Function
            </label>
            <select
              value={optimizationParams.objective_type || 'quadratic'}
              onChange={(e) => handleParamChange('objective_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="quadratic">Quadratic (Simple)</option>
              <option value="rastrigin">Rastrigin (Multi-modal)</option>
              <option value="rosenbrock">Rosenbrock (Valley)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Parameters
            </label>
            <input
              type="number"
              min="2"
              max="10"
              value={optimizationParams.num_parameters || 4}
              onChange={(e) => handleParamChange('num_parameters', parseInt(e.target.value))}
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
              value={optimizationParams.max_iter || 50}
              onChange={(e) => handleParamChange('max_iter', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {selectedOptimizer === 'parameter_shift' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                min="0.001"
                max="1"
                step="0.001"
                value={optimizationParams.learning_rate || 0.01}
                onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          {selectedOptimizer === 'bayesian' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Initial Samples
              </label>
              <input
                type="number"
                min="3"
                max="20"
                value={optimizationParams.n_initial || 5}
                onChange={(e) => handleParamChange('n_initial', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderOptimizationResults = () => {
    if (!optimizationResults) return null;

    if (!optimizationResults.success) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Optimization failed: {typeof optimizationResults.error === 'string' ? optimizationResults.error : JSON.stringify(optimizationResults.error)}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Optimization Results</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h5 className="font-medium text-green-800 mb-2">Performance</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Optimizer:</span> {optimizationResults.optimizer}</div>
              <div><span className="font-medium">Optimal Value:</span> {optimizationResults.optimal_value?.toFixed(6)}</div>
              <div><span className="font-medium">Execution Time:</span> {optimizationResults.execution_time?.toFixed(3)}s</div>
              <div><span className="font-medium">Iterations:</span> {optimizationResults.iterations}</div>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h5 className="font-medium text-blue-800 mb-2">Configuration</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Objective:</span> {optimizationResults.objective_type}</div>
              <div><span className="font-medium">Parameters:</span> {optimizationResults.num_parameters}</div>
              {optimizationResults.gradient_norms && (
                <div><span className="font-medium">Final Gradient Norm:</span> {optimizationResults.gradient_norms[optimizationResults.gradient_norms.length - 1]?.toFixed(6)}</div>
              )}
            </div>
          </div>
        </div>

        {optimizationResults.optimal_parameters && (
          <div className="bg-gray-50 p-4 rounded-lg">
            <h5 className="font-medium text-gray-800 mb-2">Optimal Parameters</h5>
            <div className="grid grid-cols-4 gap-2">
              {optimizationResults.optimal_parameters.slice(0, 8).map((param, idx) => (
                <div key={idx} className="text-center p-2 bg-white rounded border">
                  <div className="text-xs text-gray-500">θ{idx}</div>
                  <div className="font-mono text-sm">{param.toFixed(4)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {optimizationResults.convergence_data && (
          <div className="bg-purple-50 p-4 rounded-lg">
            <h5 className="font-medium text-purple-800 mb-2">Convergence Analysis</h5>
            <div className="text-sm">
              <div>Convergence achieved in {optimizationResults.convergence_data.length} iterations</div>
              <div>Best value found: {Math.min(...optimizationResults.convergence_data.map(d => d.value)).toFixed(6)}</div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderComparisonResults = () => {
    if (!comparisonResults) return null;

    if (!comparisonResults.successful_optimizers) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Comparison failed: {typeof comparisonResults.error === 'string' ? comparisonResults.error : JSON.stringify(comparisonResults.error) || 'No optimizers succeeded'}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Optimizer Comparison</h4>
        
        <div className="bg-yellow-50 p-4 rounded-lg">
          <h5 className="font-medium text-yellow-800 mb-2">Winner</h5>
          <div className="text-sm">
            <div><span className="font-medium">Best Optimizer:</span> {comparisonResults.best_optimizer}</div>
            <div><span className="font-medium">Best Value:</span> {comparisonResults.best_value?.toFixed(6)}</div>
            <div><span className="font-medium">Total Time:</span> {comparisonResults.total_comparison_time?.toFixed(3)}s</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(comparisonResults.comparison_results).map(([optimizer, result]) => (
            <div key={optimizer} className={`p-4 rounded-lg border ${
              result.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
            }`}>
              <h6 className="font-medium mb-2 flex items-center gap-2">
                <span>{getOptimizerIcon(optimizer)}</span>
                {optimizer}
                <Badge variant={result.success ? "success" : "danger"}>
                  {result.success ? "Success" : "Failed"}
                </Badge>
              </h6>
              {result.success ? (
                <div className="text-sm space-y-1">
                  <div><span className="font-medium">Value:</span> {result.optimal_value?.toFixed(6)}</div>
                  <div><span className="font-medium">Time:</span> {result.execution_time?.toFixed(3)}s</div>
                  <div><span className="font-medium">Iterations:</span> {result.iterations}</div>
                </div>
              ) : (
                <div className="text-sm text-red-600">
                  Error: {typeof result.error === 'string' ? result.error : JSON.stringify(result.error)}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading hybrid optimizers...</span>
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
            <span className="text-2xl">🔄</span>
            Hybrid Quantum-Classical Optimization
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!availableOptimizers ? (
            <Alert>
              <AlertDescription>
                Hybrid optimization is not available. Please ensure the backend modules are installed.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-6">
              {/* Optimizer Selection */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Available Optimizers</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {availableOptimizers.available_optimizers.map((optimizer) => {
                    const info = availableOptimizers.optimizer_details[optimizer];
                    return (
                      <div
                        key={optimizer}
                        className={`border rounded-lg p-4 cursor-pointer transition-all ${
                          selectedOptimizer === optimizer
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                        onClick={() => setSelectedOptimizer(optimizer)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-2xl">{getOptimizerIcon(optimizer)}</span>
                          <Badge variant={selectedOptimizer === optimizer ? "primary" : "secondary"}>
                            {optimizer}
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
              {selectedOptimizer && (
                <div className="border rounded-lg p-6">
                  {renderParameterControls()}
                  
                  <div className="mt-6 flex justify-center gap-4">
                    <Button
                      onClick={runOptimization}
                      disabled={executing}
                      size="lg"
                    >
                      {executing ? 'Optimizing...' : `Run ${selectedOptimizer}`}
                    </Button>
                    
                    <Button
                      onClick={compareOptimizers}
                      disabled={executing}
                      variant="outline"
                      size="lg"
                    >
                      {executing ? 'Comparing...' : 'Compare Optimizers'}
                    </Button>
                  </div>
                </div>
              )}

              {/* Single Optimization Results */}
              {optimizationResults && (
                <div className="border rounded-lg p-6">
                  {renderOptimizationResults()}
                </div>
              )}

              {/* Comparison Results */}
              {comparisonResults && (
                <div className="border rounded-lg p-6">
                  {renderComparisonResults()}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default HybridOptimization;