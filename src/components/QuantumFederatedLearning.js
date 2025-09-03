import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';

const QuantumFederatedLearning = () => {
  const [experiments, setExperiments] = useState({});
  const [selectedExperiment, setSelectedExperiment] = useState('');
  const [experimentParams, setExperimentParams] = useState({
    num_clients: 3,
    num_qubits: 2,
    data_distribution: 'iid',
    num_rounds: 10,
    secure: false
  });
  const [experimentResults, setExperimentResults] = useState(null);
  const [experimentStatus, setExperimentStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  // Check Phase 4 availability
  useEffect(() => {
    checkPhase4Status();
  }, []);

  const checkPhase4Status = async () => {
    try {
      const response = await fetch('http://localhost:8000/phase4/status');
      if (!response.ok) {
        console.error('Phase 4 not available');
      }
    } catch (error) {
      console.error('Error checking Phase 4 status:', error);
    }
  };

  const createExperiment = async () => {
    setLoading(true);
    const experimentId = `fed_exp_${Date.now()}`;
    
    try {
      const response = await fetch('http://localhost:8000/federated_learning/create_experiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          experiment_id: experimentId,
          ...experimentParams
        })
      });

      if (response.ok) {
        const result = await response.json();
        setExperiments(prev => ({
          ...prev,
          [experimentId]: result
        }));
        setSelectedExperiment(experimentId);
      } else {
        const error = await response.json();
        console.error('Failed to create experiment:', error);
      }
    } catch (error) {
      console.error('Error creating experiment:', error);
    } finally {
      setLoading(false);
    }
  };

  const runExperiment = async () => {
    if (!selectedExperiment) return;
    
    setRunning(true);
    setExperimentResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/federated_learning/run_experiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          experiment_id: selectedExperiment,
          ...experimentParams
        })
      });

      if (response.ok) {
        const result = await response.json();
        setExperimentResults(result);
        
        // Get detailed status
        await getExperimentStatus();
      } else {
        const error = await response.json();
        setExperimentResults({ success: false, error: error.detail });
      }
    } catch (error) {
      console.error('Error running experiment:', error);
      setExperimentResults({ success: false, error: error.message });
    } finally {
      setRunning(false);
    }
  };

  const getExperimentStatus = async () => {
    if (!selectedExperiment) return;
    
    try {
      const response = await fetch(`http://localhost:8000/federated_learning/experiment_status/${selectedExperiment}`);
      
      if (response.ok) {
        const status = await response.json();
        setExperimentStatus(status);
      }
    } catch (error) {
      console.error('Error getting experiment status:', error);
    }
  };

  const handleParamChange = (param, value) => {
    setExperimentParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const renderExperimentConfig = () => {
    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Experiment Configuration</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Clients
            </label>
            <input
              type="number"
              min="2"
              max="10"
              value={experimentParams.num_clients}
              onChange={(e) => handleParamChange('num_clients', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Qubits
            </label>
            <input
              type="number"
              min="1"
              max="6"
              value={experimentParams.num_qubits}
              onChange={(e) => handleParamChange('num_qubits', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Data Distribution
            </label>
            <select
              value={experimentParams.data_distribution}
              onChange={(e) => handleParamChange('data_distribution', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="iid">IID (Independent & Identical)</option>
              <option value="non_iid">Non-IID (Heterogeneous)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Federated Rounds
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={experimentParams.num_rounds}
              onChange={(e) => handleParamChange('num_rounds', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="secure"
            checked={experimentParams.secure}
            onChange={(e) => handleParamChange('secure', e.target.checked)}
            className="rounded"
          />
          <label htmlFor="secure" className="text-sm font-medium text-gray-700">
            Enable Secure Aggregation
          </label>
        </div>
      </div>
    );
  };

  const renderExperimentResults = () => {
    if (!experimentResults) return null;

    if (!experimentResults.success) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Experiment failed: {experimentResults.error}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Federated Learning Results</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h5 className="font-medium text-green-800 mb-2">Training Summary</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Total Rounds:</span> {experimentResults.total_rounds}</div>
              <div><span className="font-medium">Converged:</span> {experimentResults.converged ? 'Yes' : 'No'}</div>
              <div><span className="font-medium">Training Time:</span> {experimentResults.total_training_time?.toFixed(2)}s</div>
              <div><span className="font-medium">Avg Round Time:</span> {experimentResults.average_round_time?.toFixed(2)}s</div>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h5 className="font-medium text-blue-800 mb-2">Model Performance</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Final Parameter Norm:</span> {experimentResults.final_parameter_norm?.toFixed(4)}</div>
              <div><span className="font-medium">Experiment ID:</span> {experimentResults.experiment_id}</div>
            </div>
          </div>
        </div>

        {experimentResults.round_history && (
          <div className="bg-gray-50 p-4 rounded-lg">
            <h5 className="font-medium text-gray-800 mb-2">Round History</h5>
            <div className="max-h-40 overflow-y-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-1">Round</th>
                    <th className="text-left py-1">Clients</th>
                    <th className="text-left py-1">Parameter Change</th>
                    <th className="text-left py-1">Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {experimentResults.round_history.slice(-5).map((round, idx) => (
                    <tr key={idx} className="border-b">
                      <td className="py-1">{round.round}</td>
                      <td className="py-1">{round.participating_clients}</td>
                      <td className="py-1">{round.parameter_change?.toFixed(6)}</td>
                      <td className="py-1">{round.round_time?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderExperimentStatus = () => {
    if (!experimentStatus) return null;

    return (
      <div className="bg-purple-50 p-4 rounded-lg">
        <h5 className="font-medium text-purple-800 mb-2">Current Status</h5>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div><span className="font-medium">Clients:</span> {experimentStatus.num_clients}</div>
          <div><span className="font-medium">Rounds Completed:</span> {experimentStatus.num_rounds_completed}</div>
          <div><span className="font-medium">Parameter Norm:</span> {experimentStatus.current_parameter_norm?.toFixed(4)}</div>
          <div><span className="font-medium">Last Change:</span> {experimentStatus.last_parameter_change?.toFixed(6)}</div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🌐</span>
            Quantum Federated Learning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Experiment Configuration */}
            <div className="border rounded-lg p-6">
              {renderExperimentConfig()}
              
              <div className="mt-6 flex gap-4">
                <Button
                  onClick={createExperiment}
                  disabled={loading}
                  className="flex-1"
                >
                  {loading ? 'Creating...' : 'Create Experiment'}
                </Button>
                
                <Button
                  onClick={runExperiment}
                  disabled={running || !selectedExperiment}
                  variant="outline"
                  className="flex-1"
                >
                  {running ? 'Running...' : 'Run Federated Training'}
                </Button>
                
                <Button
                  onClick={getExperimentStatus}
                  disabled={!selectedExperiment}
                  variant="outline"
                >
                  Refresh Status
                </Button>
              </div>
            </div>

            {/* Experiment Status */}
            {experimentStatus && (
              <div className="border rounded-lg p-6">
                {renderExperimentStatus()}
              </div>
            )}

            {/* Experiment Results */}
            {experimentResults && (
              <div className="border rounded-lg p-6">
                {renderExperimentResults()}
              </div>
            )}

            {/* Information Panel */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg">
                <h5 className="font-medium text-blue-800 mb-2">Federated Learning</h5>
                <p className="text-sm text-blue-700">
                  Train quantum models across multiple clients without sharing raw data, 
                  preserving privacy while leveraging distributed quantum resources.
                </p>
              </div>

              <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
                <h5 className="font-medium text-green-800 mb-2">Secure Aggregation</h5>
                <p className="text-sm text-green-700">
                  Use cryptographic protocols to securely aggregate quantum model updates 
                  without revealing individual client contributions.
                </p>
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
                <h5 className="font-medium text-purple-800 mb-2">Quantum Advantage</h5>
                <p className="text-sm text-purple-700">
                  Leverage quantum parallelism and entanglement for distributed learning 
                  that could outperform classical federated approaches.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default QuantumFederatedLearning;