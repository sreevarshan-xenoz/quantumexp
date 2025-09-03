import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';

const QuantumHardwareManager = () => {
  const [hardwareStatus, setHardwareStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [costEstimate, setCostEstimate] = useState(null);
  const [executionResults, setExecutionResults] = useState(null);

  // Fetch hardware status on component mount
  useEffect(() => {
    fetchHardwareStatus();
  }, []);

  const fetchHardwareStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/quantum_hardware/status');
      if (response.ok) {
        const data = await response.json();
        setHardwareStatus(data);
      } else {
        console.error('Failed to fetch hardware status');
      }
    } catch (error) {
      console.error('Error fetching hardware status:', error);
    } finally {
      setLoading(false);
    }
  };

  const connectToProvider = async (provider) => {
    setConnecting(true);
    try {
      const response = await fetch(`http://localhost:8000/quantum_hardware/connect?provider=${provider}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Connected successfully:', data);
        // Refresh status after connection
        await fetchHardwareStatus();
      } else {
        const error = await response.json();
        console.error('Connection failed:', error);
      }
    } catch (error) {
      console.error('Error connecting to provider:', error);
    } finally {
      setConnecting(false);
    }
  };

  const estimateCost = async (provider, shots = 1000, qubits = 2) => {
    try {
      const response = await fetch(
        `http://localhost:8000/quantum_hardware/estimate_cost?provider=${provider}&shots=${shots}&num_qubits=${qubits}`,
        { method: 'POST' }
      );
      
      if (response.ok) {
        const data = await response.json();
        setCostEstimate(data);
      }
    } catch (error) {
      console.error('Error estimating cost:', error);
    }
  };

  const runOnHardware = async (provider) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/quantum_hardware/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: provider,
          shots: 1000,
          circuit_data: {
            // Placeholder circuit data
            gates: ['h', 'cx', 'measure'],
            qubits: 2
          }
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setExecutionResults(data);
      }
    } catch (error) {
      console.error('Error running on hardware:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (connected) => {
    return (
      <Badge variant={connected ? "success" : "secondary"}>
        {connected ? "Connected" : "Disconnected"}
      </Badge>
    );
  };

  if (loading && !hardwareStatus) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading quantum hardware status...</span>
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
            <span className="text-2xl">⚛️</span>
            Quantum Hardware Manager
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!hardwareStatus?.hardware_available ? (
            <Alert>
              <AlertDescription>
                Quantum hardware integration is not available. Please ensure quantum hardware modules are installed.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-6">
              {/* Provider Status */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Provider Status</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(hardwareStatus.providers || {}).map(([provider, connected]) => (
                    <div key={provider} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium capitalize">{provider}</span>
                        {getStatusBadge(connected)}
                      </div>
                      <div className="space-y-2">
                        {!connected && (
                          <Button
                            size="sm"
                            onClick={() => connectToProvider(provider)}
                            disabled={connecting}
                            className="w-full"
                          >
                            {connecting ? 'Connecting...' : 'Connect'}
                          </Button>
                        )}
                        {connected && (
                          <div className="space-y-2">
                            <Button
                              size="sm"
                              onClick={() => estimateCost(provider)}
                              variant="outline"
                              className="w-full"
                            >
                              Estimate Cost
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => runOnHardware(provider)}
                              disabled={loading}
                              className="w-full"
                            >
                              {loading ? 'Running...' : 'Run Test Circuit'}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Available Backends */}
              {hardwareStatus.available_backends && Object.keys(hardwareStatus.available_backends).length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Available Backends</h3>
                  <div className="space-y-2">
                    {Object.entries(hardwareStatus.available_backends).map(([provider, backends]) => (
                      <div key={provider} className="border rounded-lg p-3">
                        <div className="font-medium capitalize mb-2">{provider}</div>
                        <div className="flex flex-wrap gap-2">
                          {backends.map((backend) => (
                            <Badge key={backend} variant="outline">
                              {backend}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Cost Estimate */}
              {costEstimate && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Cost Estimate</h3>
                  <div className="border rounded-lg p-4 bg-blue-50">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="text-sm text-gray-600">Provider:</span>
                        <div className="font-medium capitalize">{costEstimate.provider}</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Estimated Cost:</span>
                        <div className="font-medium">${costEstimate.estimated_cost_usd}</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Shots:</span>
                        <div className="font-medium">{costEstimate.shots}</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Qubits:</span>
                        <div className="font-medium">{costEstimate.qubits}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Execution Results */}
              {executionResults && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Execution Results</h3>
                  <div className="border rounded-lg p-4 bg-green-50">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <span className="text-sm text-gray-600">Job ID:</span>
                        <div className="font-medium">{executionResults.job_id}</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Backend:</span>
                        <div className="font-medium">{executionResults.backend}</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Execution Time:</span>
                        <div className="font-medium">{executionResults.execution_time}s</div>
                      </div>
                      <div>
                        <span className="text-sm text-gray-600">Queue Time:</span>
                        <div className="font-medium">{executionResults.queue_time}s</div>
                      </div>
                    </div>
                    <div>
                      <span className="text-sm text-gray-600">Measurement Counts:</span>
                      <div className="mt-2 grid grid-cols-4 gap-2">
                        {Object.entries(executionResults.counts).map(([state, count]) => (
                          <div key={state} className="text-center p-2 bg-white rounded border">
                            <div className="font-mono text-sm">{state}</div>
                            <div className="font-bold">{count}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Refresh Button */}
              <div className="flex justify-center">
                <Button onClick={fetchHardwareStatus} disabled={loading}>
                  {loading ? 'Refreshing...' : 'Refresh Status'}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default QuantumHardwareManager;