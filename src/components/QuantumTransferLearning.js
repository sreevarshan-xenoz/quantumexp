import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';

const QuantumTransferLearning = () => {
  const [extractors, setExtractors] = useState({});
  const [models, setModels] = useState({});
  const [selectedExtractor, setSelectedExtractor] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [extractorParams, setExtractorParams] = useState({
    num_qubits: 2,
    feature_map_type: 'ZZ'
  });
  const [modelParams, setModelParams] = useState({
    classifier_type: 'quantum'
  });
  const [transferResults, setTransferResults] = useState(null);
  const [evaluationResults, setEvaluationResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);

  // Load existing models on component mount
  useEffect(() => {
    loadExistingModels();
  }, []);

  const loadExistingModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/transfer_learning/models');
      if (response.ok) {
        const data = await response.json();
        console.log('Existing models:', data);
      }
    } catch (error) {
      console.error('Error loading existing models:', error);
    }
  };

  const createFeatureExtractor = async () => {
    setLoading(true);
    const extractorId = `extractor_${Date.now()}`;
    
    try {
      const response = await fetch('http://localhost:8000/transfer_learning/create_extractor', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          extractor_id: extractorId,
          model_id: '', // Not used for extractor creation
          classifier_type: 'quantum',
          ...extractorParams
        })
      });

      if (response.ok) {
        const result = await response.json();
        setExtractors(prev => ({
          ...prev,
          [extractorId]: result
        }));
        setSelectedExtractor(extractorId);
      } else {
        const error = await response.json();
        console.error('Failed to create extractor:', error);
      }
    } catch (error) {
      console.error('Error creating extractor:', error);
    } finally {
      setLoading(false);
    }
  };

  const createTransferModel = async () => {
    if (!selectedExtractor) {
      alert('Please create a feature extractor first');
      return;
    }

    setLoading(true);
    const modelId = `model_${Date.now()}`;
    
    try {
      const response = await fetch('http://localhost:8000/transfer_learning/create_model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelId,
          extractor_id: selectedExtractor,
          ...modelParams
        })
      });

      if (response.ok) {
        const result = await response.json();
        setModels(prev => ({
          ...prev,
          [modelId]: result
        }));
        setSelectedModel(modelId);
      } else {
        const error = await response.json();
        console.error('Failed to create model:', error);
      }
    } catch (error) {
      console.error('Error creating model:', error);
    } finally {
      setLoading(false);
    }
  };

  const fineTuneModel = async () => {
    if (!selectedModel) {
      alert('Please create a transfer model first');
      return;
    }

    setTraining(true);
    setTransferResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/transfer_learning/fine_tune', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          dataset_size: 100,
          epochs: 10
        })
      });

      if (response.ok) {
        const result = await response.json();
        setTransferResults(result);
      } else {
        const error = await response.json();
        setTransferResults({ success: false, error: error.detail });
      }
    } catch (error) {
      console.error('Error fine-tuning model:', error);
      setTransferResults({ success: false, error: error.message });
    } finally {
      setTraining(false);
    }
  };

  const evaluateModel = async () => {
    if (!selectedModel) {
      alert('Please create and train a model first');
      return;
    }

    setLoading(true);
    setEvaluationResults(null);
    
    try {
      const response = await fetch('http://localhost:8000/transfer_learning/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          test_dataset_size: 50
        })
      });

      if (response.ok) {
        const result = await response.json();
        setEvaluationResults(result);
      } else {
        const error = await response.json();
        setEvaluationResults({ success: false, error: error.detail });
      }
    } catch (error) {
      console.error('Error evaluating model:', error);
      setEvaluationResults({ success: false, error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const handleExtractorParamChange = (param, value) => {
    setExtractorParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const handleModelParamChange = (param, value) => {
    setModelParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const renderExtractorConfig = () => {
    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Feature Extractor Configuration</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Qubits
            </label>
            <input
              type="number"
              min="1"
              max="6"
              value={extractorParams.num_qubits}
              onChange={(e) => handleExtractorParamChange('num_qubits', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Feature Map Type
            </label>
            <select
              value={extractorParams.feature_map_type}
              onChange={(e) => handleExtractorParamChange('feature_map_type', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="ZZ">ZZ Feature Map</option>
              <option value="Pauli">Pauli Feature Map</option>
              <option value="Custom">Custom Feature Map</option>
            </select>
          </div>
        </div>

        <div className="flex justify-center">
          <Button
            onClick={createFeatureExtractor}
            disabled={loading}
            className="w-full md:w-auto"
          >
            {loading ? 'Creating...' : 'Create Pre-trained Extractor'}
          </Button>
        </div>

        {selectedExtractor && (
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="text-sm text-green-800">
              ✅ Feature extractor created: <span className="font-mono">{selectedExtractor}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderModelConfig = () => {
    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Transfer Model Configuration</h4>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Classifier Type
          </label>
          <select
            value={modelParams.classifier_type}
            onChange={(e) => handleModelParamChange('classifier_type', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="quantum">Quantum Classifier</option>
            <option value="classical">Classical Classifier</option>
          </select>
        </div>

        <div className="flex justify-center">
          <Button
            onClick={createTransferModel}
            disabled={loading || !selectedExtractor}
            className="w-full md:w-auto"
          >
            {loading ? 'Creating...' : 'Create Transfer Model'}
          </Button>
        </div>

        {selectedModel && (
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-sm text-blue-800">
              ✅ Transfer model created: <span className="font-mono">{selectedModel}</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderTransferResults = () => {
    if (!transferResults) return null;

    if (!transferResults.success) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Fine-tuning failed: {transferResults.error}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Fine-tuning Results</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-green-50 p-4 rounded-lg">
            <h5 className="font-medium text-green-800 mb-2">Training Summary</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Training Time:</span> {transferResults.training_time?.toFixed(2)}s</div>
              <div><span className="font-medium">Features Frozen:</span> {transferResults.features_frozen ? 'Yes' : 'No'}</div>
              <div><span className="font-medium">Status:</span> {transferResults.fine_tuning_completed ? 'Completed' : 'Failed'}</div>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-lg">
            <h5 className="font-medium text-blue-800 mb-2">Performance</h5>
            <div className="space-y-1 text-sm">
              {transferResults.performance && (
                <>
                  <div><span className="font-medium">Accuracy:</span> {(transferResults.performance.accuracy * 100).toFixed(1)}%</div>
                  <div><span className="font-medium">Loss:</span> {transferResults.performance.loss?.toFixed(4)}</div>
                  {transferResults.performance.optimizer_evaluations && (
                    <div><span className="font-medium">Optimizer Evals:</span> {transferResults.performance.optimizer_evaluations}</div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderEvaluationResults = () => {
    if (!evaluationResults) return null;

    if (!evaluationResults.success) {
      return (
        <Alert variant="error">
          <AlertDescription>
            Evaluation failed: {evaluationResults.error}
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-4">
        <h4 className="font-semibold text-gray-800">Evaluation Results</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-purple-50 p-4 rounded-lg">
            <h5 className="font-medium text-purple-800 mb-2">Test Performance</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Test Accuracy:</span> {(evaluationResults.accuracy * 100).toFixed(1)}%</div>
              <div><span className="font-medium">Test Samples:</span> {evaluationResults.predictions?.length || 0}</div>
            </div>
          </div>

          <div className="bg-orange-50 p-4 rounded-lg">
            <h5 className="font-medium text-orange-800 mb-2">Model Info</h5>
            <div className="space-y-1 text-sm">
              <div><span className="font-medium">Model ID:</span> {evaluationResults.model_id}</div>
              <div><span className="font-medium">Classifier:</span> {modelParams.classifier_type}</div>
            </div>
          </div>
        </div>

        {evaluationResults.classification_report && Object.keys(evaluationResults.classification_report).length > 0 && (
          <div className="bg-gray-50 p-4 rounded-lg">
            <h5 className="font-medium text-gray-800 mb-2">Classification Report</h5>
            <div className="text-sm">
              <pre className="whitespace-pre-wrap">
                {JSON.stringify(evaluationResults.classification_report, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🔄</span>
            Quantum Transfer Learning
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Step 1: Feature Extractor */}
            <div className="border rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="primary">Step 1</Badge>
                <span className="font-semibold">Create Pre-trained Feature Extractor</span>
              </div>
              {renderExtractorConfig()}
            </div>

            {/* Step 2: Transfer Model */}
            <div className="border rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <Badge variant={selectedExtractor ? "primary" : "secondary"}>Step 2</Badge>
                <span className="font-semibold">Create Transfer Learning Model</span>
              </div>
              {renderModelConfig()}
            </div>

            {/* Step 3: Fine-tuning */}
            <div className="border rounded-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <Badge variant={selectedModel ? "primary" : "secondary"}>Step 3</Badge>
                <span className="font-semibold">Fine-tune on Target Domain</span>
              </div>
              
              <div className="flex gap-4 justify-center">
                <Button
                  onClick={fineTuneModel}
                  disabled={training || !selectedModel}
                  className="flex-1 max-w-xs"
                >
                  {training ? 'Fine-tuning...' : 'Fine-tune Model'}
                </Button>
                
                <Button
                  onClick={evaluateModel}
                  disabled={loading || !selectedModel}
                  variant="outline"
                  className="flex-1 max-w-xs"
                >
                  {loading ? 'Evaluating...' : 'Evaluate Model'}
                </Button>
              </div>
            </div>

            {/* Results */}
            {transferResults && (
              <div className="border rounded-lg p-6">
                {renderTransferResults()}
              </div>
            )}

            {evaluationResults && (
              <div className="border rounded-lg p-6">
                {renderEvaluationResults()}
              </div>
            )}

            {/* Information Panel */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg">
                <h5 className="font-medium text-blue-800 mb-2">Pre-trained Features</h5>
                <p className="text-sm text-blue-700">
                  Leverage quantum feature extractors trained on large datasets to 
                  capture quantum correlations and entanglement patterns.
                </p>
              </div>

              <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
                <h5 className="font-medium text-green-800 mb-2">Domain Adaptation</h5>
                <p className="text-sm text-green-700">
                  Fine-tune quantum models on new domains with limited data, 
                  transferring quantum knowledge across different problem spaces.
                </p>
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
                <h5 className="font-medium text-purple-800 mb-2">Few-shot Learning</h5>
                <p className="text-sm text-purple-700">
                  Achieve high performance with minimal training data by leveraging 
                  pre-trained quantum representations and transfer learning.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default QuantumTransferLearning;