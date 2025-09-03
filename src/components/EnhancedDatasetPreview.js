import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const EnhancedDatasetPreview = ({ onDatasetSelect }) => {
  const [selectedDataset, setSelectedDataset] = useState('iris_binary');
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [plots, setPlots] = useState({});
  const [loading, setLoading] = useState(false);
  const [nSamples, setNSamples] = useState(1000);
  const [noiseLevel, setNoiseLevel] = useState(0.1);
  const [featureEngineering, setFeatureEngineering] = useState(null);
  const [handleImbalance, setHandleImbalance] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const datasetOptions = [
    { value: 'circles', label: 'Concentric Circles', category: 'Synthetic' },
    { value: 'moons', label: 'Interleaving Moons', category: 'Synthetic' },
    { value: 'blobs', label: 'Gaussian Blobs', category: 'Synthetic' },
    { value: 'classification', label: 'Random Classification', category: 'Synthetic' },
    { value: 'spiral', label: 'Spiral', category: 'Synthetic' },
    { value: 'xor', label: 'XOR Problem', category: 'Synthetic' },
    { value: 'gaussian_quantum', label: 'Gaussian Quantum', category: 'Synthetic' },
    { value: 'iris_binary', label: 'Iris (Binary)', category: 'Real-World' },
    { value: 'wine_binary', label: 'Wine (Binary)', category: 'Real-World' },
    { value: 'breast_cancer', label: 'Breast Cancer', category: 'Real-World' },
    { value: 'digits_binary', label: 'Digits (Binary)', category: 'Real-World' },
  ];

  const featureEngineeringOptions = [
    { value: null, label: 'None' },
    { value: 'polynomial', label: 'Polynomial Features' },
    { value: 'pca', label: 'PCA' },
    { value: 'feature_selection', label: 'Feature Selection' },
  ];

  const imbalanceOptions = [
    { value: null, label: 'None' },
    { value: 'oversample', label: 'Oversample Minority' },
  ];

  useEffect(() => {
    loadDatasetPreview();
  }, [selectedDataset, nSamples, noiseLevel, featureEngineering, handleImbalance]);

  const loadDatasetPreview = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/enhanced_dataset_preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          datasetName: selectedDataset,
          nSamples,
          noiseLevel,
          featureEngineering,
          handleImbalance
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to load dataset preview');
      }

      const data = await response.json();
      setDatasetInfo(data.dataset_info);
      setPlots(data.plots);
      
      // Notify parent component
      if (onDatasetSelect) {
        onDatasetSelect({
          datasetName: selectedDataset,
          nSamples,
          noiseLevel,
          featureEngineering,
          handleImbalance,
          datasetInfo: data.dataset_info
        });
      }
    } catch (error) {
      console.error('Error loading dataset preview:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderImagePlot = (base64Image, title) => {
    if (!base64Image) return null;
    
    return (
      <div className="mb-6">
        <h4 className="text-lg font-semibold mb-3">{title}</h4>
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <img 
            src={`data:image/png;base64,${base64Image}`} 
            alt={title}
            className="w-full h-auto max-w-4xl mx-auto"
            style={{ maxHeight: '500px', objectFit: 'contain' }}
          />
        </div>
      </div>
    );
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
        <h3 className="text-xl font-bold text-gray-800 mb-2">{datasetInfo?.name} Dataset</h3>
        <p className="text-gray-600 mb-4">{datasetInfo?.description}</p>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl font-bold text-blue-600">{datasetInfo?.metadata?.n_samples}</div>
            <div className="text-sm text-gray-500">Samples</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl font-bold text-green-600">{datasetInfo?.metadata?.n_features}</div>
            <div className="text-sm text-gray-500">Features</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl font-bold text-purple-600">{datasetInfo?.metadata?.n_classes}</div>
            <div className="text-sm text-gray-500">Classes</div>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-sm">
            <div className="text-2xl font-bold text-red-600">{datasetInfo?.metadata?.missing_values || 0}</div>
            <div className="text-sm text-gray-500">Missing</div>
          </div>
        </div>
      </div>

      {datasetInfo?.metadata?.dataset_complexity && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h4 className="text-lg font-semibold mb-4">Dataset Complexity</h4>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Linear Separability</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${(datasetInfo.metadata.dataset_complexity.linear_separability * 100)}%` }}
                  ></div>
                </div>
                <span className="text-sm text-gray-600">
                  {(datasetInfo.metadata.dataset_complexity.linear_separability * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Feature Overlap</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-red-600 h-2 rounded-full" 
                    style={{ width: `${(datasetInfo.metadata.dataset_complexity.feature_overlap * 100)}%` }}
                  ></div>
                </div>
                <span className="text-sm text-gray-600">
                  {(datasetInfo.metadata.dataset_complexity.feature_overlap * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Overall Complexity</span>
              <div className="flex items-center space-x-2">
                <div className="w-32 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full" 
                    style={{ width: `${(datasetInfo.metadata.dataset_complexity.overall * 100)}%` }}
                  ></div>
                </div>
                <span className="text-sm text-gray-600">
                  {(datasetInfo.metadata.dataset_complexity.overall * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h4 className="text-lg font-semibold mb-4">Features</h4>
        <div className="flex flex-wrap gap-2">
          {datasetInfo?.feature_names?.map((name, index) => (
            <Badge key={index} variant="secondary" className="text-xs">
              {name}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full space-y-6">
      {/* Dataset Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Dataset Selection & Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Dataset Grid */}
          <div>
            <h4 className="text-sm font-medium mb-3">Choose Dataset</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {datasetOptions.map(option => (
                <button
                  key={option.value}
                  onClick={() => setSelectedDataset(option.value)}
                  className={`p-3 text-left rounded-lg border transition-all ${
                    selectedDataset === option.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  <div className="font-medium text-sm">{option.label}</div>
                  <div className="text-xs text-gray-500 mt-1">{option.category}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Sample Size: {nSamples}
              </label>
              <input
                type="range"
                min="100"
                max="5000"
                step="100"
                value={nSamples}
                onChange={(e) => setNSamples(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Noise Level: {noiseLevel.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={noiseLevel}
                onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Feature Engineering</label>
              <select
                value={featureEngineering || ''}
                onChange={(e) => setFeatureEngineering(e.target.value || null)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                {featureEngineeringOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Handle Imbalance</label>
              <select
                value={handleImbalance || ''}
                onChange={(e) => setHandleImbalance(e.target.value || null)}
                className="w-full p-2 border border-gray-300 rounded-md"
              >
                {imbalanceOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Dataset Preview */}
      {datasetInfo && (
        <Card>
          <CardHeader>
            <CardTitle>Dataset Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Tabs */}
            <div className="border-b border-gray-200 mb-6">
              <nav className="-mb-px flex space-x-8">
                {[
                  { id: 'overview', label: 'Overview' },
                  { id: 'distributions', label: 'Distributions' },
                  { id: 'class_distribution', label: 'Classes' },
                  { id: 'correlation', label: 'Correlation' },
                  { id: 'complexity', label: 'Complexity' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-1 border-b-2 font-medium text-sm ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="tab-content">
              {activeTab === 'overview' && renderOverview()}
              {activeTab === 'distributions' && renderImagePlot(plots.feature_distributions, 'Feature Distributions')}
              {activeTab === 'class_distribution' && renderImagePlot(plots.class_distribution, 'Class Distribution')}
              {activeTab === 'correlation' && renderImagePlot(plots.correlation_matrix, 'Feature Correlation Matrix')}
              {activeTab === 'complexity' && renderImagePlot(plots.complexity_metrics, 'Dataset Complexity Metrics')}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default EnhancedDatasetPreview;