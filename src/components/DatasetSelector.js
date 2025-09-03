import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

const DatasetSelector = ({ onDatasetChange, initialDataset = 'iris_binary' }) => {
  const [selectedDataset, setSelectedDataset] = useState(initialDataset);
  const [nSamples, setNSamples] = useState(1000);
  const [noise, setNoise] = useState(0.1);
  const [featureEngineering, setFeatureEngineering] = useState(null);
  const [handleImbalance, setHandleImbalance] = useState(null);

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
    handleDatasetChange();
  }, [selectedDataset, nSamples, noise, featureEngineering, handleImbalance]);

  const handleDatasetChange = () => {
    if (onDatasetChange) {
      onDatasetChange({
        datasetName: selectedDataset,
        nSamples,
        noiseLevel: noise,
        featureEngineering,
        handleImbalance
      });
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Dataset Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Dataset Selection */}
        <div>
          <label className="block text-sm font-medium mb-3">Dataset</label>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
              Noise Level: {noise.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="0.5"
              step="0.05"
              value={noise}
              onChange={(e) => setNoise(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* Advanced Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
  );
};

export default DatasetSelector;