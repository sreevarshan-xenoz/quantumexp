import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import DatasetVisualizer from '../components/DatasetVisualizer';
import QuantumCircuitVisualizer from '../components/QuantumCircuitVisualizer';
import ModelSelector from '../components/ModelSelector';
import ParameterSlider from '../components/ParameterSlider';
import ProgressIndicator from '../components/ProgressIndicator';
import SimulationVisualizationTabs from '../components/SimulationVisualizationTabs';
import HyperparameterOptimizer from '../components/HyperparameterOptimizer';
import EnhancedDatasetPreview from '../components/EnhancedDatasetPreview';
import DatasetComparison from '../components/DatasetComparison';
import DatasetSelector from '../components/DatasetSelector';
import { runSimulation, generateDatasetPreview, optimizeHyperparameters } from '../api/simulation';

const Simulation = () => {
  const navigate = useNavigate();
  
  // Simulation parameters
  const [datasetType, setDatasetType] = useState('circles');
  const [noiseLevel, setNoiseLevel] = useState(0.2);
  const [sampleSize, setSampleSize] = useState(1000);
  const [quantumFramework, setQuantumFramework] = useState('qiskit');
  const [quantumModel, setQuantumModel] = useState('vqc');
  
  // Enhanced dataset preview state
  const [showDatasetPreview, setShowDatasetPreview] = useState(true);
  const [showDatasetComparison, setShowDatasetComparison] = useState(false);
  const [previewDatasets, setPreviewDatasets] = useState({});
  const [selectedDatasetFromPreview, setSelectedDatasetFromPreview] = useState(null);
  const [datasetConfig, setDatasetConfig] = useState({
    datasetName: 'iris_binary',
    nSamples: 1000,
    noiseLevel: 0.1,
    featureEngineering: null,
    handleImbalance: null
  });

  // Handle dataset selection from preview
  const handleDatasetFromPreview = (datasetSelection) => {
    setSelectedDatasetFromPreview(datasetSelection);
    setDatasetConfig(datasetSelection);
    if (datasetSelection.datasetName) {
      setDatasetType(datasetSelection.datasetName);
    }
    if (datasetSelection.nSamples) {
      setSampleSize(datasetSelection.nSamples);
    }
    if (datasetSelection.noiseLevel !== undefined) {
      setNoiseLevel(datasetSelection.noiseLevel);
    }
  };
  const [classicalModel, setClassicalModel] = useState('logistic');
  const [featureMap, setFeatureMap] = useState('zz');
  
  // UI state
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [showOptimizer, setShowOptimizer] = useState(false);
  
  const progressInterval = useRef(null);

  // Generate dataset preview when parameters change
  useEffect(() => {
    const generatePreview = async () => {
      setIsLoadingPreview(true);
      try {
        const previewData = await generateDatasetPreview({
          datasetType,
          noiseLevel,
          sampleSize: Math.min(sampleSize, 200) // Limit preview size
        });
        setDatasetPreview(previewData);
      } catch (error) {
        console.error('Error generating preview:', error);
      } finally {
        setIsLoadingPreview(false);
      }
    };

    generatePreview();
  }, [datasetType, noiseLevel, sampleSize]);

  const handleRunSimulation = async () => {
    setIsRunning(true);
    setProgress(0);

    // Simulate progress updates
    progressInterval.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) {
          clearInterval(progressInterval.current);
          return prev;
        }
        return prev + Math.random() * 10 + 2;
      });
    }, 500);

    try {
      const results = await runSimulation({
        datasetType: datasetConfig.datasetName || datasetType,
        noiseLevel: datasetConfig.noiseLevel || noiseLevel,
        sampleSize: datasetConfig.nSamples || sampleSize,
        featureEngineering: datasetConfig.featureEngineering,
        handleImbalance: datasetConfig.handleImbalance,
        quantumFramework,
        quantumModel,
        classicalModel,
        featureMap
      });

      clearInterval(progressInterval.current);
      setProgress(100);

      // Navigate to results page with simulation data
      setTimeout(() => {
        navigate('/results', { 
          state: { 
            results: results.results,
            plots: results.plots,
            parameters: { 
              datasetType, 
              noiseLevel, 
              sampleSize, 
              quantumFramework,
              quantumModel, 
              classicalModel,
              featureMap 
            } 
          } 
        });
      }, 1000);
    } catch (error) {
      console.error('Simulation error:', error);
      clearInterval(progressInterval.current);
      setIsRunning(false);
      setProgress(0);
    }
  };

  const handleOptimizeHyperparameters = async (optimizationConfig) => {
    setIsOptimizing(true);
    
    try {
      const results = await optimizeHyperparameters({
        datasetType: datasetConfig.datasetName || datasetType,
        noiseLevel: datasetConfig.noiseLevel || noiseLevel,
        sampleSize: datasetConfig.nSamples || sampleSize,
        featureEngineering: datasetConfig.featureEngineering,
        handleImbalance: datasetConfig.handleImbalance,
        quantumFramework,
        quantumModel,
        classicalModel,
        featureMap,
        ...optimizationConfig
      });
      
      return results;
    } catch (error) {
      console.error('Optimization error:', error);
      throw error;
    } finally {
      setIsOptimizing(false);
    }
  };

  const datasetOptions = [
    { id: 'circles', name: 'Circles', description: 'Concentric circles' },
    { id: 'moons', name: 'Moons', description: 'Crescent shapes' },
    { id: 'blobs', name: 'Blobs', description: 'Gaussian clusters' }
  ];

  const featureMapOptions = [
    { id: 'zz', name: 'ZZ', description: 'ZZ Feature Map' },
    { id: 'z', name: 'Z', description: 'Z Feature Map' },
    { id: 'pauli', name: 'Pauli', description: 'Basic Pauli Map' },
    { id: 'pauli_full', name: 'Pauli Full', description: 'Full Pauli Feature Map' },
    { id: 'second_order', name: '2nd Order', description: 'Second Order Expansion' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold gradient-text mb-4">
            Quantum-Classical ML Simulation
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg max-w-2xl mx-auto">
            Configure parameters and run the simulation to compare quantum and classical machine learning models
          </p>
        </div>

        {/* Enhanced Dataset Preview Section */}
        {showDatasetPreview && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
                Dataset Explorer
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={() => setShowDatasetComparison(!showDatasetComparison)}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
                >
                  {showDatasetComparison ? 'Hide Comparison' : 'Compare Datasets'}
                </button>
                <button
                  onClick={() => setShowDatasetPreview(false)}
                  className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                >
                  Hide Preview
                </button>
              </div>
            </div>
            
            <EnhancedDatasetPreview
              onDatasetSelect={handleDatasetFromPreview}
            />
            
            {showDatasetComparison && (
              <div className="mt-6">
                <DatasetComparison
                  datasets={previewDatasets}
                  onDatasetSelect={(dataset, name) => {
                    setSelectedDatasetFromPreview(dataset);
                    setDatasetType(name);
                    if (dataset && dataset.X) {
                      setSampleSize(dataset.X.length);
                    }
                  }}
                />
              </div>
            )}
          </div>
        )}

        {/* Show Dataset Preview Button when hidden */}
        {!showDatasetPreview && (
          <div className="mb-8 text-center">
            <button
              onClick={() => setShowDatasetPreview(true)}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all shadow-lg"
            >
              🔍 Show Enhanced Dataset Preview
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="animate-slide-up">
              <DatasetSelector
                onDatasetChange={handleDatasetFromPreview}
                initialDataset={datasetConfig.datasetName}
              />
            </div>

            {/* Enhanced Dataset Visualization */}
            <div className="animate-slide-up">
              <DatasetVisualizer
                data={datasetPreview?.data}
                datasetType={datasetType}
                isLoading={isLoadingPreview}
                datasetInfo={selectedDatasetFromPreview?.datasetInfo}
                showEnhancedView={showDatasetPreview}
                onToggleEnhancedView={() => setShowDatasetPreview(!showDatasetPreview)}
              />
            </div>

            <div className="card animate-slide-up">
              <h2 className="text-xl font-semibold mb-6 flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
                Model Selection
              </h2>
              
              <div className="mb-6">
                <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                  Quantum Framework
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'qiskit', name: 'Qiskit', description: 'IBM Quantum Framework' },
                    { id: 'pennylane', name: 'PennyLane', description: 'Xanadu Quantum ML' }
                  ].map(framework => (
                    <button
                      key={framework.id}
                      className={`p-3 rounded-lg text-left transition-all duration-200 ${
                        quantumFramework === framework.id 
                          ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg' 
                          : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
                      }`}
                      onClick={() => setQuantumFramework(framework.id)}
                    >
                      <div className="font-medium text-sm">{framework.name}</div>
                      <div className={`text-xs ${quantumFramework === framework.id ? 'text-indigo-100' : 'text-gray-500 dark:text-gray-400'}`}>
                        {framework.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <ModelSelector 
                quantumModel={quantumModel}
                classicalModel={classicalModel}
                onQuantumChange={setQuantumModel}
                onClassicalChange={setClassicalModel}
              />

              <div className="mt-6">
                <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center">
                  <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                  Feature Map
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {featureMapOptions.map(option => (
                    <button
                      key={option.id}
                      className={`p-2 rounded-lg text-left transition-all duration-200 ${
                        featureMap === option.id 
                          ? 'bg-gradient-to-r from-purple-500 to-indigo-500 text-white shadow-lg' 
                          : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
                      }`}
                      onClick={() => setFeatureMap(option.id)}
                    >
                      <div className="font-medium text-xs">{option.name}</div>
                      <div className={`text-xs ${featureMap === option.id ? 'text-purple-100' : 'text-gray-500 dark:text-gray-400'} truncate`}>
                        {option.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="card animate-slide-up space-y-4">
              <button
                className={`w-full py-4 rounded-xl font-semibold text-lg transition-all duration-300 ${
                  isRunning 
                    ? 'bg-gray-400 cursor-not-allowed' 
                    : 'btn-primary transform hover:scale-105'
                }`}
                onClick={handleRunSimulation}
                disabled={isRunning}
              >
                {isRunning ? (
                  <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Running Simulation...</span>
                  </div>
                ) : (
                  'Run Simulation'
                )}
              </button>

              <button
                className="w-full py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
                onClick={() => setShowOptimizer(!showOptimizer)}
              >
                <div className="flex items-center justify-center space-x-2">
                  <span>🚀</span>
                  <span>{showOptimizer ? 'Hide' : 'Show'} Hyperparameter Optimizer</span>
                </div>
              </button>
            </div>
          </div>

          {/* Right Panel - Visualizations */}
          <div className="lg:col-span-2 space-y-8">
            <div className="animate-slide-up">
              <SimulationVisualizationTabs 
                datasetPreview={datasetPreview}
                datasetType={datasetType}
                isLoadingPreview={isLoadingPreview}
                quantumModel={quantumModel}
                featureMap={featureMap}
              />
            </div>

            {showOptimizer && (
              <div className="animate-slide-up">
                <HyperparameterOptimizer 
                  onOptimize={handleOptimizeHyperparameters}
                  isOptimizing={isOptimizing}
                />
              </div>
            )}

            {isRunning && (
              <div className="card animate-slide-up">
                <h2 className="text-xl font-semibold mb-4 flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-3 animate-pulse"></div>
                  Simulation Progress
                </h2>
                <ProgressIndicator progress={progress} />
                
                {progress > 80 && (
                  <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span className="text-sm text-blue-700 dark:text-blue-300">
                        Generating comprehensive visualizations...
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Simulation;