import React, { useState } from 'react';
import DatasetVisualizer from './DatasetVisualizer';
import ThreeJSVisualizer from './ThreeJSVisualizer';
import QuantumCircuitVisualizer from './QuantumCircuitVisualizer';

const SimulationVisualizationTabs = ({ 
  datasetPreview, 
  datasetType, 
  isLoadingPreview, 
  quantumModel, 
  featureMap,
  predictions = null 
}) => {
  const [activeTab, setActiveTab] = useState('dataset');

  const tabs = [
    { id: 'dataset', name: '2D Dataset', icon: '📊', description: 'Interactive 2D scatter plot' },
    { id: '3d', name: '3D View', icon: '🎯', description: 'Three.js 3D visualization' },
    { id: 'circuit', name: 'Quantum Circuit', icon: '⚛️', description: 'Quantum circuit diagram' }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 dark:border-gray-700">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-4 py-3 text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-b-2 border-blue-500'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700/50'
            }`}
            title={tab.description}
          >
            <div className="flex flex-col items-center space-y-1">
              <span className="text-lg">{tab.icon}</span>
              <span className="text-xs">{tab.name}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'dataset' && (
          <div className="animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                Dataset Preview
              </h3>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Interactive D3.js visualization
              </div>
            </div>
            <DatasetVisualizer 
              data={datasetPreview}
              datasetType={datasetType}
              isLoading={isLoadingPreview}
            />
          </div>
        )}

        {activeTab === '3d' && (
          <div className="animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                3D Decision Boundary
              </h3>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Three.js 3D rendering
              </div>
            </div>
            {datasetPreview ? (
              <ThreeJSVisualizer 
                data={datasetPreview}
                predictions={predictions}
                title={`3D View - ${datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Dataset`}
              />
            ) : (
              <div className="h-96 bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <div className="text-4xl mb-2">🎯</div>
                  <p className="text-gray-600 dark:text-gray-400 mb-2">
                    Generate dataset to view 3D visualization
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    Adjust parameters above and the preview will update automatically
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'circuit' && (
          <div className="animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                Quantum Circuit Diagram
              </h3>
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {featureMap.toUpperCase()} feature map
              </div>
            </div>
            <QuantumCircuitVisualizer 
              model={quantumModel} 
              featureMap={featureMap}
            />
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
                Circuit Information
              </h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-blue-600 dark:text-blue-400">Model:</span>
                  <span className="ml-2 font-medium">{quantumModel.toUpperCase()}</span>
                </div>
                <div>
                  <span className="text-blue-600 dark:text-blue-400">Feature Map:</span>
                  <span className="ml-2 font-medium">{featureMap.toUpperCase()}</span>
                </div>
                <div>
                  <span className="text-blue-600 dark:text-blue-400">Qubits:</span>
                  <span className="ml-2 font-medium">2 (for 2D data)</span>
                </div>
                <div>
                  <span className="text-blue-600 dark:text-blue-400">Depth:</span>
                  <span className="ml-2 font-medium">Variable</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SimulationVisualizationTabs;