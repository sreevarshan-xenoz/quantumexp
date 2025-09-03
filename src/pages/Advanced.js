import React, { useState } from 'react';
import HyperparameterOptimizer from '../components/HyperparameterOptimizer';
import OptimizationResults from '../components/OptimizationResults';
import QuantumHardwareManager from '../components/QuantumHardwareManager';
import AdvancedQuantumAlgorithms from '../components/AdvancedQuantumAlgorithms';
import HybridOptimization from '../components/HybridOptimization';
import AdvancedGraphAnalytics from '../components/AdvancedGraphAnalytics';
import QuantumAnalyticsDashboard from '../components/QuantumAnalyticsDashboard';
import PerformanceComparisonGraphs from '../components/PerformanceComparisonGraphs';
import { optimizeHyperparameters } from '../api/simulation';

const Advanced = () => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [showResults, setShowResults] = useState(false);

  const handleOptimization = async (config) => {
    setIsOptimizing(true);
    try {
      const results = await optimizeHyperparameters(config);
      setOptimizationResults(results);
      setShowResults(true);
    } catch (error) {
      console.error('Optimization failed:', error);
      // You could add error handling UI here
    } finally {
      setIsOptimizing(false);
    }
  };

  const features = [
    {
      icon: '🚀',
      title: 'Hyperparameter Optimization',
      description: 'Automated parameter tuning using Grid Search, Random Search, or Bayesian optimization',
      status: 'Available'
    },
    {
      icon: '⚛️',
      title: 'Quantum Hardware Integration',
      description: 'Connect to real quantum computers from IBM, IonQ, and other providers',
      status: 'Available'
    },
    {
      icon: '🛡️',
      title: 'Quantum Error Mitigation',
      description: 'Advanced error correction and noise mitigation for quantum hardware',
      status: 'Available'
    },
    {
      icon: '🔬',
      title: 'Advanced Quantum Algorithms',
      description: 'QAOA, VQE, QNN, QFT, and QPE for cutting-edge quantum computing',
      status: 'Available'
    },
    {
      icon: '🔄',
      title: 'Hybrid Optimization',
      description: 'Quantum-classical hybrid optimization with parameter shift and Bayesian methods',
      status: 'Available'
    },
    {
      icon: '📊',
      title: 'Advanced Graph Analytics',
      description: 'Interactive D3.js visualizations with 8+ graph types for deep analysis',
      status: 'Available'
    },
    {
      icon: '📈',
      title: 'Real-time Quantum Dashboard',
      description: 'Live quantum metrics monitoring with fidelity, entanglement, and coherence tracking',
      status: 'Available'
    },
    {
      icon: '⚖️',
      title: 'Performance Comparison Suite',
      description: 'Multi-dimensional algorithm comparison with scalability and efficiency analysis',
      status: 'Available'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold gradient-text mb-4">
            Advanced Features
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg max-w-3xl mx-auto">
            Cutting-edge tools for quantum-classical machine learning research and optimization
          </p>
        </div>

        {/* Feature Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <div 
              key={index}
              className={`card animate-slide-up hover:shadow-xl transition-all duration-300 ${
                feature.status === 'Available' ? 'border-l-4 border-green-500' : 'border-l-4 border-gray-300'
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="text-3xl">{feature.icon}</div>
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                  feature.status === 'Available' 
                    ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}>
                  {feature.status}
                </div>
              </div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Hyperparameter Optimization Section */}
        <div className="mb-12">
          <HyperparameterOptimizer 
            onOptimize={handleOptimization}
            isOptimizing={isOptimizing}
          />
        </div>

        {/* Quantum Hardware Integration Section */}
        <div className="mb-12">
          <QuantumHardwareManager />
        </div>

        {/* Advanced Quantum Algorithms Section */}
        <div className="mb-12">
          <AdvancedQuantumAlgorithms />
        </div>

        {/* Hybrid Optimization Section */}
        <div className="mb-12">
          <HybridOptimization />
        </div>

        {/* Advanced Graph Analytics Section */}
        <div className="mb-12">
          <AdvancedGraphAnalytics />
        </div>

        {/* Quantum Analytics Dashboard Section */}
        <div className="mb-12">
          <QuantumAnalyticsDashboard />
        </div>

        {/* Performance Comparison Graphs Section */}
        <div className="mb-12">
          <PerformanceComparisonGraphs />
        </div>

        {/* Advanced Analytics Preview */}
        <div className="card animate-slide-up">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-3"></div>
            Advanced Analytics Dashboard
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
              <div className="text-3xl mb-3">📈</div>
              <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Performance Tracking</h4>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                Monitor model performance across multiple runs and parameter configurations.
              </p>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
              <div className="text-3xl mb-3">🎯</div>
              <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">Quantum Advantage</h4>
              <p className="text-sm text-green-600 dark:text-green-400">
                Quantify and visualize quantum advantage across different problem types.
              </p>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
              <div className="text-3xl mb-3">🔬</div>
              <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">Explainability</h4>
              <p className="text-sm text-purple-600 dark:text-purple-400">
                Understand model decisions with advanced interpretability tools.
              </p>
            </div>
            
            <div className="text-center p-6 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg">
              <div className="text-3xl mb-3">⚡</div>
              <h4 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">Real-time Analysis</h4>
              <p className="text-sm text-orange-600 dark:text-orange-400">
                Live performance monitoring and parameter adjustment capabilities.
              </p>
            </div>
          </div>
        </div>

        {/* Research Tools */}
        <div className="card animate-slide-up">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center">
            <div className="w-3 h-3 bg-indigo-500 rounded-full mr-3"></div>
            Research & Development Tools
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">
                🧪 Experimental Features
              </h4>
              <ul className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Quantum circuit depth optimization
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Noise-aware quantum training
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></div>
                  Multi-qubit entanglement analysis
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></div>
                  Quantum advantage benchmarking
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mr-3"></div>
                  Quantum error correction integration
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">
                📚 Research Integration
              </h4>
              <ul className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Export results to research papers
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                  Reproducible experiment tracking
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></div>
                  Collaboration tools for research teams
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></div>
                  Integration with quantum hardware
                </li>
                <li className="flex items-center">
                  <div className="w-2 h-2 bg-gray-400 rounded-full mr-3"></div>
                  Automated literature comparison
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center animate-slide-up mt-12">
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">Ready to Push the Boundaries?</h3>
            <p className="mb-6 opacity-90 max-w-2xl mx-auto">
              These advanced features represent the cutting edge of quantum-classical machine learning research. 
              Start with hyperparameter optimization and explore the future of ML.
            </p>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => window.location.href = '/'}
                className="bg-white text-indigo-600 px-6 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Start Basic Simulation
              </button>
              <button
                onClick={() => document.querySelector('.hyperparameter-optimizer')?.scrollIntoView({ behavior: 'smooth' })}
                className="bg-indigo-700 text-white px-6 py-3 rounded-lg font-semibold hover:bg-indigo-800 transition-colors border border-indigo-500"
              >
                Try Optimization
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Optimization Results Modal */}
      {showResults && optimizationResults && (
        <OptimizationResults 
          results={optimizationResults}
          onClose={() => setShowResults(false)}
        />
      )}
    </div>
  );
};

export default Advanced;