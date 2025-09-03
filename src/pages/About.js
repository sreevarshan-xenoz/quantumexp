import React from 'react';

const About = () => {
  const features = [
    {
      icon: '🔬',
      title: 'Quantum Algorithms',
      description: 'Variational Quantum Classifiers (VQC) and Quantum Support Vector Classifiers (QSVC) with multiple feature maps.'
    },
    {
      icon: '🤖',
      title: 'Classical Models',
      description: 'Comprehensive suite of classical ML algorithms including Logistic Regression, Random Forest, SVM, and XGBoost.'
    },
    {
      icon: '🔄',
      title: 'Hybrid Approach',
      description: 'Innovative hybrid models that combine quantum feature extraction with classical machine learning techniques.'
    },
    {
      icon: '📊',
      title: 'Interactive Visualization',
      description: 'Real-time dataset visualization, quantum circuit diagrams, and decision boundary plots.'
    },
    {
      icon: '⚡',
      title: 'Performance Analysis',
      description: 'Detailed comparison of accuracy, precision, recall, F1-score, and training time across all models.'
    },
    {
      icon: '🎛️',
      title: 'Parameter Control',
      description: 'Fine-tune dataset parameters, noise levels, model configurations, and quantum circuit designs.'
    }
  ];

  const quantumFeatures = [
    {
      name: 'Variational Quantum Classifier (VQC)',
      description: 'Parameterized quantum circuits optimized for classification tasks'
    },
    {
      name: 'Quantum Support Vector Machine (QSVM)',
      description: 'Quantum kernel methods for high-dimensional feature mapping'
    },
    {
      name: 'Quantum Neural Networks (QNN)',
      description: 'Neural network architectures implemented on quantum circuits'
    },
    {
      name: 'Quantum Kernel SVM',
      description: 'Advanced quantum kernel methods with custom feature maps'
    },
    {
      name: 'Multiple Feature Maps',
      description: 'ZZ, Z, Pauli, and second-order expansion feature encodings'
    }
  ];

  const datasets = [
    {
      name: 'Circles',
      description: 'Concentric circles with adjustable noise and separation'
    },
    {
      name: 'Moons',
      description: 'Two interleaving half-moon shapes'
    },
    {
      name: 'Blobs',
      description: 'Gaussian clusters with configurable separation'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-4xl font-bold gradient-text mb-4">
            About Quantum ML Simulator
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg max-w-3xl mx-auto">
            An advanced platform for exploring and comparing quantum and classical machine learning algorithms 
            through interactive simulations and comprehensive performance analysis.
          </p>
        </div>

        {/* Overview */}
        <div className="card mb-8 animate-slide-up">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Platform Overview
          </h2>
          <p className="text-gray-600 dark:text-gray-400 leading-relaxed mb-4">
            This simulation platform provides researchers, students, and practitioners with a comprehensive 
            environment to explore the capabilities and limitations of quantum machine learning algorithms 
            compared to their classical counterparts.
          </p>
          <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
            Built with cutting-edge quantum computing frameworks and modern web technologies, the platform 
            offers real-time visualization, parameter tuning, and detailed performance analysis across 
            multiple datasets and model configurations.
          </p>
        </div>

        {/* Key Features */}
        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 text-center">
            Key Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="card animate-slide-up hover:shadow-xl transition-all duration-300"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="text-3xl mb-3">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <div className="card animate-slide-up">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
              Quantum Algorithms
            </h3>
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {quantumFeatures.map((feature, index) => (
                <div key={index} className="border-l-4 border-green-500 pl-4">
                  <div className="font-medium text-gray-800 dark:text-gray-200 text-sm">
                    {feature.name}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {feature.description}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card animate-slide-up">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-3"></div>
              Classical Algorithms
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {[
                'Logistic Regression', 'Support Vector Machines', 'Random Forest',
                'XGBoost', 'Gradient Boosting', 'AdaBoost', 'Extra Trees',
                'K-Nearest Neighbors', 'Naive Bayes', 'Decision Trees',
                'Multi-Layer Perceptron', 'Ridge Classifier', 'SGD Classifier',
                'Linear/Quadratic Discriminant Analysis'
              ].map((algorithm, index) => (
                <div key={index} className="border-l-4 border-blue-500 pl-4">
                  <div className="font-medium text-gray-800 dark:text-gray-200 text-sm">
                    {algorithm}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card animate-slide-up">
            <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center">
              <div className="w-3 h-3 bg-purple-500 rounded-full mr-3"></div>
              Dataset Types
            </h3>
            <div className="space-y-3">
              {datasets.map((dataset, index) => (
                <div key={index} className="border-l-4 border-purple-500 pl-4">
                  <div className="font-medium text-gray-800 dark:text-gray-200">
                    {dataset.name}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {dataset.description}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="card mb-8 animate-slide-up">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Technology Stack
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Frontend</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• React 18 with modern hooks</li>
                <li>• Tailwind CSS for styling</li>
                <li>• D3.js for data visualization</li>
                <li>• React Router for navigation</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">Quantum Computing</h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <li>• Qiskit quantum framework</li>
                <li>• Qiskit Machine Learning</li>
                <li>• Variational quantum algorithms</li>
                <li>• Quantum kernel methods</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Usage Guide */}
        <div className="card mb-8 animate-slide-up">
          <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-4">
            How to Use
          </h3>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                1
              </div>
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">Configure Dataset</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Choose dataset type, adjust noise level, and set sample size
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                2
              </div>
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">Select Models</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Pick quantum and classical models, choose feature maps
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                3
              </div>
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">Run Simulation</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Execute the simulation and monitor progress in real-time
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                4
              </div>
              <div>
                <div className="font-medium text-gray-800 dark:text-gray-200">Analyze Results</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Compare performance metrics and visualize decision boundaries
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center animate-slide-up">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-4">Ready to Explore Quantum ML?</h3>
            <p className="mb-6 opacity-90">
              Start your journey into quantum machine learning with our interactive simulation platform
            </p>
            <button
              onClick={() => window.location.href = '/'}
              className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Start Simulation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;