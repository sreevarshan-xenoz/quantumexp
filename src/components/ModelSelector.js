import React from 'react';

const ModelSelector = ({ quantumModel, classicalModel, onQuantumChange, onClassicalChange }) => {
  const quantumModels = [
    { id: 'vqc', name: 'VQC', description: 'Variational Quantum Classifier' },
    { id: 'qsvc', name: 'QSVC', description: 'Quantum Support Vector Classifier' }
  ];

  const classicalModels = [
    { id: 'logistic', name: 'Logistic', description: 'Logistic Regression' },
    { id: 'random_forest', name: 'Random Forest', description: 'Random Forest Classifier' },
    { id: 'svm', name: 'SVM', description: 'Support Vector Machine' },
    { id: 'xgboost', name: 'XGBoost', description: 'Gradient Boosting' }
  ];

  const featureMaps = [
    { id: 'zz', name: 'ZZ', description: 'ZZ Feature Map' },
    { id: 'z', name: 'Z', description: 'Z Feature Map' },
    { id: 'pauli', name: 'Pauli', description: 'Pauli Feature Map' }
  ];

  return (
    <div className="space-y-6 animate-slide-up">
      <div>
        <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center">
          <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
          Quantum Model
        </h3>
        <div className="grid grid-cols-1 gap-2">
          {quantumModels.map(model => (
            <button
              key={model.id}
              className={`p-3 rounded-lg text-left transition-all duration-200 ${
                quantumModel === model.id 
                  ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg' 
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
              }`}
              onClick={() => onQuantumChange(model.id)}
            >
              <div className="font-medium">{model.name}</div>
              <div className={`text-sm ${quantumModel === model.id ? 'text-green-100' : 'text-gray-500 dark:text-gray-400'}`}>
                {model.description}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div>
        <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-3 flex items-center">
          <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
          Classical Model
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {classicalModels.map(model => (
            <button
              key={model.id}
              className={`p-3 rounded-lg text-left transition-all duration-200 ${
                classicalModel === model.id 
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg' 
                  : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300'
              }`}
              onClick={() => onClassicalChange(model.id)}
            >
              <div className="font-medium text-sm">{model.name}</div>
              <div className={`text-xs ${classicalModel === model.id ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}`}>
                {model.description}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ModelSelector;