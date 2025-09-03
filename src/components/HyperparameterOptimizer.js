import React, { useState } from 'react';

const HyperparameterOptimizer = ({ onOptimize, isOptimizing = false }) => {
  const [optimizationConfig, setOptimizationConfig] = useState({
    method: 'grid_search',
    cv_folds: 5,
    scoring: 'accuracy',
    n_trials: 50,
    timeout: 300, // 5 minutes
    optimize_classical: true,
    optimize_quantum: true,
    optimize_hybrid: true
  });

  const [parameterRanges, setParameterRanges] = useState({
    classical: {
      logistic: {
        C: { min: 0.01, max: 100, type: 'log' },
        max_iter: { min: 100, max: 2000, type: 'int' }
      },
      random_forest: {
        n_estimators: { min: 10, max: 200, type: 'int' },
        max_depth: { min: 3, max: 20, type: 'int' },
        min_samples_split: { min: 2, max: 20, type: 'int' }
      },
      svm: {
        C: { min: 0.1, max: 100, type: 'log' },
        gamma: { min: 0.001, max: 1, type: 'log' }
      },
      xgboost: {
        n_estimators: { min: 50, max: 300, type: 'int' },
        max_depth: { min: 3, max: 10, type: 'int' },
        learning_rate: { min: 0.01, max: 0.3, type: 'float' }
      }
    },
    quantum: {
      vqc: {
        reps: { min: 1, max: 5, type: 'int' },
        maxiter: { min: 50, max: 500, type: 'int' }
      },
      qsvc: {
        C: { min: 0.1, max: 100, type: 'log' }
      }
    }
  });

  const optimizationMethods = [
    { id: 'grid_search', name: 'Grid Search', description: 'Exhaustive search over parameter grid' },
    { id: 'random_search', name: 'Random Search', description: 'Random sampling of parameter space' },
    { id: 'bayesian', name: 'Bayesian Optimization', description: 'Smart parameter exploration using Gaussian processes' },
    { id: 'optuna', name: 'Optuna TPE', description: 'Tree-structured Parzen Estimator optimization' }
  ];

  const scoringMetrics = [
    { id: 'accuracy', name: 'Accuracy', description: 'Overall classification accuracy' },
    { id: 'f1', name: 'F1 Score', description: 'Harmonic mean of precision and recall' },
    { id: 'precision', name: 'Precision', description: 'True positives / (True positives + False positives)' },
    { id: 'recall', name: 'Recall', description: 'True positives / (True positives + False negatives)' },
    { id: 'roc_auc', name: 'ROC AUC', description: 'Area under the ROC curve' }
  ];

  const handleOptimizationStart = () => {
    const config = {
      ...optimizationConfig,
      parameter_ranges: parameterRanges
    };
    onOptimize(config);
  };

  const ParameterRangeEditor = ({ modelType, modelName, params, onChange }) => (
    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
      <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3 capitalize">
        {modelName} Parameters
      </h4>
      <div className="space-y-3">
        {Object.entries(params).map(([paramName, paramConfig]) => (
          <div key={paramName} className="grid grid-cols-4 gap-2 items-center">
            <label className="text-sm text-gray-600 dark:text-gray-400 capitalize">
              {paramName.replace('_', ' ')}
            </label>
            <input
              type="number"
              step={paramConfig.type === 'float' ? '0.01' : '1'}
              value={paramConfig.min}
              onChange={(e) => onChange(modelType, modelName, paramName, 'min', parseFloat(e.target.value))}
              className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
              placeholder="Min"
            />
            <input
              type="number"
              step={paramConfig.type === 'float' ? '0.01' : '1'}
              value={paramConfig.max}
              onChange={(e) => onChange(modelType, modelName, paramName, 'max', parseFloat(e.target.value))}
              className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
              placeholder="Max"
            />
            <select
              value={paramConfig.type}
              onChange={(e) => onChange(modelType, modelName, paramName, 'type', e.target.value)}
              className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800"
            >
              <option value="int">Integer</option>
              <option value="float">Float</option>
              <option value="log">Log Scale</option>
            </select>
          </div>
        ))}
      </div>
    </div>
  );

  const updateParameterRange = (modelType, modelName, paramName, field, value) => {
    setParameterRanges(prev => ({
      ...prev,
      [modelType]: {
        ...prev[modelType],
        [modelName]: {
          ...prev[modelType][modelName],
          [paramName]: {
            ...prev[modelType][modelName][paramName],
            [field]: value
          }
        }
      }
    }));
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gray-800 dark:text-gray-200 flex items-center">
          <div className="w-3 h-3 bg-orange-500 rounded-full mr-3"></div>
          Hyperparameter Optimization
        </h3>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Automated parameter tuning
        </div>
      </div>

      {/* Optimization Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Optimization Method
            </label>
            <select
              value={optimizationConfig.method}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, method: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {optimizationMethods.map(method => (
                <option key={method.id} value={method.id}>{method.name}</option>
              ))}
            </select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {optimizationMethods.find(m => m.id === optimizationConfig.method)?.description}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Scoring Metric
            </label>
            <select
              value={optimizationConfig.scoring}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, scoring: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {scoringMetrics.map(metric => (
                <option key={metric.id} value={metric.id}>{metric.name}</option>
              ))}
            </select>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              {scoringMetrics.find(m => m.id === optimizationConfig.scoring)?.description}
            </p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                CV Folds
              </label>
              <input
                type="number"
                min="3"
                max="10"
                value={optimizationConfig.cv_folds}
                onChange={(e) => setOptimizationConfig(prev => ({ ...prev, cv_folds: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Max Trials
              </label>
              <input
                type="number"
                min="10"
                max="200"
                value={optimizationConfig.n_trials}
                onChange={(e) => setOptimizationConfig(prev => ({ ...prev, n_trials: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Timeout (seconds)
            </label>
            <input
              type="number"
              min="60"
              max="3600"
              value={optimizationConfig.timeout}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, timeout: parseInt(e.target.value) }))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <div className="mb-6">
        <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3">
          Models to Optimize
        </h4>
        <div className="flex space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={optimizationConfig.optimize_classical}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, optimize_classical: e.target.checked }))}
              className="mr-2"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Classical Models</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={optimizationConfig.optimize_quantum}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, optimize_quantum: e.target.checked }))}
              className="mr-2"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Quantum Models</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={optimizationConfig.optimize_hybrid}
              onChange={(e) => setOptimizationConfig(prev => ({ ...prev, optimize_hybrid: e.target.checked }))}
              className="mr-2"
            />
            <span className="text-sm text-gray-700 dark:text-gray-300">Hybrid Models</span>
          </label>
        </div>
      </div>

      {/* Parameter Ranges */}
      <div className="mb-6">
        <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-4">
          Parameter Ranges
        </h4>
        
        <div className="space-y-6">
          {/* Classical Parameters */}
          {optimizationConfig.optimize_classical && (
            <div>
              <h5 className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-3">
                Classical Model Parameters
              </h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(parameterRanges.classical).map(([modelName, params]) => (
                  <ParameterRangeEditor
                    key={modelName}
                    modelType="classical"
                    modelName={modelName}
                    params={params}
                    onChange={updateParameterRange}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Quantum Parameters */}
          {optimizationConfig.optimize_quantum && (
            <div>
              <h5 className="text-sm font-medium text-green-600 dark:text-green-400 mb-3">
                Quantum Model Parameters
              </h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(parameterRanges.quantum).map(([modelName, params]) => (
                  <ParameterRangeEditor
                    key={modelName}
                    modelType="quantum"
                    modelName={modelName}
                    params={params}
                    onChange={updateParameterRange}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Start Optimization Button */}
      <div className="flex justify-center">
        <button
          onClick={handleOptimizationStart}
          disabled={isOptimizing}
          className={`px-8 py-3 rounded-xl font-semibold text-lg transition-all duration-300 ${
            isOptimizing
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white shadow-lg hover:shadow-xl transform hover:scale-105'
          }`}
        >
          {isOptimizing ? (
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Optimizing...</span>
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <span>🚀</span>
              <span>Start Optimization</span>
            </div>
          )}
        </button>
      </div>

      {/* Information Panel */}
      <div className="mt-6 p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
        <h4 className="font-medium text-orange-800 dark:text-orange-200 mb-2 flex items-center">
          <span className="mr-2">💡</span>
          Optimization Tips
        </h4>
        <ul className="text-sm text-orange-700 dark:text-orange-300 space-y-1">
          <li>• Start with fewer trials for quick exploration, then increase for fine-tuning</li>
          <li>• Bayesian optimization is most efficient for expensive evaluations</li>
          <li>• Grid search is thorough but can be slow with many parameters</li>
          <li>• Consider the trade-off between optimization time and performance gain</li>
          <li>• Quantum model optimization may take longer due to circuit complexity</li>
        </ul>
      </div>
    </div>
  );
};

export default HyperparameterOptimizer;