import React from 'react';

const OptimizationResults = ({ results, onClose }) => {
  if (!results || !results.optimization_results) return null;

  const { optimization_results, dataset_info = {}, optimization_config = {} } = results;

  const ModelOptimizationCard = ({ modelType, modelName, result, color }) => {
    if (result.error) {
      return (
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-red-200 dark:border-red-700">
          <h4 className="font-semibold text-red-600 dark:text-red-400 mb-2 capitalize">
            {modelName} - Error
          </h4>
          <p className="text-sm text-red-500 dark:text-red-400">{typeof result.error === 'string' ? result.error : JSON.stringify(result.error)}</p>
        </div>
      );
    }

    const improvement = result.test_score && result.validation_score 
      ? ((result.test_score - 0.5) / 0.5 * 100).toFixed(1)
      : 'N/A';

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200 capitalize">
            {modelName}
          </h4>
          <div className={`px-2 py-1 rounded-full bg-gradient-to-r ${color} text-white text-xs font-medium`}>
            {modelType}
          </div>
        </div>

        <div className="space-y-3">
          {/* Scores */}
          <div className="grid grid-cols-3 gap-2 text-sm">
            <div className="text-center p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
              <div className="font-medium text-blue-800 dark:text-blue-200">CV Score</div>
              <div className="text-blue-600 dark:text-blue-400">
                {(result.best_score * 100).toFixed(1)}%
              </div>
            </div>
            <div className="text-center p-2 bg-green-50 dark:bg-green-900/20 rounded">
              <div className="font-medium text-green-800 dark:text-green-200">Val Score</div>
              <div className="text-green-600 dark:text-green-400">
                {result.validation_score ? (result.validation_score * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
            <div className="text-center p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
              <div className="font-medium text-purple-800 dark:text-purple-200">Test Score</div>
              <div className="text-purple-600 dark:text-purple-400">
                {result.test_score ? (result.test_score * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
          </div>

          {/* Best Parameters */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2 text-sm">
              Optimal Parameters
            </h5>
            <div className="space-y-1">
              {Object.entries(result.best_params || {}).map(([param, value]) => (
                <div key={param} className="flex justify-between text-xs">
                  <span className="text-gray-600 dark:text-gray-400 capitalize">
                    {param.replace('_', ' ')}:
                  </span>
                  <span className="font-medium text-gray-800 dark:text-gray-200">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Indicator */}
          {improvement !== 'N/A' && (
            <div className="flex items-center justify-center">
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                parseFloat(improvement) > 60 
                  ? 'bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300'
                  : parseFloat(improvement) > 40
                  ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300'
                  : 'bg-red-100 dark:bg-red-900/20 text-red-700 dark:text-red-300'
              }`}>
                {improvement}% performance
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
              🚀 Hyperparameter Optimization Results
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mt-1">
              Optimized using {optimization_config?.method?.replace('_', ' ') || 'unknown method'} with {optimization_config?.cv_folds || 'N/A'}-fold CV
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-8">
          {/* Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {optimization_config?.method?.replace('_', ' ').toUpperCase() || 'N/A'}
              </div>
              <div className="text-sm text-blue-500 dark:text-blue-300">Optimization Method</div>
            </div>
            <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {dataset_info?.samples?.toLocaleString() || 'N/A'}
              </div>
              <div className="text-sm text-green-500 dark:text-green-300">Total Samples</div>
            </div>
            <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {optimization_config?.cv_folds || 'N/A'}
              </div>
              <div className="text-sm text-purple-500 dark:text-purple-300">CV Folds</div>
            </div>
            <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {optimization_config?.scoring?.toUpperCase() || 'N/A'}
              </div>
              <div className="text-sm text-orange-500 dark:text-orange-300">Scoring Metric</div>
            </div>
          </div>

          {/* Classical Models Results */}
          {optimization_results.classical && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center">
                <div className="w-3 h-3 bg-blue-500 rounded-full mr-3"></div>
                Classical Models Optimization
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(optimization_results.classical).map(([modelName, result]) => (
                  <ModelOptimizationCard
                    key={modelName}
                    modelType="Classical"
                    modelName={modelName}
                    result={result}
                    color="from-blue-500 to-indigo-500"
                  />
                ))}
              </div>
            </div>
          )}

          {/* Quantum Models Results */}
          {optimization_results.quantum && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center">
                <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
                Quantum Models Optimization
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(optimization_results.quantum).map(([modelName, result]) => (
                  <ModelOptimizationCard
                    key={modelName}
                    modelType="Quantum"
                    modelName={modelName}
                    result={result}
                    color="from-green-500 to-emerald-500"
                  />
                ))}
              </div>
            </div>
          )}

          {/* Data Split Information */}
          <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
            <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3">
              📊 Data Split Information
            </h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="font-semibold text-blue-600 dark:text-blue-400">
                  {dataset_info?.train_size || 'N/A'}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Training</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-green-600 dark:text-green-400">
                  {dataset_info?.val_size || 'N/A'}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Validation</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-purple-600 dark:text-purple-400">
                  {dataset_info?.test_size || 'N/A'}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Test</div>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-6">
            <h4 className="font-medium text-gray-800 dark:text-gray-200 mb-3 flex items-center">
              <span className="mr-2">💡</span>
              Optimization Insights
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <h5 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Best Performing Models</h5>
                <ul className="text-blue-700 dark:text-blue-300 space-y-1">
                  {optimization_results.classical && Object.entries(optimization_results.classical)
                    .filter(([_, result]) => !result.error)
                    .sort((a, b) => (b[1].test_score || b[1].best_score) - (a[1].test_score || a[1].best_score))
                    .slice(0, 3)
                    .map(([modelName, result]) => (
                      <li key={modelName}>
                        • {modelName}: {((result.test_score || result.best_score) * 100).toFixed(1)}%
                      </li>
                    ))}
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-purple-800 dark:text-purple-200 mb-2">Key Findings</h5>
                <ul className="text-purple-700 dark:text-purple-300 space-y-1">
                  <li>• Optimization method: {optimization_config?.method?.replace('_', ' ') || 'N/A'}</li>
                  <li>• Cross-validation: {optimization_config?.cv_folds || 'N/A'} folds</li>
                  <li>• Scoring metric: {optimization_config?.scoring || 'N/A'}</li>
                  <li>• Dataset: {dataset_info?.type || 'N/A'} ({dataset_info?.samples || 'N/A'} samples)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            Close Results
          </button>
        </div>
      </div>
    </div>
  );
};

export default OptimizationResults;