import React from 'react';

const AdvancedAnalytics = ({ results }) => {
  if (!results) return null;

  const { feature_importance, quantum_advantage } = results;

  const FeatureImportanceChart = ({ importance, title, color }) => {
    if (!importance) return null;

    const maxImportance = Math.max(...importance);
    
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-4">{title}</h4>
        <div className="space-y-3">
          {importance.map((imp, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Feature {index + 1}</span>
                <span className="font-medium text-gray-800 dark:text-gray-200">
                  {(imp * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full bg-gradient-to-r ${color} transition-all duration-1000 ease-out`}
                  style={{ width: `${maxImportance > 0 ? (imp / maxImportance) * 100 : 0}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const QuantumAdvantageCard = ({ metric, value, title, description, isPositive }) => {
    const getColorClass = () => {
      if (value > 0.1) return isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
      if (value > 0) return 'text-yellow-600 dark:text-yellow-400';
      return isPositive ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400';
    };

    const getIcon = () => {
      if (value > 0.1) return isPositive ? '📈' : '📉';
      if (value > 0) return '➡️';
      return isPositive ? '📉' : '📈';
    };

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-semibold text-gray-800 dark:text-gray-200">{title}</h4>
          <span className="text-2xl">{getIcon()}</span>
        </div>
        <div className={`text-2xl font-bold mb-1 ${getColorClass()}`}>
          {value > 0 ? '+' : ''}{(value * 100).toFixed(1)}%
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* Feature Importance Analysis */}
      {feature_importance && (
        <div className="card animate-slide-up">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-3"></div>
            Feature Importance Analysis
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FeatureImportanceChart 
              importance={feature_importance.classical}
              title="Classical Model Feature Importance"
              color="from-blue-500 to-indigo-500"
            />
            <FeatureImportanceChart 
              importance={feature_importance.quantum}
              title="Quantum Model Feature Importance"
              color="from-green-500 to-emerald-500"
            />
          </div>
          
          <div className="mt-6 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
            <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">
              💡 Interpretation
            </h4>
            <p className="text-sm text-purple-700 dark:text-purple-300">
              Feature importance shows how much each input feature contributes to the model's predictions. 
              Higher values indicate more important features. Comparing classical and quantum importance 
              can reveal different feature utilization patterns.
            </p>
          </div>
        </div>
      )}

      {/* Quantum Advantage Analysis */}
      {quantum_advantage && (
        <div className="card animate-slide-up">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-3"></div>
            Quantum Advantage Analysis
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <QuantumAdvantageCard 
              metric="quantum_accuracy_advantage"
              value={quantum_advantage.quantum_accuracy_advantage}
              title="Quantum Accuracy"
              description="Accuracy improvement over classical"
              isPositive={true}
            />
            <QuantumAdvantageCard 
              metric="hybrid_accuracy_advantage"
              value={quantum_advantage.hybrid_accuracy_advantage}
              title="Hybrid Advantage"
              description="Hybrid model accuracy boost"
              isPositive={true}
            />
            <QuantumAdvantageCard 
              metric="quantum_time_efficiency"
              value={quantum_advantage.quantum_time_efficiency - 1}
              title="Time Efficiency"
              description="Training time comparison"
              isPositive={true}
            />
            <QuantumAdvantageCard 
              metric="quantum_overall_advantage"
              value={quantum_advantage.quantum_overall_advantage}
              title="Overall Score"
              description="Combined advantage metric"
              isPositive={true}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <h4 className="font-medium text-green-800 dark:text-green-200 mb-2 flex items-center">
                <span className="mr-2">🎯</span>
                Quantum Advantage Indicators
              </h4>
              <ul className="text-sm text-green-700 dark:text-green-300 space-y-1">
                <li>• Positive accuracy advantage indicates quantum benefit</li>
                <li>• Time efficiency > 1 means quantum is faster</li>
                <li>• Overall score combines accuracy and efficiency</li>
                <li>• Hybrid models often show the best performance</li>
              </ul>
            </div>
            
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2 flex items-center">
                <span className="mr-2">📊</span>
                Performance Summary
              </h4>
              <div className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
                <div>Quantum vs Classical: {quantum_advantage.quantum_accuracy_advantage > 0 ? 'Advantage' : 'Disadvantage'}</div>
                <div>Hybrid Performance: {quantum_advantage.hybrid_accuracy_advantage > 0 ? 'Superior' : 'Comparable'}</div>
                <div>Training Efficiency: {quantum_advantage.quantum_time_efficiency > 1 ? 'Faster' : 'Slower'}</div>
                <div>Recommendation: {quantum_advantage.hybrid_overall_advantage > 0.1 ? 'Use Hybrid' : quantum_advantage.quantum_overall_advantage > 0 ? 'Consider Quantum' : 'Use Classical'}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Comparison Insights */}
      <div className="card animate-slide-up">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-6 flex items-center">
          <div className="w-3 h-3 bg-orange-500 rounded-full mr-3"></div>
          Model Comparison Insights
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
            <div className="text-3xl mb-3">🤖</div>
            <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">Classical Models</h4>
            <p className="text-sm text-blue-600 dark:text-blue-400">
              Fast training, interpretable results, well-established algorithms with proven performance on classical data.
            </p>
          </div>
          
          <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg">
            <div className="text-3xl mb-3">⚛️</div>
            <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">Quantum Models</h4>
            <p className="text-sm text-green-600 dark:text-green-400">
              Novel feature representations, potential for quantum advantage, exploration of high-dimensional spaces.
            </p>
          </div>
          
          <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
            <div className="text-3xl mb-3">🔄</div>
            <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">Hybrid Models</h4>
            <p className="text-sm text-purple-600 dark:text-purple-400">
              Best of both worlds, quantum feature extraction with classical processing, often superior performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedAnalytics;