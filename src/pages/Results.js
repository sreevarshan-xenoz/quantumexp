import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  
  const { results, parameters } = location.state || {};

  if (!results) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-4">
            No Results Available
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Please run a simulation first to see results.
          </p>
          <button
            onClick={() => navigate('/')}
            className="btn-primary"
          >
            Go to Simulation
          </button>
        </div>
      </div>
    );
  }

  const modelTypes = ['classical', 'quantum', 'hybrid'];
  const modelNames = {
    classical: 'Classical Model',
    quantum: 'Quantum Model',
    hybrid: 'Hybrid Model'
  };

  const modelColors = {
    classical: 'from-blue-500 to-indigo-500',
    quantum: 'from-green-500 to-emerald-500',
    hybrid: 'from-purple-500 to-pink-500'
  };

  const MetricCard = ({ title, value, unit = '', color = 'blue' }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
      <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">{title}</div>
      <div className={`text-2xl font-bold text-${color}-600 dark:text-${color}-400`}>
        {typeof value === 'number' ? (unit === '%' ? `${(value * 100).toFixed(1)}%` : `${value.toFixed(3)}${unit}`) : value}
      </div>
    </div>
  );

  const ModelResultCard = ({ modelType, data }) => (
    <div className="card animate-slide-up">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
          {modelNames[modelType]}
        </h3>
        <div className={`px-3 py-1 rounded-full bg-gradient-to-r ${modelColors[modelType]} text-white text-sm font-medium`}>
          {(data.accuracy * 100).toFixed(1)}% Accuracy
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard title="Accuracy" value={data.accuracy} unit="%" />
        <MetricCard title="Precision" value={data.precision} unit="%" />
        <MetricCard title="Recall" value={data.recall} unit="%" />
        <MetricCard title="F1 Score" value={data.f1} unit="%" />
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <span className="text-sm text-gray-600 dark:text-gray-400">Training Time</span>
          <span className="font-medium text-gray-800 dark:text-gray-200">
            {data.training_time.toFixed(2)}s
          </span>
        </div>
      </div>
    </div>
  );

  const ComparisonChart = () => {
    const maxAccuracy = Math.max(...modelTypes.map(type => results[type].accuracy));
    
    return (
      <div className="card animate-slide-up">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-6">
          Model Comparison
        </h3>
        
        <div className="space-y-4">
          {modelTypes.map(modelType => {
            const accuracy = results[modelType].accuracy;
            const percentage = (accuracy / maxAccuracy) * 100;
            
            return (
              <div key={modelType} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-700 dark:text-gray-300">
                    {modelNames[modelType]}
                  </span>
                  <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                    {(accuracy * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full bg-gradient-to-r ${modelColors[modelType]} transition-all duration-1000 ease-out`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const ParameterSummary = () => (
    <div className="card animate-slide-up">
      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
        Simulation Parameters
      </h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Dataset</span>
          <div className="font-medium text-gray-800 dark:text-gray-200 capitalize">
            {parameters.datasetType}
          </div>
        </div>
        <div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Sample Size</span>
          <div className="font-medium text-gray-800 dark:text-gray-200">
            {parameters.sampleSize.toLocaleString()}
          </div>
        </div>
        <div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Noise Level</span>
          <div className="font-medium text-gray-800 dark:text-gray-200">
            {parameters.noiseLevel}
          </div>
        </div>
        <div>
          <span className="text-sm text-gray-600 dark:text-gray-400">Feature Map</span>
          <div className="font-medium text-gray-800 dark:text-gray-200 uppercase">
            {parameters.featureMap}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8 animate-fade-in">
          <h1 className="text-4xl font-bold gradient-text mb-4">
            Simulation Results
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Comprehensive analysis of quantum vs classical machine learning performance
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          <div className="lg:col-span-2">
            <div className="space-y-6">
              {modelTypes.map(modelType => (
                <ModelResultCard 
                  key={modelType}
                  modelType={modelType}
                  data={results[modelType]}
                />
              ))}
            </div>
          </div>
          
          <div className="space-y-6">
            <ComparisonChart />
            <ParameterSummary />
          </div>
        </div>

        {/* Performance Insights */}
        <div className="card animate-slide-up">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4">
            Performance Insights
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-2xl mb-2">🏆</div>
              <div className="font-semibold text-blue-800 dark:text-blue-200">
                Best Accuracy
              </div>
              <div className="text-sm text-blue-600 dark:text-blue-400">
                {modelNames[modelTypes.reduce((best, current) => 
                  results[current].accuracy > results[best].accuracy ? current : best
                )]}
              </div>
            </div>
            
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-2xl mb-2">⚡</div>
              <div className="font-semibold text-green-800 dark:text-green-200">
                Fastest Training
              </div>
              <div className="text-sm text-green-600 dark:text-green-400">
                {modelNames[modelTypes.reduce((fastest, current) => 
                  results[current].training_time < results[fastest].training_time ? current : fastest
                )]}
              </div>
            </div>
            
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-2xl mb-2">🎯</div>
              <div className="font-semibold text-purple-800 dark:text-purple-200">
                Best F1 Score
              </div>
              <div className="text-sm text-purple-600 dark:text-purple-400">
                {modelNames[modelTypes.reduce((best, current) => 
                  results[current].f1 > results[best].f1 ? current : best
                )]}
              </div>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex justify-center space-x-4 mt-8">
          <button
            onClick={() => navigate('/')}
            className="btn-primary"
          >
            Run New Simulation
          </button>
          <button
            onClick={() => window.print()}
            className="px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          >
            Export Results
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;