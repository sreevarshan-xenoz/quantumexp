import React from 'react';

const ProgressIndicator = ({ progress, stage = 'Training' }) => {
  const getProgressColor = () => {
    if (progress < 30) return 'from-red-500 to-red-600';
    if (progress < 70) return 'from-yellow-500 to-orange-500';
    return 'from-green-500 to-emerald-500';
  };

  const stages = [
    { name: 'Data Prep', threshold: 20, icon: '📊' },
    { name: 'Training', threshold: 60, icon: '🧠' },
    { name: 'Evaluation', threshold: 90, icon: '📈' },
    { name: 'Complete', threshold: 100, icon: '✅' }
  ];

  const getCurrentStage = () => {
    return stages.find(stage => progress < stage.threshold) || stages[stages.length - 1];
  };

  const currentStage = getCurrentStage();

  return (
    <div className="progress-indicator animate-slide-up">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-2xl">{currentStage.icon}</span>
          <span className="font-semibold text-gray-800 dark:text-gray-200">
            {currentStage.name}
          </span>
        </div>
        <span className="font-bold text-lg text-gray-800 dark:text-gray-200">
          {Math.round(progress)}%
        </span>
      </div>

      {/* Main progress bar */}
      <div className="relative w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 mb-4 overflow-hidden">
        <div 
          className={`h-full bg-gradient-to-r ${getProgressColor()} transition-all duration-500 ease-out relative`}
          style={{ width: `${progress}%` }}
        >
          <div className="absolute inset-0 bg-white opacity-20 animate-pulse"></div>
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-full h-1 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"></div>
        </div>
      </div>

      {/* Stage indicators */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {stages.slice(0, -1).map((stageItem, index) => {
          const isActive = progress >= stageItem.threshold;
          const isCurrent = currentStage.name === stageItem.name;
          
          return (
            <div 
              key={stageItem.name}
              className={`text-center p-3 rounded-lg transition-all duration-300 ${
                isActive 
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-lg transform scale-105' 
                  : isCurrent
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 border-2 border-blue-500 animate-pulse'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
              }`}
            >
              <div className="text-lg mb-1">{stageItem.icon}</div>
              <div className="text-xs font-medium">{stageItem.name}</div>
            </div>
          );
        })}
      </div>

      {/* Detailed progress info */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
          <span>Current Stage:</span>
          <span className="font-medium">{currentStage.name}</span>
        </div>
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>Estimated Time:</span>
          <span className="font-medium">
            {progress < 100 ? `${Math.max(1, Math.round((100 - progress) / 10))} min remaining` : 'Complete!'}
          </span>
        </div>
      </div>

      {/* Animated dots for active state */}
      {progress < 100 && (
        <div className="flex justify-center mt-4 space-x-1">
          {[0, 1, 2].map(i => (
            <div
              key={i}
              className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 0.2}s` }}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ProgressIndicator;