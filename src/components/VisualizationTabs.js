import React, { useState } from 'react';

const VisualizationTabs = ({ plots, results }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [activeModel, setActiveModel] = useState('classical');

  const tabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'exploration', name: 'Data Exploration', icon: '🔍' },
    { id: 'evaluation', name: 'Model Evaluation', icon: '📈' },
    { id: 'importance', name: 'Feature Analysis', icon: '🎯' },
    { id: 'advanced', name: 'Advanced Analysis', icon: '🧠' }
  ];

  const models = [
    { id: 'classical', name: 'Classical', color: 'blue' },
    { id: 'quantum', name: 'Quantum', color: 'green' },
    { id: 'hybrid', name: 'Hybrid', color: 'purple' }
  ];

  const PlotImage = ({ plotKey, title, description }) => {
    const plotData = plots[plotKey];
    if (!plotData) return null;

    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">{title}</h4>
        {description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{description}</p>
        )}
        <div className="flex justify-center">
          <img 
            src={plotData} 
            alt={title}
            className="max-w-full h-auto rounded-lg shadow-sm"
            style={{ maxHeight: '400px' }}
          />
        </div>
      </div>
    );
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      <PlotImage 
        plotKey="comparison" 
        title="Model Performance Comparison"
        description="Comprehensive comparison of all models across multiple metrics"
      />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <PlotImage 
          plotKey="classical_decision_boundary" 
          title="Classical Decision Boundary"
        />
        <PlotImage 
          plotKey="quantum_decision_boundary" 
          title="Quantum Decision Boundary"
        />
        <PlotImage 
          plotKey="hybrid_decision_boundary" 
          title="Hybrid Decision Boundary"
        />
      </div>
    </div>
  );

  const renderExplorationTab = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PlotImage 
          plotKey="exploratory_feature_histograms" 
          title="Feature Distributions"
          description="Histogram showing the distribution of each feature"
        />
        <PlotImage 
          plotKey="exploratory_boxplots" 
          title="Outlier Detection"
          description="Box plots to identify outliers in the dataset"
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PlotImage 
          plotKey="exploratory_pca" 
          title="PCA Visualization"
          description="Principal Component Analysis showing data structure"
        />
        <PlotImage 
          plotKey="exploratory_tsne" 
          title="t-SNE Visualization"
          description="t-SNE embedding revealing data clusters"
        />
      </div>
      <PlotImage 
        plotKey="exploratory_correlation_heatmap" 
        title="Feature Correlation Matrix"
        description="Correlation between different features in the dataset"
      />
    </div>
  );

  const renderEvaluationTab = () => (
    <div className="space-y-6">
      <div className="flex justify-center mb-6">
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {models.map(model => (
            <button
              key={model.id}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeModel === model.id
                  ? `bg-${model.color}-500 text-white shadow-md`
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
              }`}
              onClick={() => setActiveModel(model.id)}
            >
              {model.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PlotImage 
          plotKey={`${activeModel}_confusion_matrix`} 
          title="Confusion Matrix"
          description="Detailed breakdown of correct and incorrect predictions"
        />
        <PlotImage 
          plotKey={`${activeModel}_roc_curve`} 
          title="ROC Curve"
          description="Receiver Operating Characteristic curve showing model performance"
        />
      </div>
      <PlotImage 
        plotKey={`${activeModel}_precision_recall_curve`} 
        title="Precision-Recall Curve"
        description="Trade-off between precision and recall at different thresholds"
      />
    </div>
  );

  const renderImportanceTab = () => (
    <div className="space-y-6">
      <div className="flex justify-center mb-6">
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {models.map(model => (
            <button
              key={model.id}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeModel === model.id
                  ? `bg-${model.color}-500 text-white shadow-md`
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
              }`}
              onClick={() => setActiveModel(model.id)}
            >
              {model.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PlotImage 
          plotKey={`${activeModel}_feature_importance`} 
          title="Feature Importance"
          description="Importance of each feature as determined by the model"
        />
        <PlotImage 
          plotKey={`${activeModel}_permutation_importance`} 
          title="Permutation Importance"
          description="Feature importance based on performance decrease when shuffled"
        />
      </div>
    </div>
  );

  const renderAdvancedTab = () => (
    <div className="space-y-6">
      <div className="flex justify-center mb-6">
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {models.map(model => (
            <button
              key={model.id}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                activeModel === model.id
                  ? `bg-${model.color}-500 text-white shadow-md`
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
              }`}
              onClick={() => setActiveModel(model.id)}
            >
              {model.name}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PlotImage 
          plotKey={`${activeModel}_learning_curve`} 
          title="Learning Curve"
          description="Model performance vs training set size"
        />
        <PlotImage 
          plotKey={`${activeModel}_decision_uncertainty`} 
          title="Decision Boundary with Uncertainty"
          description="Decision boundary showing model confidence regions"
        />
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverviewTab();
      case 'exploration':
        return renderExplorationTab();
      case 'evaluation':
        return renderEvaluationTab();
      case 'importance':
        return renderImportanceTab();
      case 'advanced':
        return renderAdvancedTab();
      default:
        return renderOverviewTab();
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8 px-6 py-4 overflow-x-auto">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm whitespace-nowrap transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="text-lg">{tab.icon}</span>
              <span>{tab.name}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default VisualizationTabs;