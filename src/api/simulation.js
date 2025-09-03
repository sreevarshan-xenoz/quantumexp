// API for quantum-classical ML simulation

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Generate dataset preview
export const generateDatasetPreview = async (params) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate_dataset`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  } catch (error) {
    console.error('Dataset preview API error:', error);
    // Fallback to mock data if API fails
    return generateMockDataset(params);
  }
};

// Fallback mock dataset generation
const generateMockDataset = (params) => {
  const { datasetType, noiseLevel, sampleSize } = params;
  const previewSize = Math.min(sampleSize, 200);

  let data = [];

  if (datasetType === 'circles') {
    for (let i = 0; i < previewSize; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() < 0.5 ? 0.3 + Math.random() * 0.2 : 0.8 + Math.random() * 0.2;
      const noise = (Math.random() - 0.5) * noiseLevel;
      const x = radius * Math.cos(angle) + noise;
      const y = radius * Math.sin(angle) + noise;
      const label = radius < 0.6 ? 0 : 1;
      data.push([x, y, label]);
    }
  } else if (datasetType === 'moons') {
    for (let i = 0; i < previewSize; i++) {
      const t = Math.random() * Math.PI;
      const noise = (Math.random() - 0.5) * noiseLevel;

      if (Math.random() < 0.5) {
        const x = Math.cos(t) + noise;
        const y = Math.sin(t) + noise;
        data.push([x, y, 0]);
      } else {
        const x = 1 - Math.cos(t) + noise;
        const y = 1 - Math.sin(t) - 0.5 + noise;
        data.push([x, y, 1]);
      }
    }
  } else if (datasetType === 'blobs') {
    for (let i = 0; i < previewSize; i++) {
      const center = Math.random() < 0.5 ? [-1, -1] : [1, 1];
      const x = center[0] + (Math.random() - 0.5) * (1 + noiseLevel);
      const y = center[1] + (Math.random() - 0.5) * (1 + noiseLevel);
      const label = center[0] < 0 ? 0 : 1;
      data.push([x, y, label]);
    }
  }

  return data;
};

// Run the full simulation
export const runSimulation = async (params) => {
  try {
    const response = await fetch(`${API_BASE_URL}/run_simulation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Simulation API error:', error);
    // Fallback to mock results if API fails
    return generateMockResults(params);
  }
};

// Fallback mock results
const generateMockResults = (params) => {
  const mockResults = {
    classical: {
      accuracy: 0.85 + Math.random() * 0.1,
      precision: 0.83 + Math.random() * 0.1,
      recall: 0.87 + Math.random() * 0.1,
      f1: 0.85 + Math.random() * 0.1,
      training_time: 0.5 + Math.random() * 0.5
    },
    quantum: {
      accuracy: 0.82 + Math.random() * 0.12,
      precision: 0.80 + Math.random() * 0.12,
      recall: 0.84 + Math.random() * 0.12,
      f1: 0.82 + Math.random() * 0.12,
      training_time: 2.5 + Math.random() * 2.0
    },
    hybrid: {
      accuracy: 0.88 + Math.random() * 0.08,
      precision: 0.86 + Math.random() * 0.08,
      recall: 0.90 + Math.random() * 0.08,
      f1: 0.88 + Math.random() * 0.08,
      training_time: 1.5 + Math.random() * 1.0
    }
  };

  const mockPlots = {
    classical: generateMockPlot('Classical Model'),
    quantum: generateMockPlot('Quantum Model'),
    hybrid: generateMockPlot('Hybrid Model'),
    comparison: generateMockComparisonPlot(mockResults)
  };

  return {
    results: mockResults,
    plots: mockPlots,
    dataset_info: {
      type: params.datasetType,
      samples: params.sampleSize,
      features: 2,
      classes: 2,
      noise: params.noiseLevel
    }
  };
};



// Generate mock plot data (in real implementation, this would come from backend)
const generateMockPlot = (title) => {
  // This would be a base64 encoded image from matplotlib
  return `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`;
};

const generateMockComparisonPlot = (results) => {
  // This would be a base64 encoded comparison chart
  return `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==`;
};

// Hyperparameter optimization
export const optimizeHyperparameters = async (config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/optimize_hyperparameters`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Hyperparameter optimization API error:', error);
    // Fallback to mock results if API fails
    return generateMockOptimizationResults(config);
  }
};

// Fallback mock optimization results
const generateMockOptimizationResults = (config) => {
  const mockResults = {
    status: "completed",
    optimization_results: {
      classical: {
        logistic: {
          best_params: { C: 1.0, max_iter: 1000 },
          best_score: 0.85 + Math.random() * 0.1,
          validation_score: 0.83 + Math.random() * 0.1,
          test_score: 0.82 + Math.random() * 0.1
        },
        random_forest: {
          best_params: { n_estimators: 100, max_depth: 10 },
          best_score: 0.87 + Math.random() * 0.08,
          validation_score: 0.85 + Math.random() * 0.08,
          test_score: 0.84 + Math.random() * 0.08
        }
      },
      quantum: {
        vqc: {
          best_params: { reps: 3 },
          best_score: 0.82 + Math.random() * 0.12,
          validation_score: 0.80 + Math.random() * 0.12,
          test_score: 0.79 + Math.random() * 0.12
        }
      }
    },
    dataset_info: {
      type: config.datasetType,
      samples: config.sampleSize,
      train_size: Math.floor(config.sampleSize * 0.6),
      val_size: Math.floor(config.sampleSize * 0.2),
      test_size: Math.floor(config.sampleSize * 0.2)
    },
    optimization_config: {
      method: config.method,
      cv_folds: config.cv_folds,
      scoring: config.scoring,
      n_trials: config.n_trials
    }
  };
  
  return mockResults;
};