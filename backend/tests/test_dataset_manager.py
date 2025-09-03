import pytest
import numpy as np
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset_manager import EnhancedDatasetManager


class TestEnhancedDatasetManager:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.dm = EnhancedDatasetManager()

    def test_load_circles_dataset(self):
        """Test loading circles dataset."""
        X, y, feature_names, description = self.dm.load_dataset('circles', n_samples=100)
        
        assert X.shape[0] == 100
        assert X.shape[1] == 2
        assert len(np.unique(y)) == 2
        assert len(feature_names) == 2
        assert description is not None
        assert 'circles' in description.lower()

    def test_load_moons_dataset(self):
        """Test loading moons dataset."""
        X, y, feature_names, description = self.dm.load_dataset('moons', n_samples=200)
        
        assert X.shape[0] == 200
        assert X.shape[1] == 2
        assert len(np.unique(y)) == 2
        assert len(feature_names) == 2
        assert 'moons' in description.lower()

    def test_load_real_dataset(self):
        """Test loading real-world dataset."""
        X, y, feature_names, description = self.dm.load_dataset('iris_binary')
        
        assert X.shape[0] > 0
        assert X.shape[1] > 0
        assert len(np.unique(y)) == 2
        assert len(feature_names) == X.shape[1]
        assert description is not None

    def test_dataset_metadata(self):
        """Test dataset metadata calculation."""
        self.dm.load_dataset('circles', n_samples=100)
        
        metadata = self.dm.metadata
        assert metadata['n_samples'] == 100
        assert metadata['n_features'] == 2
        assert metadata['n_classes'] == 2
        assert 'class_distribution' in metadata
        assert 'dataset_complexity' in metadata
        assert 'feature_types' in metadata

    def test_preprocess_data(self):
        """Test data preprocessing."""
        self.dm.load_dataset('circles', n_samples=100)
        X_train, X_test, y_train, y_test = self.dm.preprocess_data(test_size=0.2)
        
        assert X_train.shape[0] == 80  # 80% for training
        assert X_test.shape[0] == 20   # 20% for testing
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_feature_engineering_pca(self):
        """Test PCA feature engineering."""
        self.dm.load_dataset('breast_cancer')
        original_features = self.dm.X.shape[1]
        
        # Test PCA
        self.dm.feature_engineering('pca', n_components=2)
        assert self.dm.X.shape[1] == 2
        assert len(self.dm.feature_names) == 2
        assert all('PC' in name for name in self.dm.feature_names)

    def test_feature_engineering_polynomial(self):
        """Test polynomial feature engineering."""
        self.dm.load_dataset('circles', n_samples=50)
        original_features = self.dm.X.shape[1]
        
        # Test polynomial features
        self.dm.feature_engineering('polynomial', degree=2)
        # For 2 features with degree 2: x1, x2, x1^2, x1*x2, x2^2 = 5 features
        expected_features = 5
        assert self.dm.X.shape[1] == expected_features

    def test_feature_engineering_selection(self):
        """Test feature selection."""
        self.dm.load_dataset('breast_cancer')
        original_features = self.dm.X.shape[1]
        
        # Test feature selection
        self.dm.feature_engineering('feature_selection', n_components=5)
        assert self.dm.X.shape[1] == 5
        assert len(self.dm.feature_names) == 5

    def test_complexity_calculation(self):
        """Test dataset complexity calculation."""
        self.dm.load_dataset('circles', n_samples=100)
        
        complexity = self.dm.metadata['dataset_complexity']
        assert 'linear_separability' in complexity
        assert 'feature_overlap' in complexity
        assert 'overall' in complexity
        
        # Check that values are in valid range [0, 1]
        assert 0 <= complexity['linear_separability'] <= 1
        assert 0 <= complexity['feature_overlap'] <= 1
        assert 0 <= complexity['overall'] <= 1

    def test_generate_visualizations(self):
        """Test visualization generation."""
        self.dm.load_dataset('circles', n_samples=100)
        plots = self.dm.generate_visualizations()
        
        expected_plots = ['class_distribution', 'feature_distributions', 'complexity_metrics']
        for plot_name in expected_plots:
            assert plot_name in plots
            assert plots[plot_name] is not None
            assert isinstance(plots[plot_name], str)  # Base64 encoded string

    def test_invalid_dataset(self):
        """Test loading invalid dataset."""
        with pytest.raises(ValueError):
            self.dm.load_dataset('invalid_dataset')

    def test_handle_imbalance_oversample(self):
        """Test oversampling for imbalanced data."""
        self.dm.load_dataset('circles', n_samples=100)
        
        # Create artificial imbalance
        imbalanced_indices = np.where(self.dm.y == 0)[0][:10]  # Keep only 10 samples of class 0
        balanced_indices = np.where(self.dm.y == 1)[0]
        all_indices = np.concatenate([imbalanced_indices, balanced_indices])
        
        self.dm.X = self.dm.X[all_indices]
        self.dm.y = self.dm.y[all_indices]
        
        X_train, X_test, y_train, y_test = self.dm.preprocess_data(
            handle_imbalance='oversample', test_size=0.2
        )
        
        # Check that classes are more balanced after oversampling
        unique, counts = np.unique(y_train, return_counts=True)
        assert len(unique) == 2
        # After oversampling, classes should be more balanced
        assert abs(counts[0] - counts[1]) <= max(counts) * 0.1  # Within 10% difference

    def test_feature_types_inference(self):
        """Test feature type inference."""
        self.dm.load_dataset('circles', n_samples=100)
        feature_types = self.dm.metadata['feature_types']
        
        assert len(feature_types) == self.dm.X.shape[1]
        assert all(ft in ['numerical', 'categorical'] for ft in feature_types)

    def test_correlation_matrix_generation(self):
        """Test correlation matrix generation for multi-feature datasets."""
        self.dm.load_dataset('breast_cancer')
        plots = self.dm.generate_visualizations()
        
        if self.dm.X.shape[1] > 1:
            assert 'correlation_matrix' in plots
            assert plots['correlation_matrix'] is not None


if __name__ == '__main__':
    pytest.main([__file__])