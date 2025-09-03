import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import (
    make_circles, make_moons, make_blobs, make_classification,
    load_iris, load_wine, load_breast_cancer, load_digits
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from scipy import stats
import io
import base64
import warnings
warnings.filterwarnings('ignore')

class EnhancedDatasetManager:
    def __init__(self):
        self.datasets = {
            # Synthetic datasets
            'circles': self._generate_circles,
            'moons': self._generate_moons,
            'blobs': self._generate_blobs,
            'classification': self._generate_classification,
            'spiral': self._generate_spiral,
            'xor': self._generate_xor,
            'gaussian_quantum': self._generate_gaussian_quantum,
            
            # Real-world datasets (converted to binary classification)
            'iris_binary': self._load_iris_binary,
            'wine_binary': self._load_wine_binary,
            'breast_cancer': self._load_breast_cancer,
            'digits_binary': self._load_digits_binary,
        }
        
        self.current_dataset = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_name = None
        self.description = None
        self.metadata = {}
        self.scaler = MinMaxScaler((0, 2 * np.pi))  # For quantum compatibility

    def _generate_circles(self, n_samples=1000, noise=0.1, factor=0.5):
        """Generate concentric circles dataset"""
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = "Concentric circles dataset with adjustable noise and separation factor."
        return X, y, feature_names, description

    def _generate_moons(self, n_samples=1000, noise=0.1):
        """Generate two interleaving half circles"""
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = "Two interleaving half circles dataset with adjustable noise."
        return X, y, feature_names, description

    def _generate_blobs(self, n_samples=1000, noise=0.1, centers=3, cluster_std=1.0):
        """Generate isotropic Gaussian blobs"""
        X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)
        # Convert to binary classification
        y = (y >= centers // 2).astype(int)
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = f"Isotropic Gaussian blobs with {centers} centers and adjustable cluster standard deviation."
        return X, y, feature_names, description

    def _generate_classification(self, n_samples=1000, noise=0.1, n_features=2, n_informative=2, n_redundant=0):
        """Generate a random n-class classification problem"""
        X, y = make_classification(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative, 
            n_redundant=n_redundant, n_classes=2, random_state=42
        )
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = f"Random classification problem with {n_features} features, {n_informative} informative."
        return X, y, feature_names, description

    def _generate_spiral(self, n_samples=1000, noise=0.1):
        """Generate spiral dataset"""
        n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
        X = np.hstack((d1x, d1y))
        y = np.zeros(n_samples)
        y[n_samples // 2:] = 1
        # Add additional noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = "Spiral dataset with adjustable noise."
        return X, y, feature_names, description

    def _generate_xor(self, n_samples=1000, noise=0.1):
        """Generate XOR dataset"""
        X = np.random.randn(n_samples, 2)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
        # Add noise
        X += noise * np.random.randn(n_samples, 2)
        feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
        description = "XOR dataset with adjustable noise."
        return X, y, feature_names, description

    def _generate_gaussian_quantum(self, n_samples=1000, n_qubits=2, noise=0.1):
        """Generate dataset simulating quantum measurements"""
        # Create quantum states
        states = np.random.randn(n_samples, 2**n_qubits) + 1j * np.random.randn(n_samples, 2**n_qubits)
        states = states / np.linalg.norm(states, axis=1, keepdims=True)
        
        # Simulate measurements
        measurements = np.abs(states)**2
        
        # Use first two measurement probabilities as features
        X = measurements[:, :2]
        
        # Create binary labels based on quantum state properties
        y = (np.angle(states[:, 0]) > 0).astype(int)
        
        # Add noise
        X += noise * np.random.randn(n_samples, 2)
        feature_names = [f'Measurement_{i+1}' for i in range(X.shape[1])]
        description = f"Dataset simulating quantum measurements on {n_qubits} qubits with adjustable noise."
        return X, y, feature_names, description

    def _load_iris_binary(self, n_samples=None, noise=0.0):
        """Load Iris dataset and convert to binary classification"""
        data = load_iris()
        X = data.data
        y = (data.target == 0).astype(int)  # Setosa vs others
        
        # Handle sampling if requested
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
            
        feature_names = data.feature_names
        description = "Iris dataset converted to binary classification (Setosa vs others)."
        return X, y, feature_names, description

    def _load_wine_binary(self, n_samples=None, noise=0.0):
        """Load Wine dataset and convert to binary classification"""
        data = load_wine()
        X = data.data
        y = (data.target == 0).astype(int)  # Class 0 vs others
        
        # Handle sampling if requested
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
            
        feature_names = data.feature_names
        description = "Wine dataset converted to binary classification (Class 0 vs others)."
        return X, y, feature_names, description

    def _load_breast_cancer(self, n_samples=None, noise=0.0):
        """Load Breast Cancer dataset"""
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Handle sampling if requested
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
            
        feature_names = data.feature_names
        description = "Breast Cancer Wisconsin dataset for binary classification."
        return X, y, feature_names, description

    def _load_digits_binary(self, n_samples=None, noise=0.0):
        """Load Digits dataset and convert to binary classification"""
        data = load_digits()
        X = data.data
        y = (data.target % 2 == 0).astype(int)  # Even vs odd digits
        
        # Handle sampling if requested
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        # Add noise if requested
        if noise > 0:
            X += noise * np.random.randn(*X.shape)
            
        feature_names = [f'pixel_{i}' for i in range(X.shape[1])]
        description = "Digits dataset converted to binary classification (Even vs Odd)."
        return X, y, feature_names, description

    def load_dataset(self, dataset_name, **kwargs):
        """Load a dataset by name"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available.")
        
        # Load dataset
        X, y, feature_names, description = self.datasets[dataset_name](**kwargs)
        
        # Store dataset information
        self.current_dataset = dataset_name
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_name = 'Target'
        self.description = description
        
        # Calculate metadata (convert numpy types to native Python types for JSON serialization)
        self.metadata = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_classes': int(len(np.unique(y))),
            'class_distribution': (np.bincount(y.astype(int)) / len(y)).tolist(),
            'missing_values': int(np.isnan(X).sum()) if X.dtype.kind in 'fc' else 0,
            'feature_types': self._infer_feature_types(X),
            'dataset_complexity': self._calculate_complexity(X, y)
        }
        
        return X, y, feature_names, description

    def _infer_feature_types(self, X):
        """Infer feature types (numerical/categorical)"""
        feature_types = []
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            # Check if values are categorical (few unique values and can be treated as integers)
            if len(unique_vals) < 10:
                try:
                    # Try to convert to integer and check if it's categorical
                    if np.allclose(unique_vals, unique_vals.astype(int)):
                        feature_types.append('categorical')
                    else:
                        feature_types.append('numerical')
                except:
                    feature_types.append('numerical')
            else:
                feature_types.append('numerical')
        return feature_types

    def _calculate_complexity(self, X, y):
        """Calculate dataset complexity metrics"""
        # Linear separability
        try:
            clf = LinearSVC(random_state=42, max_iter=1000)
            linear_score = np.mean(cross_val_score(clf, X, y, cv=3))
        except:
            linear_score = 0.5
        
        # Feature overlap
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        overlaps = []
        
        for i in range(X.shape[1]):
            if len(class_0) > 0 and len(class_1) > 0:
                min_0, max_0 = np.min(class_0[:, i]), np.max(class_0[:, i])
                min_1, max_1 = np.min(class_1[:, i]), np.max(class_1[:, i])
                
                # Calculate overlap
                overlap_start = max(min_0, min_1)
                overlap_end = min(max_0, max_1)
                
                if overlap_end > overlap_start:
                    range_0 = max_0 - min_0
                    range_1 = max_1 - min_1
                    overlap_ratio = (overlap_end - overlap_start) / max(range_0, range_1, 1e-8)
                else:
                    overlap_ratio = 0.0
                
                overlaps.append(overlap_ratio)
            else:
                overlaps.append(0.0)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        
        return {
            'linear_separability': float(linear_score),
            'feature_overlap': float(avg_overlap),
            'overall': float((linear_score + (1 - avg_overlap)) / 2)
        }

    def preprocess_data(self, test_size=0.2, handle_imbalance=None, scale=True):
        """Preprocess the dataset"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        
        # Handle imbalanced data
        if handle_imbalance == 'oversample':
            from collections import Counter
            counter = Counter(y_train)
            max_count = max(counter.values())
            
            X_resampled = []
            y_resampled = []
            
            for class_label in counter.keys():
                class_indices = np.where(y_train == class_label)[0]
                class_samples = X_train[class_indices]
                
                # Oversample to match max_count
                n_samples_needed = max_count - len(class_samples)
                if n_samples_needed > 0:
                    additional_indices = np.random.choice(len(class_samples), n_samples_needed, replace=True)
                    additional_samples = class_samples[additional_indices]
                    class_samples = np.vstack([class_samples, additional_samples])
                
                X_resampled.append(class_samples)
                y_resampled.extend([class_label] * len(class_samples))
            
            X_train = np.vstack(X_resampled)
            y_train = np.array(y_resampled)
            
            # Shuffle
            shuffle_indices = np.random.permutation(len(X_train))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
        
        # Scale features
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def feature_engineering(self, method='polynomial', degree=2, n_components=None):
        """Apply feature engineering techniques"""
        if method == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            self.X = poly.fit_transform(self.X)
            self.feature_names = [f'poly_{i}' for i in range(self.X.shape[1])]
        
        elif method == 'pca':
            n_components = n_components or min(2, self.X.shape[1])
            pca = PCA(n_components=n_components)
            self.X = pca.fit_transform(self.X)
            self.feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        elif method == 'feature_selection':
            n_components = n_components or min(2, self.X.shape[1])
            selector = SelectKBest(f_classif, k=n_components)
            self.X = selector.fit_transform(self.X, self.y)
            selected_indices = selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
        
        # Update metadata
        self.metadata['n_features'] = self.X.shape[1]
        self.metadata['feature_types'] = self._infer_feature_types(self.X)
        self.metadata['dataset_complexity'] = self._calculate_complexity(self.X, self.y)
        
        return self.X, self.y

    def generate_visualizations(self):
        """Generate comprehensive dataset visualizations"""
        plots = {}
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Class distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts = np.bincount(self.y.astype(int))
        colors = ['skyblue', 'salmon']
        bars = ax.bar(range(len(class_counts)), class_counts, color=colors[:len(class_counts)])
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels([f'Class {i}' for i in range(len(class_counts))])
        
        # Add percentage labels
        for i, (bar, count) in enumerate(zip(bars, class_counts)):
            percentage = 100 * count / len(self.y)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * max(class_counts), 
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plots['class_distribution'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # 2. Feature distributions
        n_features = min(self.X.shape[1], 8)  # Limit to 8 features for display
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(len(axes)):
            if i < n_features:
                axes[i].hist(self.X[:, i], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                axes[i].set_title(f'{self.feature_names[i]}', fontweight='bold')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['feature_distributions'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # 3. Correlation matrix (if more than 1 feature)
        if self.X.shape[1] > 1:
            df = pd.DataFrame(self.X[:, :min(10, self.X.shape[1])], 
                            columns=self.feature_names[:min(10, self.X.shape[1])])
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            plots['correlation_matrix'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # 4. Dataset complexity visualization
        complexity = self.metadata['dataset_complexity']
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Linear\nSeparability', 'Feature\nOverlap', 'Overall\nComplexity']
        values = [complexity['linear_separability'], 
                 1 - complexity['feature_overlap'], 
                 complexity['overall']]
        colors = ['skyblue', 'salmon', 'lightgreen']
        
        bars = ax.bar(metrics, values, color=colors)
        ax.set_title('Dataset Complexity Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plots['complexity_metrics'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        return plots

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')