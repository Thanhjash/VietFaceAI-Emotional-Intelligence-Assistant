# face_analysis/src/data_processing/analyzers/statistical_analyzer.py

import numpy as np
from scipy import stats
from typing import Dict, Any, List
from face_analysis.src.data_processing.analyzers.base_analyzer import BaseAnalyzer
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class StatisticalAnalyzer(BaseAnalyzer):
    """Analyzer for advanced statistical analysis of dataset characteristics"""
    
    def __init__(self, data_dir: str, cache_dir: str = None):
        """
        Initialize statistical analyzer
        
        Args:
            data_dir: Path to dataset directory
            cache_dir: Optional path to cache computed results
        """
        super().__init__(data_dir, cache_dir)
        self.logger = logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis
        
        Returns:
            Dictionary containing:
            - descriptive_stats: Basic statistical measures
            - distribution_analysis: Distribution characteristics
            - correlation_analysis: Feature correlations
            - outlier_analysis: Outlier detection results
            - cluster_analysis: Data clustering information
        """
        # Try loading cached results
        cached = self.load_results("statistical_analysis.npy")
        if cached is not None:
            return cached
            
        # Ensure metadata is loaded
        if self.metadata is None:
            self.scan_dataset()
            
        results = {
            'descriptive_stats': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'outlier_analysis': {},
            'cluster_analysis': {}
        }
        
        # Perform analyses
        self._analyze_descriptive_statistics(results)
        self._analyze_distributions(results)
        self._analyze_correlations(results)
        self._analyze_outliers(results)
        self._analyze_clusters(results)
        
        # Cache results
        self.save_results(results, "statistical_analysis.npy")
        return results
        
    def _analyze_descriptive_statistics(self, results: Dict[str, Any]):
        """Analyze basic descriptive statistics"""
        # Age statistics
        age_stats = self.metadata['age'].describe()
        results['descriptive_stats']['age'] = {
            'mean': age_stats['mean'],
            'std': age_stats['std'],
            'min': age_stats['min'],
            'max': age_stats['max'],
            'quartiles': [
                age_stats['25%'],
                age_stats['50%'],
                age_stats['75%']
            ],
            'skewness': stats.skew(self.metadata['age']),
            'kurtosis': stats.kurtosis(self.metadata['age'])
        }
        
        # Gender statistics
        gender_counts = self.metadata['gender'].value_counts()
        results['descriptive_stats']['gender'] = {
            'counts': gender_counts.to_dict(),
            'proportions': gender_counts.apply(lambda x: x/len(self.metadata)).to_dict()
        }
        
    def _analyze_distributions(self, results: Dict[str, Any]):
        """Analyze distribution characteristics"""
        # Age distribution tests
        age_data = self.metadata['age']
        
        # Normality tests
        _, normality_p = stats.normaltest(age_data)
        results['distribution_analysis']['age'] = {
            'normality_test': {
                'statistic': float(_),
                'p_value': float(normality_p),
                'is_normal': normality_p > 0.05
            }
        }
        
        # Distribution fitting
        distributions = [stats.norm, stats.gamma, stats.beta]
        best_fit = {'name': None, 'aic': float('inf')}
        
        for dist in distributions:
            try:
                params = dist.fit(age_data)
                aic = stats.aic(lambda x, *params: np.log(dist.pdf(x, *params)),
                              params, age_data)
                if aic < best_fit['aic']:
                    best_fit = {
                        'name': dist.name,
                        'aic': aic,
                        'params': params
                    }
            except:
                continue
                
        results['distribution_analysis']['age']['best_fit'] = best_fit
        
    def _analyze_correlations(self, results: Dict[str, Any]):
        """Analyze correlations between features"""
        # Create dummy variables for gender
        df = pd.get_dummies(self.metadata, columns=['gender'])
        
        # Calculate correlations
        corr_matrix = df.corr()
        results['correlation_analysis'] = {
            'matrix': corr_matrix.to_dict(),
            'significant_correlations': []
        }
        
        # Find significant correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.1:  # Correlation threshold
                    results['correlation_analysis']['significant_correlations'].append({
                        'features': (col1, col2),
                        'correlation': corr
                    })
                    
    def _analyze_outliers(self, results: Dict[str, Any]):
        """Detect and analyze outliers"""
        # Z-score based outlier detection
        age_zscore = np.abs(stats.zscore(self.metadata['age']))
        outliers_zscore = np.where(age_zscore > 3)[0]
        
        # IQR based outlier detection
        Q1 = self.metadata['age'].quantile(0.25)
        Q3 = self.metadata['age'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = self.metadata[
            (self.metadata['age'] < (Q1 - 1.5 * IQR)) |
            (self.metadata['age'] > (Q3 + 1.5 * IQR))
        ].index
        
        results['outlier_analysis'] = {
            'zscore': {
                'count': len(outliers_zscore),
                'indices': outliers_zscore.tolist()
            },
            'iqr': {
                'count': len(outliers_iqr),
                'indices': outliers_iqr.tolist()
            }
        }
        
    def _analyze_clusters(self, results: Dict[str, Any]):
        """Analyze data clusters and patterns"""
        # Prepare data for clustering
        df = pd.get_dummies(self.metadata, columns=['gender'])
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # PCA analysis
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        results['cluster_analysis'] = {
            'pca': {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
            }
        }