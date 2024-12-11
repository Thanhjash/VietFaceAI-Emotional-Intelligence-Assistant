# face_analysis/src/data_processing/analyzers/distribution_analyzer.py

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from collections import Counter
from face_analysis.src.data_processing.analyzers.base_analyzer import BaseAnalyzer
import logging

class DistributionAnalyzer(BaseAnalyzer):
    """Analyzer for age and gender distribution analysis"""
    
    def __init__(
        self, 
        data_dir: str, 
        cache_dir: Optional[str] = None,
        age_range: Optional[Tuple[int, int]] = None,
        gender_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize distribution analyzer
        
        Args:
            data_dir: Path to dataset directory
            cache_dir: Optional path to cache computed results
            age_range: Tuple of (min_age, max_age) for validation
            gender_mapping: Dictionary mapping folder names to gender labels
        """
        super().__init__(data_dir, cache_dir)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Set other attributes
        self.age_range = age_range or (0, 100)  # Default age range if not specified
        self.gender_mapping = gender_mapping or {"111": "male", "112": "female"}  # Default mapping
    def scan_dataset(self) -> pd.DataFrame:
        """
        Override scan_dataset to use config-based gender mapping
        """
        data = []
        for age_dir in self.data_dir.glob("*"):
            if not age_dir.is_dir() or not age_dir.name.isdigit():
                continue
                
            age = int(age_dir.name)
            # Skip ages outside the specified range
            if not (self.age_range[0] <= age <= self.age_range[1]):
                continue
                
            for gender_code in self.gender_mapping.keys():
                gender_dir = age_dir / gender_code
                if not gender_dir.exists():
                    continue
                    
                for img_path in gender_dir.glob("*.jpg"):
                    data.append({
                        "path": str(img_path),
                        "age": age,
                        "gender": self.gender_mapping[gender_code],
                        "filename": img_path.name
                    })
                    
        df = pd.DataFrame(data)
        self.metadata = df
        return df

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze age and gender distribution
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Try loading cached results first
            cached = self.load_results("distribution_analysis.npy")
            if cached is not None:
                self.logger.info("Loaded cached distribution analysis results")
                return cached
                
            # Ensure metadata is loaded
            if self.metadata is None:
                self.logger.info("Scanning dataset for metadata...")
                self.scan_dataset()
                
            self.logger.info("Starting distribution analysis...")
            
            # Calculate basic distributions
            age_dist = Counter(self.metadata['age'])
            gender_dist = Counter(self.metadata['gender'])
            
            # Log distribution info
            self.logger.debug(f"Age distribution range: {min(age_dist.keys())} to {max(age_dist.keys())}")
            self.logger.debug(f"Gender distribution: {dict(gender_dist)}")
            
            # Calculate 2D distribution with logging
            self.logger.debug("Starting age-gender distribution calculation")
            
            # Group by age and gender, then convert to size counts
            age_gender_counts = self.metadata.groupby(['age', 'gender']).size()
            self.logger.debug(f"Age-gender grouped data shape: {age_gender_counts.shape}")
            
            # Create distribution dictionary with proper typing
            age_gender_dist = {
                (int(age), str(gender)): int(count)
                for (age, gender), count in age_gender_counts.items()
            }
            
            # Verify distribution data
            self.logger.debug(f"Age-gender distribution entries: {len(age_gender_dist)}")
            self.logger.debug(f"Sample entries: {list(age_gender_dist.items())[:5]}")
            
            # Validation checks
            if not age_gender_dist:
                self.logger.warning("No age-gender distribution data generated")
            
            # Calculate detailed statistics
            stats = {
                'age_mean': float(self.metadata['age'].mean()),
                'age_std': float(self.metadata['age'].std()),
                'age_median': float(self.metadata['age'].median()),
                'age_mode': int(self.metadata['age'].mode()[0]),
                'age_range': (int(self.metadata['age'].min()), 
                            int(self.metadata['age'].max())),
                'total_samples': int(len(self.metadata)),
                'gender_ratio': float(gender_dist['male'] / gender_dist['female']
                                    if gender_dist['female'] != 0 else 0)
            }
            
            # Calculate quartiles and IQR
            q1 = float(self.metadata['age'].quantile(0.25))
            q3 = float(self.metadata['age'].quantile(0.75))
            iqr = float(q3 - q1)
            
            stats.update({
                'age_q1': q1,
                'age_q3': q3,
                'age_iqr': iqr,
                'age_skewness': float(self.metadata['age'].skew()),
                'age_kurtosis': float(self.metadata['age'].kurtosis())
            })
            
            # Log statistics
            self.logger.debug(f"Calculated statistics: {stats}")
            
            # Analyze distribution balance/imbalance
            age_imbalance = self._analyze_imbalance(age_dist)
            gender_imbalance = self._analyze_imbalance(gender_dist)
            
            self.logger.debug(f"Age imbalance metrics: {age_imbalance}")
            self.logger.debug(f"Gender imbalance metrics: {gender_imbalance}")
            
            # Compile final results
            results = {
                'age_distribution': dict(age_dist),
                'gender_distribution': dict(gender_dist),
                'age_gender_distribution': age_gender_dist,
                'statistics': stats,
                'imbalance_analysis': {
                    'age': age_imbalance,
                    'gender': gender_imbalance
                },
                'metadata': {
                    'total_samples': len(self.metadata),
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'version': '2.0'
                }
            }
            
            # Validate results before saving
            self._validate_results(results)
            
            # Cache results
            self.logger.info("Caching distribution analysis results...")
            self.save_results(results, "distribution_analysis.npy")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in distribution analysis: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            raise
            
    def _validate_results(self, results: Dict[str, Any]) -> None:
        """Validate analysis results"""
        try:
            # Check required keys
            required_keys = ['age_distribution', 'gender_distribution', 
                            'age_gender_distribution', 'statistics']
            for key in required_keys:
                if key not in results:
                    raise ValueError(f"Missing required key in results: {key}")
            
            # Validate age-gender distribution
            age_gender_dist = results['age_gender_distribution']
            if not isinstance(age_gender_dist, dict):
                raise TypeError("age_gender_distribution must be a dictionary")
                
            # Check key format
            for key in age_gender_dist.keys():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(f"Invalid key format in age_gender_distribution: {key}")
                
                age, gender = key
                if not isinstance(age, (int, np.integer)):
                    raise TypeError(f"Age must be integer, got {type(age)}")
                if not isinstance(gender, str):
                    raise TypeError(f"Gender must be string, got {type(gender)}")
                    
            self.logger.debug("Results validation passed")
            
        except Exception as e:
            self.logger.error(f"Results validation failed: {str(e)}")
            raise
        
    def _analyze_imbalance(self, distribution: Counter) -> Dict[str, float]:
        """Analyze imbalance in a distribution"""
        total = sum(distribution.values())
        proportions = {k: v/total for k, v in distribution.items()}
        
        # Calculate entropy as imbalance metric
        entropy = -sum(p * np.log2(p) for p in proportions.values())
        max_entropy = np.log2(len(distribution))
        
        return {
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
            'proportions': proportions
        }