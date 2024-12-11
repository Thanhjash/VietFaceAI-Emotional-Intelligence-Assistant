# face_analysis/src/data_processing/analyzers/quality_analyzer.py

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from face_analysis.src.data_processing.analyzers.base_analyzer import BaseAnalyzer
import logging

class QualityAnalyzer(BaseAnalyzer):
    """Analyzer for assessing image quality metrics"""
    
    def __init__(self, data_dir: str, cache_dir: str = None):
        """
        Initialize quality analyzer
        
        Args:
            data_dir: Path to dataset directory
            cache_dir: Optional path to cache computed results
        """
        super().__init__(data_dir, cache_dir)
        self.logger = logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive image quality analysis
        
        Returns:
            Dictionary containing quality metrics:
            - resolution_stats: Statistics about image resolutions
            - brightness_stats: Brightness analysis results
            - contrast_stats: Contrast analysis results
            - blur_stats: Blur detection results
            - overall_quality: Aggregated quality scores
        """
        # Try loading cached results
        cached = self.load_results("quality_analysis.npy")
        if cached is not None:
            return cached
            
        # Ensure metadata is loaded
        if self.metadata is None:
            self.scan_dataset()
            
        # Initialize results storage
        results = {
            'resolution_stats': {},
            'brightness_stats': {},
            'contrast_stats': {},
            'blur_stats': {},
            'overall_quality': {}
        }
        
        # Process each image
        total_images = len(self.metadata)
        for idx, row in self.metadata.iterrows():
            try:
                # Load image
                img_path = row['path']
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Analyze image quality
                quality_metrics = self._analyze_single_image(img)
                
                # Update statistics
                self._update_statistics(results, quality_metrics)
                
                # Log progress
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Processed {idx + 1}/{total_images} images")
                    
            except Exception as e:
                self.logger.error(f"Error processing image {img_path}: {str(e)}")
                continue
        
        # Compute final statistics
        self._compute_final_statistics(results)
        
        # Cache results
        self.save_results(results, "quality_analysis.npy")
        return results
    
    def _analyze_single_image(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze quality metrics for a single image
        
        Args:
            img: Input image as numpy array
            
        Returns:
            Dictionary containing quality metrics for the image
        """
        # Get resolution
        height, width = img.shape[:2]
        
        # Calculate brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Detect blur using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        # Calculate histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        hist_entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        return {
            'resolution': (width, height),
            'brightness': brightness,
            'contrast': contrast,
            'blur_score': blur_score,
            'hist_std': hist_std,
            'hist_entropy': hist_entropy
        }
    
    def _update_statistics(self, results: Dict[str, Any], metrics: Dict[str, Any]):
        """Update running statistics with new image metrics"""
        # Update resolution statistics
        width, height = metrics['resolution']
        results['resolution_stats'].setdefault('widths', []).append(width)
        results['resolution_stats'].setdefault('heights', []).append(height)
        results['resolution_stats'].setdefault('aspects', []).append(width/height)
        
        # Update brightness statistics
        results['brightness_stats'].setdefault('values', []).append(metrics['brightness'])
        
        # Update contrast statistics
        results['contrast_stats'].setdefault('values', []).append(metrics['contrast'])
        
        # Update blur statistics
        results['blur_stats'].setdefault('scores', []).append(metrics['blur_score'])
        
    def _compute_final_statistics(self, results: Dict[str, Any]):
        """Compute final statistical measures"""
        # Resolution statistics
        for key in ['widths', 'heights', 'aspects']:
            values = np.array(results['resolution_stats'][key])
            results['resolution_stats'][f'{key}_stats'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
        # Brightness statistics
        brightness_values = np.array(results['brightness_stats']['values'])
        results['brightness_stats']['statistics'] = {
            'mean': np.mean(brightness_values),
            'std': np.std(brightness_values),
            'histogram': np.histogram(brightness_values, bins=10)[0].tolist()
        }
        
        # Contrast statistics
        contrast_values = np.array(results['contrast_stats']['values'])
        results['contrast_stats']['statistics'] = {
            'mean': np.mean(contrast_values),
            'std': np.std(contrast_values),
            'histogram': np.histogram(contrast_values, bins=10)[0].tolist()
        }
        
        # Blur statistics
        blur_scores = np.array(results['blur_stats']['scores'])
        results['blur_stats']['statistics'] = {
            'mean': np.mean(blur_scores),
            'std': np.std(blur_scores),
            'potential_blur_count': np.sum(blur_scores < 100)  # threshold for blur detection
        }
        
        # Compute overall quality scores
        self._compute_overall_quality(results)
    
    def _compute_overall_quality(self, results: Dict[str, Any]):
        """Compute overall quality scores"""
        # Initialize quality categories
        quality_scores = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0
        }
        
        # Get all quality metrics
        blur_scores = np.array(results['blur_stats']['scores'])
        brightness_values = np.array(results['brightness_stats']['values'])
        contrast_values = np.array(results['contrast_stats']['values'])
        
        total_images = len(blur_scores)
        
        for i in range(total_images):
            score = 0
            # Score based on blur (0-30 points)
            score += 30 * min(blur_scores[i] / 1000, 1)
            
            # Score based on brightness (0-35 points)
            brightness_score = 35 * (1 - abs(brightness_values[i] - 127.5) / 127.5)
            score += brightness_score
            
            # Score based on contrast (0-35 points)
            contrast_score = 35 * min(contrast_values[i] / 80, 1)
            score += contrast_score
            
            # Categorize based on final score
            if score >= 85:
                quality_scores['excellent'] += 1
            elif score >= 70:
                quality_scores['good'] += 1
            elif score >= 50:
                quality_scores['fair'] += 1
            else:
                quality_scores['poor'] += 1
        
        # Convert to percentages
        results['overall_quality'] = {
            category: (count / total_images) * 100
            for category, count in quality_scores.items()
        }