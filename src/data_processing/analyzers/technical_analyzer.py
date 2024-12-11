# face_analysis/src/data_processing/analyzers/technical_analyzer.py

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import imghdr
import os
from face_analysis.src.data_processing.analyzers.base_analyzer import BaseAnalyzer
import logging

class TechnicalAnalyzer(BaseAnalyzer):
    """Analyzer for technical characteristics of images"""
    
    def __init__(self, data_dir: str, cache_dir: str = None):
        """
        Initialize technical analyzer
        
        Args:
            data_dir: Path to dataset directory
            cache_dir: Optional path to cache computed results
        """
        super().__init__(data_dir, cache_dir)
        self.logger = logging.getLogger(__name__)
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis of images
        
        Returns:
            Dictionary containing:
            - format_analysis: Image format statistics
            - size_analysis: File size statistics
            - color_analysis: Color channel analysis
            - histogram_analysis: Color histogram statistics
            - metadata_analysis: Image metadata information
        """
        # Try loading cached results
        cached = self.load_results("technical_analysis.npy")
        if cached is not None:
            return cached
            
        # Ensure metadata is loaded
        if self.metadata is None:
            self.scan_dataset()
            
        results = {
            'format_analysis': {'formats': {}},
            'size_analysis': {'sizes': []},
            'color_analysis': {
                'channels': {},
                'color_stats': []
            },
            'histogram_analysis': {
                'means': [],
                'distributions': []
            },
            'metadata_analysis': {
                'exif_stats': {},
                'creation_dates': []
            }
        }
        
        total_images = len(self.metadata)
        for idx, row in self.metadata.iterrows():
            try:
                img_path = row['path']
                
                # Analyze file characteristics
                self._analyze_file_characteristics(img_path, results)
                
                # Analyze image content
                img = cv2.imread(img_path)
                if img is not None:
                    self._analyze_image_content(img, results)
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Processed {idx + 1}/{total_images} images")
                    
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                continue
                
        # Compute final statistics
        self._compute_final_statistics(results)
        
        # Cache results
        self.save_results(results, "technical_analysis.npy")
        return results
        
    def _analyze_file_characteristics(self, img_path: str, results: Dict[str, Any]):
        """Analyze file format and size"""
        # Get file format
        format_type = imghdr.what(img_path) or 'unknown'
        results['format_analysis']['formats'][format_type] = \
            results['format_analysis']['formats'].get(format_type, 0) + 1
            
        # Get file size
        file_size = os.path.getsize(img_path)
        results['size_analysis']['sizes'].append(file_size)
        
    def _analyze_image_content(self, img: np.ndarray, results: Dict[str, Any]):
        """Analyze image content characteristics"""
        # Channel analysis
        channels = img.shape[2] if len(img.shape) > 2 else 1
        results['color_analysis']['channels'][channels] = \
            results['color_analysis']['channels'].get(channels, 0) + 1
            
        # Color statistics
        color_means = np.mean(img, axis=(0, 1))
        color_stds = np.std(img, axis=(0, 1))
        results['color_analysis']['color_stats'].append({
            'means': color_means.tolist(),
            'stds': color_stds.tolist()
        })
        
        # Histogram analysis
        histograms = []
        for i in range(channels):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            histograms.append(hist.flatten().tolist())
        
        results['histogram_analysis']['distributions'].append(histograms)
        results['histogram_analysis']['means'].append(
            [np.mean(hist) for hist in histograms]
        )
        
    def _compute_final_statistics(self, results: Dict[str, Any]):
        """Compute final statistical measures"""
        # Format statistics
        total_files = sum(results['format_analysis']['formats'].values())
        results['format_analysis']['percentages'] = {
            fmt: (count / total_files) * 100
            for fmt, count in results['format_analysis']['formats'].items()
        }
        
        # Size statistics
        sizes = np.array(results['size_analysis']['sizes'])
        results['size_analysis']['statistics'] = {
            'mean': np.mean(sizes),
            'std': np.std(sizes),
            'min': np.min(sizes),
            'max': np.max(sizes),
            'median': np.median(sizes),
            'total': np.sum(sizes)
        }
        
        # Add size categories
        size_categories = {
            'small': (0, 50_000),
            'medium': (50_000, 200_000),
            'large': (200_000, float('inf'))
        }
        
        results['size_analysis']['categories'] = {
            category: np.sum((sizes >= range[0]) & (sizes < range[1]))
            for category, range in size_categories.items()
        }
        
        # Color statistics
        color_stats = np.array(results['color_analysis']['color_stats'])
        results['color_analysis']['summary'] = {
            'means': np.mean(color_stats, axis=0).tolist(),
            'stds': np.std(color_stats, axis=0).tolist()
        }
        
        # Histogram statistics
        hist_means = np.array(results['histogram_analysis']['means'])
        results['histogram_analysis']['summary'] = {
            'mean_distribution': np.mean(hist_means, axis=0).tolist(),
            'std_distribution': np.std(hist_means, axis=0).tolist()
        }