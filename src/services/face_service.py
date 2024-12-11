import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st
from face_analysis.src.core.face_analyzer import FaceAnalyzer
from face_analysis.app.state.managers.cache_manager import CacheManager
from face_analysis.app.config.model_config import ModelConfig

class FaceAnalysisService:
    """Service layer for face analysis with enhanced error handling"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.analyzer = FaceAnalyzer(self.cache_manager)
            
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze static image with improved error handling"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            if self.cache_manager:
                cached_result = self.cache_manager.get_analysis_result(image)
                if cached_result is not None:
                    return self._prepare_response(cached_result, True)
            
            # New analysis with proper model configuration
            results = self.analyzer.analyze_face(image)
            
            if 'error' in results:
                return self._prepare_error_response(results['error'])
            
            # Cache successful results
            if self.cache_manager:
                self.cache_manager.cache_analysis_result(image, results)
                
            return self._prepare_response(results, False)
            
        except Exception as e:
            return self._prepare_error_response(str(e))
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self._log_performance_metrics(duration)
    
    def _prepare_response(self, results: Dict[str, Any], cached: bool) -> Dict[str, Any]:
        """Prepare success response"""
        return {
            'success': True,
            'analysis': results,
            'marked_image': results.get('marked_image'),
            'timestamp': datetime.now().isoformat(),
            'cached': cached
        }
    
    def _prepare_error_response(self, error: str) -> Dict[str, Any]:
        """Prepare error response with details"""
        return {
            'success': False,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_performance_metrics(self, duration: float):
        """Log performance metrics"""
        pass  # Implement if needed