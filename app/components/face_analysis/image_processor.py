# face_analysis/app/components/face_analysis/image_processor.py

import cv2
import numpy as np
from typing import Dict, Any, Optional
import streamlit as st
from face_analysis.src.services.face_service import FaceAnalysisService
from face_analysis.app.state.managers.cache_manager import CacheManager

@st.cache_resource
def _get_service():
    """Initialize FaceAnalysisService as a singleton"""
    return FaceAnalysisService()

class ImageProcessor:
    """Component for processing static images"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager
        self.service = FaceAnalysisService(self.cache_manager)
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Process single image with caching"""
        try:
            # Resize if needed
            max_size = 800
            height, width = image.shape[:2]
            if height > max_size or width > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(
                    image, 
                    (int(width * scale), int(height * scale))
                )
            
            # Process image
            results = self.service.analyze_image(image)
            
            if not results['success']:
                return {
                    'success': False,
                    'error': results.get('error', 'Unknown error')
                }
                
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_file_upload(self, file_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Process uploaded file"""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    'success': False,
                    'error': 'Invalid image file'
                }
                
            return self.process_image(image)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'File processing error: {str(e)}'
            }