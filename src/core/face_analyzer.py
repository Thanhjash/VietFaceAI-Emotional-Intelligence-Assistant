import cv2
import numpy as np
import logging
from deepface import DeepFace
from typing import Dict, Any, Optional
import tensorflow as tf
import traceback
from face_analysis.app.state.managers.cache_manager import CacheManager
from face_analysis.app.config.model_config import ModelConfig

class FaceAnalyzer:
    """Main face analysis class with enhanced error handling and logging"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.logger = logging.getLogger('FaceAnalyzer')
        self.logger.setLevel(logging.INFO)
        self.cache_manager = cache_manager
        ModelConfig.configure_gpu()
        self.check_gpu_usage()

    def check_gpu_usage(self):
        """Check and log GPU usage status"""
        try:
            tf_devices = tf.config.list_physical_devices()
            tf_gpu_available = any(device.device_type == "GPU" for device in tf_devices)
            
            self.logger.info(f"TensorFlow devices available: {tf_devices}")
            self.logger.info(f"Using GPU: {tf_gpu_available}")
            
            if tf_gpu_available:
                gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
                self.logger.info(f"GPU Memory usage: {gpu_mem}")
                
            return tf_gpu_available
        except Exception as e:
            self.logger.error(f"GPU check error: {str(e)}")
            return False

    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face with enhanced error handling and caching"""
        try:
            self.logger.info("Starting face analysis...")
            
            # Check cache first
            if self.cache_manager:
                cached_result = self.cache_manager.get_analysis_result(image)
                if cached_result is not None:
                    self.logger.info("Using cached analysis result")
                    return cached_result

            # Get DeepFace config
            deepface_config = ModelConfig.get_deepface_config()
            
            # Use DeepFace's built-in model management (defaults to VGG-Face)
            results = DeepFace.analyze(
                img_path=image,
                actions=['gender', 'age', 'emotion'],
                **deepface_config
            )
            
            if not isinstance(results, list):
                results = [results]
                
            if len(results) == 0:
                return {'error': 'No face detected'}

            result = results[0]
            
            # Process results
            analysis_results = self._process_analysis_results(result, image)
            
            # Cache results if enabled
            if self.cache_manager:
                self.cache_manager.cache_analysis_result(image, analysis_results)
                
            return analysis_results

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Analysis error: {error_msg}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': error_msg}

    def _process_analysis_results(self, result: Dict, image: np.ndarray) -> Dict:
        """Process and format analysis results"""
        # Extract face region
        region = result.get('region', {})
        facial_area = {
            'x': region.get('x', 0),
            'y': region.get('y', 0),
            'w': region.get('w', 0),
            'h': region.get('h', 0)
        }
        
        # Format results
        analysis_results = {
            'face_detected': True,
            'facial_area': facial_area,
            'gender': {
                'label': result['dominant_gender'],
                'confidence': result['gender'][result['dominant_gender']]
            },
            'age': {
                'value': result['age'],
                'confidence': 1.0
            },
            'emotion': {
                'label': result['dominant_emotion'],
                'confidence': result['emotion'][result['dominant_emotion']]
            }
        }

        # Draw results on image
        marked_image = self.draw_results(
            image.copy(),
            facial_area,
            analysis_results
        )
        
        analysis_results['marked_image'] = marked_image
        
        return analysis_results

    def draw_results(self, image: np.ndarray, 
                    facial_area: Dict[str, int], 
                    results: Dict[str, Any]) -> np.ndarray:
        """Draw analysis results on image with error handling"""
        try:
            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']
            
            # Draw face rectangle
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            
            # Prepare text lines
            text_lines = [
                f"Gender: {results['gender']['label']} ({results['gender']['confidence']:.2f})",
                f"Age: {results['age']['value']}",
                f"Emotion: {results['emotion']['label']} ({results['emotion']['confidence']:.2f})"
            ]
            
            # Draw text with background
            for i, text in enumerate(text_lines):
                y_pos = y + h + 20 + i*20
                # Draw black background for better readability
                cv2.putText(image, text, (x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0,0,0), 3)
                # Draw text in green
                cv2.putText(image, text, (x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0,255,0), 2)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Drawing error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return image