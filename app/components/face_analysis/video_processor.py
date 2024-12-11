import cv2
import logging
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
from typing import Dict, List, Any, Optional
from face_analysis.app.state.managers.state_manager import StateManager
from face_analysis.app.config.model_config import ModelConfig

class VideoProcessor(VideoProcessorBase):
    """Video processor with enhanced state management and error handling"""
    
    def __init__(self, face_analyzer, state_manager: Optional[StateManager] = None):
        self.logger = logging.getLogger('VideoProcessor')
        self.face_analyzer = face_analyzer
        self.state_manager = state_manager or StateManager()
        self.frame_count = 0
        self._init_cache()
        
    def _init_cache(self):
        """Initialize cache with configuration"""
        self.cache = {}
        self.cache_ttl = 1  # 1 second cache TTL
        
    @st.cache_data(ttl=1)
    def process_frame_cached(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame with caching and error handling"""
        try:
            return self.face_analyzer.analyze_face(frame)
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return {'error': str(e)}
        
    def recv(self, frame):
        """Process received frame with enhanced error handling"""
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame
            results = self.process_frame_cached(img)
            
            if 'error' in results:
                self.logger.error(f"Analysis error: {results['error']}")
                return frame
            
            # Draw results and update state
            if results.get('face_detected', False):
                img = self.draw_results(img.copy(), results)
                self._update_analysis_state(results)
                
            return frame.from_ndarray(img)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return frame

    def _update_analysis_state(self, results: Dict[str, Any]):
        """Update analysis state through state manager"""
        try:
            analysis_data = {
                'age': results.get('age', {}).get('value', 'N/A'),
                'gender': results.get('gender', {}).get('label', 'N/A'),
                'emotion': results.get('emotion', {}).get('label', 'N/A'),
                'confidence': {
                    'age': results.get('age', {}).get('confidence', 0),
                    'gender': results.get('gender', {}).get('confidence', 0),
                    'emotion': results.get('emotion', {}).get('confidence', 0)
                }
            }
            
            self.state_manager.update_analysis(
                image=None,  # Don't store the image
                results=analysis_data
            )
            
        except Exception as e:
            self.logger.error(f"State update error: {str(e)}")

    def draw_results(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Draw analysis results with improved visualization"""
        try:
            facial_area = results.get('facial_area', {})
            if not facial_area:
                return frame
                
            x = facial_area.get('x', 0)
            y = facial_area.get('y', 0)
            w = facial_area.get('w', 0)
            h = facial_area.get('h', 0)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare text lines with confidence scores
            text_lines = [
                f"Age: {results.get('age', {}).get('value', 'N/A')} ({results.get('age', {}).get('confidence', 0):.2f})",
                f"Gender: {results.get('gender', {}).get('label', 'N/A')} ({results.get('gender', {}).get('confidence', 0):.2f})",
                f"Emotion: {results.get('emotion', {}).get('label', 'N/A')} ({results.get('emotion', {}).get('confidence', 0):.2f})"
            ]
            
            self._draw_text_results(frame, x, y, text_lines)
            return frame
            
        except Exception as e:
            self.logger.error(f"Drawing error: {str(e)}")
            return frame

    def _draw_text_results(self, frame: np.ndarray, x: int, y: int, text_lines: List[str]):
        """Draw text results with improved visibility"""
        try:
            y_offset = y - 10
            
            for line in text_lines:
                text_size = cv2.getTextSize(
                    line, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2
                )[0]
                
                # Draw background for better readability
                cv2.rectangle(
                    frame,
                    (x, y_offset - text_size[1] - 5),
                    (x + text_size[0], y_offset + 5),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    line,
                    (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                y_offset -= text_size[1] + 10
                
        except Exception as e:
            self.logger.error(f"Text drawing error: {str(e)}")