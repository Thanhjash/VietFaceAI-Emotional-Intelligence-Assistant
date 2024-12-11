from pathlib import Path
from typing import Dict, Any
import logging.handlers
import os

class AppConfig:
    """Application-wide configuration"""
    
    # Cache settings
    CACHE_CONFIG = {
        'analysis_ttl': 300,  # 5 minutes
        'max_cache_size': 100,  # Maximum number of cached results
        'cleanup_interval': 3600,  # Cleanup every hour
        'max_memory_mb': 1024,  # Maximum memory usage in MB
        'cache_monitoring': True,  # Enable cache monitoring
    }
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    CACHE_DIR = DATA_DIR / 'cache'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Create directories
    for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Logging settings
    LOG_FILE = LOGS_DIR / 'face_analysis.log'
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console handler
                logging.FileHandler(cls.LOG_FILE)  # File handler
            ]
        )
    
    # UI Configuration
    UI_CONFIG = {
        'max_image_size': (800, 800),
        'supported_formats': ['jpg', 'jpeg', 'png'],  # Removed dots for consistency
        'chat_history_limit': 100,
        'chat_container_height': 400,  # Height in pixels
        'enable_animations': True,
        'show_timestamps': True,
    }
    
    # Face Analysis Configuration
    FACE_CONFIG = {
        'min_face_size': 20,
        'detection_confidence': 0.5,
        'enable_age_prediction': True,
        'enable_gender_prediction': True,
        'enable_emotion_prediction': True,
    }