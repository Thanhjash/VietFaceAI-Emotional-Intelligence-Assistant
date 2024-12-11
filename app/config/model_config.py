import logging
from typing import Dict, Any
import tensorflow as tf

class ModelConfig:
    # DeepFace analysis configurations
    DEEPFACE_CONFIG = {
        'detector_backend': 'retinaface',  # Fast and accurate face detection
        'enforce_detection': False,        # Don't raise error if face not detected
        'align': True,                    # Align faces for better accuracy
        'silent': True                    # Suppress DeepFace logs
    }
    
    # GPU Configuration for optimal performance
    GPU_CONFIG = {
        'allow_growth': True,              # Grow GPU memory as needed
        'per_process_gpu_memory_fraction': 0.7,  # Use up to 70% GPU memory
        'memory_limit': 4096               # 4GB limit
    }
    
    @classmethod
    def get_deepface_config(cls) -> Dict[str, Any]:
        """Get DeepFace analysis configuration"""
        return cls.DEEPFACE_CONFIG.copy()
    
    @classmethod
    def configure_gpu(cls) -> None:
        """Configure GPU settings with error handling"""
        logger = logging.getLogger(__name__)
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(
                        gpu, 
                        cls.GPU_CONFIG['allow_growth']
                    )
                
                # Set memory limit
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=cls.GPU_CONFIG['memory_limit']
                    )]
                )
                logger.info(f"GPU configured successfully with {len(gpus)} device(s)")
        except Exception as e:
            logger.error(f"GPU configuration error: {str(e)}")
            raise