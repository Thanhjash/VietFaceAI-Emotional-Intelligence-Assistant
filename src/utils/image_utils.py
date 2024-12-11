# src/utils/image_utils.py

from typing import Union, Tuple, Any
import cv2
import numpy as np

class ImageUtils:
    @staticmethod
    def convert_to_cv2(image_input: Union[bytes, np.ndarray, Any]) -> np.ndarray:
        """Convert various image formats to cv2"""
        try:
            if isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            elif hasattr(image_input, 'read'):
                bytes_data = image_input.read()
                nparr = np.frombuffer(bytes_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            elif isinstance(image_input, np.ndarray):
                return image_input.copy()
            
            else:
                raise ValueError(f"Unsupported image type: {type(image_input)}")
                
        except Exception as e:
            raise ValueError(f"Error converting image: {str(e)}")

    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize image"""
        return cv2.resize(image, target_size)