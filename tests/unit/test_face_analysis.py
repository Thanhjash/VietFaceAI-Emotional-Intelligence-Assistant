# test_face_analysis.py

import cv2
import numpy as np
from src.core.face_analyzer import FaceAnalyzer, VideoStream

def draw_results(frame, results):
    """Draw analysis results on frame"""
    for result in results:
        if 'region' not in result:
            continue
            
        # Get coordinates
        region = result['region']
        x = region.get('x', region.get('left', 0))
        y = region.get('y', region.get('top', 0))
        w = region.get('w', region.get('right', 0) - x)
        h = region.get('h', region.get('bottom', 0) - y)
        
        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        
        # Prepare text
        age = result.get('age', 'N/A')
        gender = result.get('gender', 'N/A')
        emotion = result.get('emotion', 'N/A')
        
        # Draw results text with better positioning and background
        text_lines = [
            f"Age: {age}",
            f"Gender: {gender}",
            f"Emotion: {emotion}"
        ]
        
        y_offset = y - 10
        for line in text_lines:
            text_size = cv2.getTextSize(
                line, 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )[0]
            
            # Draw background rectangle
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
    
    return frame

def main():
    print("Initializing face analysis...")
    analyzer = FaceAnalyzer(
        detector_backend="retinaface",
        model_name="VGG-Face"
    )
    
    print("Starting video stream...")
    stream = VideoStream(src=0)
    
    if not stream.start():
        print("Error: Could not start video stream")
        return
        
    print("Press 'q' to quit")
    
    try:
        while True:
            frame = stream.read()
            if frame is None:
                break
                
            # Process frame
            results = analyzer.process_frame(frame)
            
            # Draw results
            frame = draw_results(frame, results)
            
            # Add FPS counter
            cv2.imshow("Face Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        print("Cleaning up...")
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()