import streamlit as st
import cv2
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

# Add CS_AGE to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from face_analysis.src.services.face_service import FaceAnalysisService
from face_analysis.src.services.chat_service import ChatService
from face_analysis.app.state.managers.state_manager import StateManager
from face_analysis.app.state.managers.cache_manager import CacheManager
from face_analysis.app.state.event_bus.event_bus import EventBus
from face_analysis.app.components.emoti_chat.chat_ui import ChatUI
from face_analysis.app.components.face_analysis.image_processor import ImageProcessor
from face_analysis.app.components.face_analysis.video_processor import VideoProcessor
from face_analysis.app.config.app_config import AppConfig
from face_analysis.app.config.model_config import ModelConfig

# Setup logging
AppConfig.setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource
def init_system() -> Tuple[StateManager, EventBus, CacheManager]:
    """Initialize core system components"""
    try:
        ModelConfig.configure_gpu()  # Configure GPU settings
        state_manager = StateManager()
        event_bus = EventBus()
        cache_manager = CacheManager(max_memory_mb=AppConfig.CACHE_CONFIG['max_memory_mb'])
        logger.info("System components initialized successfully")
        return state_manager, event_bus, cache_manager
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise

@st.cache_resource
def init_processors(_state_manager: StateManager, _cache_manager: CacheManager) -> Tuple[ImageProcessor, FaceAnalysisService, ChatService]:
    """
    Initialize processors and services with proper dependency injection
    
    Args:
        _state_manager: StateManager instance
        _cache_manager: CacheManager instance
        (underscore prefix to avoid streamlit warning)
    """
    try:
        return (
            ImageProcessor(_cache_manager),
            FaceAnalysisService(_cache_manager),
            ChatService(_state_manager)
        )
    except Exception as e:
        logger.error(f"Error initializing processors: {str(e)}")
        raise

def camera_capture() -> Optional[np.ndarray]:
    """Capture image from camera"""
    try:
        camera_image = st.camera_input(
            "Take a picture",
            key="camera_input"
        )
        
        if camera_image:
            bytes_data = camera_image.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            return image
        return None
    except Exception as e:
        logger.error(f"Error capturing camera image: {str(e)}")
        st.error("Error accessing camera. Please try again.")
        return None

def render_analysis_results(state_manager: StateManager, analysis: dict, event_bus: EventBus):
    """Render analysis results with metrics"""
    try:
        if analysis.get('cached', False):
            st.info("Using cached analysis result")
        
        st.success("Analysis Complete!")
        
        with st.expander("Gender", expanded=True):
            st.metric(
                "Prediction",
                analysis['gender']['label'],
                f"Confidence: {analysis['gender']['confidence']:.2f}"
            )
            
        with st.expander("Age", expanded=True):
            st.metric(
                "Prediction",
                analysis['age']['value'],
                "Years"
            )
            
        with st.expander("Emotion", expanded=True):
            st.metric(
                "Prediction",
                analysis['emotion']['label'],
                f"Confidence: {analysis['emotion']['confidence']:.2f}"
            )
            # Update emotion state and publish event
            state_manager.set_emotion(analysis['emotion']['label'])
            event_bus.publish('emotion_update', analysis['emotion'])
    except Exception as e:
        logger.error(f"Error rendering analysis results: {str(e)}")
        st.error("Error displaying analysis results.")

def process_image(image: np.ndarray, image_processor: ImageProcessor, 
                 state_manager: StateManager, event_bus: EventBus):
    """Process image and update state"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(
                image,
                channels="BGR",
                caption="Original/Captured Image",
                use_column_width=True
            )
        
        with col2:
            with st.spinner('Analyzing...'):
                results = image_processor.process_image(image)
                
                if results['success']:
                    st.image(
                        results['marked_image'],
                        channels="BGR",
                        caption="Analysis Results",
                        use_column_width=True
                    )
                    
                    # Update state
                    state_manager.update_analysis(image, results)
                    render_analysis_results(state_manager, results['analysis'], event_bus)
                else:
                    st.error(f"Error: {results['error']}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error("Error processing image. Please try again.")

def render_face_analysis(state_manager: StateManager, image_processor: ImageProcessor, 
                        event_bus: EventBus):
    """Render face analysis section"""
    try:
        st.markdown("""
        ## Face Analysis
        This system analyzes faces to detect:
        - Gender
        - Age
        - Emotion
        """)
        
        tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Camera"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=AppConfig.UI_CONFIG['supported_formats'],
                key="face_upload"
            )
            
            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                process_image(image, image_processor, state_manager, event_bus)
        
        with tab2:
            image = camera_capture()
            if image is not None:
                process_image(image, image_processor, state_manager, event_bus)
    except Exception as e:
        logger.error(f"Error rendering face analysis: {str(e)}")
        st.error("Error in face analysis component. Please refresh the page.")

def main():
    try:
        # Page config
        st.set_page_config(
            page_title="Face Analysis System",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Initialize system components (cached)
        state_manager, event_bus, cache_manager = init_system()
        
        # Initialize processors and services (cached)
        image_processor, face_service, chat_service = init_processors(state_manager, cache_manager)
        
        # App title with custom styling
        st.markdown("""
            <h1 style='text-align: center; color: #1E88E5;'>
                Face Analysis System
                <span style='font-size: 24px;'>🎭</span>
            </h1>
            <hr>
        """, unsafe_allow_html=True)
        
        # Main layout with face analysis and chat
        face_col, chat_col = st.columns([0.6, 0.4])
        
        with face_col:
            render_face_analysis(state_manager, image_processor, event_bus)
        
        with chat_col:
            chat_ui = ChatUI(state_manager=state_manager, chat_service=chat_service)
            chat_ui.render()
            
        # Add performance metrics in sidebar
        with st.sidebar:
            st.markdown("### System Stats")
            stats = cache_manager.get_cache_stats()
            st.write("Cache Statistics:")
            st.write(f"- Analysis Cache Size: {stats['analysis_cache_size']}")
            st.write(f"- Model Cache Size: {stats['model_cache_size']}")
            st.write(f"- Memory Usage: {stats['memory_usage_mb']} MB")
            st.write(f"- Last Cleanup: {stats['last_cleanup']}")
            if AppConfig.CACHE_CONFIG['cache_monitoring']:
                st.write(f"- Cache Hits: {stats['cache_hits']}")
                st.write(f"- Cache Hit Rate: {stats['hit_rate']}%")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page or contact support.")

if __name__ == "__main__":
    main()