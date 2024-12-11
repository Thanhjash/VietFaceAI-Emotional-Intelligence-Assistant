import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
from threading import Lock

@dataclass
class AnalysisState:
    """State for face analysis results"""
    image: Optional[Any] = None
    results: Optional[Dict] = None
    timestamp: Optional[datetime] = None
    cached_analysis: Optional[Dict] = None  # Store cached analysis results
    model_loaded: bool = False  # Track if model is loaded
    performance: Dict[str, Any] = field(default_factory=lambda: {'processing_time': 0})

@dataclass
class ChatState:
    """State for chat functionality"""
    messages: List[Dict] = field(default_factory=list)
    current_emotion: Optional[str] = None
    context: Optional[Dict] = None
    last_interaction: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """System performance tracking"""
    fps: float = 0
    latency: float = 0
    memory_usage: float = 0
    last_update: Optional[datetime] = None

class StateManager:
    """Thread-safe state manager with enhanced error handling and caching"""
    
    def __init__(self):
        """Initialize StateManager with single initialization check"""
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Single initialization check
        if 'state_initialized' not in st.session_state:
            with self._lock:
                self._initialize_state()
                st.session_state.state_initialized = True
        
        # Always set reference to current state
        self.state = st.session_state.app_state
        
    def _initialize_state(self) -> None:
        """Initialize application state once"""
        try:
            if 'app_state' not in st.session_state:
                st.session_state.app_state = {
                    'analysis': AnalysisState(),
                    'chat': ChatState(),
                    'performance': PerformanceMetrics(),
                    'initialization_time': datetime.now()
                }
                
                # Initialize chat history in session state
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
            self.logger.info("State initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing state: {str(e)}")
            raise

    def get_analysis_state(self) -> AnalysisState:
        """Get current analysis state safely"""
        with self._lock:
            return self.state['analysis']

    def get_chat_state(self) -> ChatState:
        """Get current chat state safely"""
        with self._lock:
            return self.state['chat']

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self._lock:
            return self.state['performance']
        
    def update_analysis(self, image: Any, results: Dict, from_cache: bool = False) -> None:
        """Thread-safe analysis update with cache awareness"""
        try:
            with self._lock:
                start_time = datetime.now()
                
                analysis_state = self.state['analysis']
                analysis_state.image = image
                analysis_state.results = results
                analysis_state.timestamp = datetime.now()
                
                if from_cache:
                    self.logger.info("Using cached analysis results")
                else:
                    # Only store in cache if it's a new analysis
                    analysis_state.cached_analysis = results
                
                # Track performance
                processing_time = (datetime.now() - start_time).total_seconds()
                analysis_state.performance['processing_time'] = processing_time
                self._update_performance_metrics(processing_time)
                
                self.logger.info(f"Analysis updated successfully in {processing_time:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Error updating analysis: {str(e)}")
            raise
            
    def update_chat_state(self, message: Dict, emotion: Optional[str] = None) -> None:
        """Update chat state with history management"""
        try:
            with self._lock:
                chat_state = self.state['chat']
                
                # Add message to both session state and chat state
                st.session_state.chat_history.append(message)
                chat_state.messages = st.session_state.chat_history
                
                if emotion:
                    chat_state.current_emotion = emotion
                chat_state.last_interaction = datetime.now()
                
                self.logger.info("Chat state updated successfully")
                
        except Exception as e:
            self.logger.error(f"Error updating chat state: {str(e)}")
            raise

    def set_emotion(self, emotion: str) -> None:
        """Update current emotion state"""
        try:
            with self._lock:
                self.state['chat'].current_emotion = emotion
                self.state['chat'].last_interaction = datetime.now()
                self.logger.info(f"Emotion updated to: {emotion}")
        except Exception as e:
            self.logger.error(f"Error setting emotion: {str(e)}")
            raise
            
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update system performance metrics"""
        try:
            metrics = self.state['performance']
            current_time = datetime.now()
            
            if metrics.last_update:
                time_diff = (current_time - metrics.last_update).total_seconds()
                if time_diff > 0:
                    metrics.fps = 1.0 / time_diff
                    metrics.latency = processing_time * 1000  # Convert to ms
                    
            metrics.last_update = current_time
            self.logger.debug(f"Performance metrics updated - FPS: {metrics.fps:.2f}, Latency: {metrics.latency:.2f}ms")
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    def clear_chat_history(self) -> None:
        """Clear chat history from both states"""
        try:
            with self._lock:
                self.state['chat'].messages = []
                st.session_state.chat_history = []
                self.state['chat'].context = None
                self.logger.info("Chat history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing chat history: {str(e)}")
            raise

    def clear_analysis(self) -> None:
        """Clear analysis state"""
        try:
            with self._lock:
                self.state['analysis'] = AnalysisState()
                self.logger.info("Analysis state cleared")
        except Exception as e:
            self.logger.error(f"Error clearing analysis state: {str(e)}")
            raise