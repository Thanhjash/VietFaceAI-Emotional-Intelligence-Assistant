# face_analysis/src/services/chat_service.py

from typing import Dict, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
import streamlit as st
from face_analysis.app.state.managers.state_manager import StateManager

class ChatService:
    """Enhanced chat service with Gemma integration and emotion awareness"""
    
    BASE_PROMPT = """You are EmotiChat, an empathetic AI assistant specializing in emotional support. Follow these rules:
    1. **Adapt to Emotion**:
       - Acknowledge emotions (e.g., "I see you're feeling {emotion}.").
       - Tailor your tone and response to the emotion:
         - Happy 😊: Celebrate with them.
         - Anxious 😟: Reassure calmly.
         - Angry 😠: Acknowledge frustration, offer solutions.
         - Sad 😢: Show care, encourage positivity.
         - Fearful 😨: Reassure safety.
         - Neutral 🙂: Be balanced and supportive.
    2. **Concise & Empathetic**:
       - Keep responses short (2-3 sentences).
       - Use empathetic language and emojis.
    3. **Follow-Up**:
       - Ask thoughtful questions or offer helpful suggestions.
    Human: {input}
    Assistant:"""
        
    def __init__(self, state_manager: StateManager):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info("Initializing ChatService...")
        self.state_manager = state_manager
        
        self.logger.info("Starting Gemma initialization...")
        self._initialize_gemma()
        
        self.logger.info("Ensuring chat state...")
        self._ensure_initialized()
        
        if hasattr(self, 'model') and hasattr(self, 'tokenizer'):
            self.logger.info("ChatService fully initialized with model")
        else:
            self.logger.error("Model or tokenizer not initialized properly")

    def _initialize_gemma(self) -> None:
        """Initialize Gemma model with optimized settings"""
        try:
            self.logger.info("Starting Gemma initialization...")
            
            cuda_available = torch.cuda.is_available()
            self.logger.info(f"CUDA available: {cuda_available}")
            
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
            
            self.device = "cuda" if cuda_available else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            self.logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2-2b-it",
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if not hasattr(self, 'model'):
                raise Exception("Model failed to load")
                
            self.logger.info(f"Gemma initialized successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing Gemma: {str(e)}")
            raise

    def _ensure_initialized(self) -> None:
        """Ensure chat state is properly initialized"""
        try:
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
                self.logger.info("Chat state initialized")
        except Exception as e:
            self.logger.error(f"Error initializing chat state: {str(e)}")
            raise

    def generate_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        """Generate emotion-aware response using Gemma"""
        try:
            self.logger.info("=== Starting Response Generation ===")
            if emotion is None:
                emotion = self.get_current_emotion()
            
            prompt = self.BASE_PROMPT.format(emotion=emotion or "neutral", input=user_input)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    min_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            response = response.split("Human:")[0].strip()
            
            # Clean the response
            response = self._clean_response(response)

            return response

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I'm having trouble processing that right now. Could you try again?"

    def add_message(self, content: str, is_user: bool, emotion: Optional[str] = None) -> Dict:
        """Add new message to chat history"""
        try:
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
                
            if st.session_state.chat_messages:
                last_msg = st.session_state.chat_messages[-1]
                if last_msg['content'] == content and last_msg['is_user'] == is_user:
                    return last_msg
                    
            message = {
                'content': content,
                'is_user': is_user,
                'emotion': emotion,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.chat_messages.append(message)
            
            # Thay đổi cách log để tránh emoji
            try:
                # Chỉ log những ký tự ASCII
                ascii_content = ''.join(char for char in content[:50] if ord(char) < 128)
                self.logger.debug(f"Added message: {ascii_content}...")
            except:
                self.logger.debug("Added new message (content contains non-ASCII characters)")
                
            return message

        except Exception as e:
            self.logger.error(f"Error adding message: {str(e)}")
            raise
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model's response"""
        # Remove any remaining special tokens or artifacts
        response = response.replace("<s>", "").replace("</s>", "")
        # Remove multiple newlines
        response = " ".join(response.split())
        # Ensure proper emoji spacing
        response = response.replace("  ", " ")
        return response.strip()

    def get_history(self) -> List[Dict]:
        """Get chat history from session state"""
        return st.session_state.get('chat_messages', [])

    def clear_history(self) -> None:
        """Clear chat history"""
        try:
            st.session_state.chat_messages = []
            self.logger.info("Chat history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing chat history: {str(e)}")
            raise

    def update_emotion(self, emotion: str, confidence: float) -> None:
        """Update current emotion in state"""
        try:
            emotion_data = {
                'value': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            self.state_manager.set_emotion(emotion)
            self.logger.debug(f"Updated emotion: {emotion} ({confidence:.2f})")
        except Exception as e:
            self.logger.error(f"Error updating emotion: {str(e)}")
            raise

    def get_current_emotion(self) -> Optional[str]:
        """Get current emotion from state"""
        try:
            chat_state = self.state_manager.get_chat_state()
            return chat_state.current_emotion
        except Exception as e:
            self.logger.error(f"Error getting current emotion: {str(e)}")
            return None

    def process_message_chain(self, user_input: str, emotion: Optional[str] = None) -> Dict:
        """Process complete message chain with error recovery"""
        try:
            current_emotion = emotion or self.get_current_emotion()
            
            with torch.inference_mode():
                user_message = self.add_message(user_input, True, current_emotion)
                response = self.generate_response(user_input, current_emotion)
                bot_message = self.add_message(response, False, current_emotion)
                
                return {
                    'user_message': user_message,
                    'bot_message': bot_message,
                    'emotion': current_emotion,
                    'success': True
                }
        except Exception as e:
            self.logger.error(f"Error in message chain: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def monitor_performance(self) -> Dict:
        """Monitor chat service performance"""
        try:
            return {
                'device': self.device,
                'model_loaded': bool(self.model),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'max_memory': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {str(e)}")
            return {}