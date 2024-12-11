# face_analysis/app/components/emoti_chat/chat_ui.py

import streamlit as st 
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from face_analysis.src.services.chat_service import ChatService
from face_analysis.app.state.managers.state_manager import StateManager

@dataclass
class ChatUIComponents:
    @staticmethod
    def message_container(message_html: str) -> str:
        return (
            f'<div class="chat-container" id="chat-container">'
            f'{message_html}'
            f'</div>'
        )

    @staticmethod
    def emotion_badge(emotion: str) -> str:
        colors = {
            'happy': ('#4CAF50', '#E8F5E9'),     
            'sad': ('#5C6BC0', '#E8EAF6'),       
            'angry': ('#F44336', '#FFEBEE'),     
            'surprised': ('#FF9800', '#FFF3E0'), 
            'fear': ('#7E57C2', '#EDE7F6'),   
            'disgusted': ('#795548', '#EFEBE9'), 
            'neutral': ('#78909C', '#ECEFF1')    
        }
        
        emoji_map = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'surprised': '😮', 'fear': '😨',
            'disgusted': '😖', 'neutral': '😐'
        }
        
        if not emotion or not isinstance(emotion, str):
            return ""
        emotion = emotion.lower()
        if emotion not in colors:
            return ""
            
        color, bg = colors[emotion]
        emoji = emoji_map[emotion]
        return f'<div class="emotion-badge" style="color:{color};background-color:{bg};border:1px solid {color}40">{emoji} {emotion}</div>'

    @staticmethod
    def message_bubble(content: str, is_user: bool, emotion: Optional[str], timestamp: str) -> str:
        message_class = "chat-message-user" if is_user else "chat-message-bot"
        emotion_display = ChatUIComponents.emotion_badge(emotion) if emotion else ""
        return (
            f'<div class="{message_class}">'
            f'<div class="message-content">{content}</div>'
            f'{emotion_display}'
            f'<div class="timestamp">{timestamp}</div>'
            f'</div>'
        )

class ChatUI:
    def __init__(self, state_manager: StateManager, chat_service: ChatService):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.state_manager = state_manager
        self.chat_service = chat_service
        self.components = ChatUIComponents()
        
        if not hasattr(st.session_state, 'chat_ui_initialized'):
            self._init_session_state()
            self._init_styles()
            st.session_state.chat_ui_initialized = True

    def _init_session_state(self) -> None:
        if 'chat_input' not in st.session_state:
            st.session_state.chat_input = ''
            
    def _init_styles(self) -> None:
        st.markdown("""
            <style>
            .chat-container {
                max-height: 60vh;
                min-height: 400px;
                height: calc(100vh - 300px);
                overflow-y: auto;
                padding: 20px;
                background-color: #FFFFFF;
                border-radius: 12px;
                margin-bottom: 20px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.1);
                scrollbar-width: thin;
                scrollbar-color: #90A4AE #F5F5F5;
            }
            
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: #F5F5F5;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: #90A4AE;
                border-radius: 3px;
            }
            
            .chat-container::-webkit-scrollbar-thumb:hover {
                background: #78909C;
            }
            
            .chat-message-user {
                background: linear-gradient(135deg, #0D47A1, #1976D2);
                color: white;
                padding: 15px;
                border-radius: 18px;
                border-bottom-right-radius: 4px;
                margin: 10px 0;
                max-width: min(80%, 600px);
                float: right;
                clear: both;
                box-shadow: 0 4px 15px rgba(25,118,210,0.2);
                animation: slideInRight 0.3s ease-out;
            }
            
            .chat-message-bot {
                background-color: #FFFFFF;
                color: #37474F;
                padding: 15px;
                border-radius: 18px;
                border-bottom-left-radius: 4px;
                margin: 10px 0;
                max-width: min(80%, 600px);
                float: left;
                clear: both;
                border: 1px solid #E0E0E0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                animation: slideInLeft 0.3s ease-out;
            }
            
            .message-content {
                line-height: 1.6;
                font-size: 15px;
                letter-spacing: 0.3px;
                word-wrap: break-word;
            }
            
            .emotion-badge {
                font-size: 12px;
                padding: 4px 12px;
                margin-top: 8px;
                border-radius: 20px;
                font-weight: 500;
                display: inline-block;
                animation: fadeIn 0.3s ease-out;
            }
            
            .timestamp {
                font-size: 10px;
                margin-top: 6px;
                opacity: 0.8;
            }
            
            .chat-message-user .timestamp {
                color: rgba(255,255,255,0.9);
            }
            
            .chat-message-bot .timestamp {
                color: #90A4AE;
            }
            
            @keyframes slideInRight {
                from {transform: translateX(30px); opacity: 0;}
                to {transform: translateX(0); opacity: 1;}
            }
            
            @keyframes slideInLeft {
                from {transform: translateX(-30px); opacity: 0;}
                to {transform: translateX(0); opacity: 1;}
            }
            
            @keyframes fadeIn {
                from {opacity: 0;}
                to {opacity: 1;}
            }

            @media (max-width: 768px) {
                .chat-container {
                    height: calc(100vh - 250px);
                }
                .chat-message-user,
                .chat-message-bot {
                    max-width: 90%;
                }
            }
            </style>
        """, unsafe_allow_html=True)
            
    def _render_header(self) -> None:
        st.markdown("""
            <div style='margin-bottom: 25px;'>
                <h3 style='color: #1976D2; margin: 0; display: flex; align-items: center; gap: 8px;'>
                    💬 <span style='background: linear-gradient(135deg, #0D47A1, #1976D2);
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                        EmotiChat
                    </span>
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        current_emotion = self.state_manager.get_chat_state().current_emotion
        if current_emotion:
            st.markdown(f"""
                <div style='
                    padding: 12px 16px;
                    background: linear-gradient(135deg, #0D47A1, #1976D2);
                    color: white;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    font-size: 15px;
                    letter-spacing: 0.3px;
                    box-shadow: 0 2px 8px rgba(25,118,210,0.2);
                '>
                    <span style='opacity: 0.9;'>Current Emotion:</span>
                    {self.components.emotion_badge(current_emotion)}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👋 Upload or capture an image to start emotion-aware chat!")

    def _render_messages(self) -> None:
        try:
            messages = self.chat_service.get_history()
            messages_html = "".join(
                self.components.message_bubble(
                    msg['content'],
                    msg.get('is_user', False),
                    msg.get('emotion'),
                    datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
                ) for msg in messages
            )
            
            st.markdown(
                self.components.message_container(messages_html) + 
                '<script>setTimeout(function(){document.getElementById("chat-container").scrollTop=1e6},100)</script>',
                unsafe_allow_html=True
            )
        except Exception as e:
            self.logger.error(f"Error rendering messages: {str(e)}")
            st.error("Chat error. Please refresh.")

    def _handle_input(self) -> None:
        message = st.session_state.get('chat_input', '').strip()
        if not message:
            return
            
        if len(message) > 500:
            st.error("Message too long! Please keep it under 500 characters.")
            return
            
        current_emotion = self.state_manager.get_chat_state().current_emotion
        self.logger.info(f"Processing message with emotion: {current_emotion}")
        
        try:
            # Add user message & generate response
            self.chat_service.add_message(message, True, current_emotion)
            response = self.chat_service.generate_response(message, current_emotion)
            self.chat_service.add_message(response, False, current_emotion)
            
            # Clear input
            st.session_state.chat_input = ''
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self.chat_service.add_message(
                "I'm having trouble right now. Please try again.",
                False
            )

    def render(self) -> None:
        with st.container():
            self._render_header()
            self._render_messages()
            
            col1, col2 = st.columns([6,1])
            with col1:
                st.text_input(
                    "Message",
                    key="chat_input",
                    placeholder="Type your message here...",
                    on_change=self._handle_input,
                    label_visibility="collapsed"
                )
            with col2:
                if st.button("Clear", key="clear_chat"):
                    self.chat_service.clear_history()

    def _clean_message_content(self, content: str) -> str:
        """Clean and escape message content for safe rendering"""
        # Remove code block markers that might cause rendering issues
        content = content.replace('```', '')
        # Escape HTML special characters except emojis
        content = (
            content
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
        )
        return content

    