"""
MARMAR PillSight Configuration
Centralized configuration file for all application settings.
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration settings"""
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv(
        "MONGODB_URI", 
        "mongodb+srv://marmarpillsight:XE714yxuw5wVPpYz@marmar-pillsight.dfxcpbb.mongodb.net/?retryWrites=true&w=majority&appName=marmar-pillsight"
    )
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "marmar-pillsight")
    MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "drug_forms")
    
    # Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    
    # Audio Configuration
    DEFAULT_SAMPLE_RATE = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
    DEFAULT_AUDIO_FORMAT = os.getenv("DEFAULT_AUDIO_FORMAT", "wav")
    SUPPORTED_AUDIO_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'm4a']
    
    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    
    # Search Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    DEFAULT_SEARCH_RESULTS = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    # Streamlit Configuration
    PAGE_TITLE = "🔍 MARMAR PillSight"
    PAGE_ICON = "💊"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"
    
    # Text-to-Speech Configuration
    TTS_RATE = int(os.getenv("TTS_RATE", "150"))
    TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.9"))
    TTS_VOICE = os.getenv("TTS_VOICE", "en-US")
    
    # Speech-to-Text Configuration
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-US")
    STT_TIMEOUT = int(os.getenv("STT_TIMEOUT", "30"))
    
    # WebRTC Configuration
    WEBRTC_ICE_SERVERS = [{"urls": ["stun:stun.l.google.com:19302"]}]
    WEBRTC_AUDIO_RECEIVER_SIZE = int(os.getenv("WEBRTC_AUDIO_RECEIVER_SIZE", "1024"))
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # UI Messages and Texts
    UI_MESSAGES = {
        "welcome": "Welcome to MARMAR PillSight - AI-Powered Drug Discovery",
        "upload_audio": "Upload an audio file containing your drug query",
        "processing_audio": "Processing audio...",
        "transcription_success": "✅ Transcription completed",
        "transcription_error": "Failed to transcribe audio. Please try again.",
        "search_placeholder": "e.g., 'pain relief medication', 'blood pressure tablets', etc.",
        "no_results": "No results found for your query",
        "db_connected": "✅ Database Connected",
        "db_disconnected": "❌ Database Disconnected",
        "model_loaded": "✅ AI Model Loaded",
        "model_failed": "❌ AI Model Failed",
        "google_cloud_available": "☁️ Google Cloud APIs Available",
        "offline_mode": "💻 Using Offline APIs",
        "webrtc_available": "🎙️ Live Recording Available",
        "file_upload_only": "📁 File Upload Only",
        "playing_audio": "🔊 Playing audio...",
    }
    
    # Error Messages
    ERROR_MESSAGES = {
        "mongodb_connection": "Failed to connect to MongoDB",
        "embedding_model": "Failed to load embedding model",
        "speech_recognition": "Speech recognition error",
        "text_to_speech": "Text-to-speech error",
        "vector_search": "Vector search error",
        "audio_processing": "Audio processing error",
        "invalid_query": "Please provide a valid query",
        "database_not_connected": "Database not connected",
        "model_not_available": "Embedding model not available",
    }
    
    # Azure OpenAI Real-time Configuration
    AZURE_OPENAI_ENDPOINT = "https://marmar-pillsight-resource.cognitiveservices.azure.com/openai/realtime"
    AZURE_OPENAI_API_KEY = "G6EVtLQBEI3qUv0tExf1OfqmnV7ukozJFi6Jr05ReRm12qYedsBkJQQJ99BFACHYHv6XJ3w3AAAAACOGrUZt"
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o-realtime-preview"
    AZURE_OPENAI_API_VERSION = "2024-10-01-preview"
    
    # Google Cloud Configuration (for fallback TTS)
    GOOGLE_CLOUD_PROJECT = "your-project-id"  # Update this when you set up GCP
    
    @classmethod
    def get_streamlit_config(cls) -> Dict[str, Any]:
        """Get Streamlit page configuration"""
        return {
            "page_title": cls.PAGE_TITLE,
            "page_icon": cls.PAGE_ICON,
            "layout": cls.LAYOUT,
            "initial_sidebar_state": cls.SIDEBAR_STATE
        }
    
    @classmethod
    def get_mongodb_config(cls) -> Dict[str, str]:
        """Get MongoDB configuration"""
        return {
            "uri": cls.MONGODB_URI,
            "db_name": cls.MONGODB_DB_NAME,
            "collection_name": cls.MONGODB_COLLECTION_NAME
        }
    
    @classmethod
    def get_audio_config(cls) -> Dict[str, Any]:
        """Get audio processing configuration"""
        return {
            "sample_rate": cls.DEFAULT_SAMPLE_RATE,
            "audio_format": cls.DEFAULT_AUDIO_FORMAT,
            "supported_formats": cls.SUPPORTED_AUDIO_FORMATS,
            "tts_rate": cls.TTS_RATE,
            "tts_volume": cls.TTS_VOLUME,
            "tts_voice": cls.TTS_VOICE,
            "stt_language": cls.STT_LANGUAGE,
            "stt_timeout": cls.STT_TIMEOUT
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "max_results": cls.MAX_SEARCH_RESULTS,
            "default_results": cls.DEFAULT_SEARCH_RESULTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "embedding_model": cls.EMBEDDING_MODEL_NAME,
            "max_sequence_length": cls.MAX_SEQUENCE_LENGTH
        }
    
    @classmethod
    def is_google_cloud_available(cls) -> bool:
        """Check if Google Cloud credentials are available"""
        return cls.GOOGLE_APPLICATION_CREDENTIALS is not None
    
    @classmethod
    def get_webrtc_config(cls) -> Dict[str, Any]:
        """Get WebRTC configuration"""
        return {
            "ice_servers": cls.WEBRTC_ICE_SERVERS,
            "audio_receiver_size": cls.WEBRTC_AUDIO_RECEIVER_SIZE
        }

# Create a global config instance
config = Config() 