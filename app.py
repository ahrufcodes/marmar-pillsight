import streamlit as st
import numpy as np
import io
import base64
from typing import List, Dict, Any, Optional
import tempfile
import os
from datetime import datetime
import sys
import soundfile as sf
import sounddevice as sd

# Set page config first
st.set_page_config(
    page_title="MARMAR PillSight",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Audio processing setup with better error handling
AUDIO_AVAILABLE = False
try:
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    st.warning("Audio processing libraries not available. Voice features will be disabled.")

# ML and embeddings
from sentence_transformers import SentenceTransformer
import torch

# Database
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Google Cloud AI Services (HACKATHON REQUIREMENT)
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import storage
from google.cloud import aiplatform
import google.auth

# Conversational AI (REVOLUTIONARY FEATURE!)
from conversational_ai import StreamlitConversationalInterface

# Streamlit WebRTC for real-time audio
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# Configuration
ATLAS_URI = "mongodb+srv://marmarpillsight:XE714yxuw5wVPpYz@marmar-pillsight.dfxcpbb.mongodb.net/?retryWrites=true&w=majority&appName=marmar-pillsight"
DB_NAME = "marmar-pillsight"
COLLECTION_NAME = "drug_forms"

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.is_recording = False
        self.audio_data = None
        
    def process_audio_file(self, audio_file) -> Optional[str]:
        """Process uploaded audio file"""
        if not AUDIO_AVAILABLE:
            st.error("Audio processing is not available in this environment")
            return None
            
        try:
            # Save uploaded file
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            return "temp_audio.wav"
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

class PillSightApp:
    def __init__(self):
        self.setup_page_config()
        self.initialize_models()
        self.initialize_database()
        self.conversational_ai = StreamlitConversationalInterface(self)
        self.audio_available = AUDIO_AVAILABLE
        self.audio_processor = AudioProcessor() if AUDIO_AVAILABLE else None
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        # Page config is already set at the top level
        pass
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load and cache the sentence transformer model"""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            return None
    
    def initialize_models(self):
        """Initialize ML models"""
        with st.spinner("Loading AI models..."):
            self.embedding_model = self.load_embedding_model()
            
    def initialize_database(self):
        """Initialize MongoDB connection with timeout and fallback"""
        try:
            # Add connection timeout to prevent hanging
            self.client = MongoClient(
                ATLAS_URI,
                serverSelectionTimeoutMS=2000,  # 2 second timeout
                connectTimeoutMS=2000,
                socketTimeoutMS=2000
            )
            self.database = self.client[DB_NAME]
            self.collection = self.database[COLLECTION_NAME]
            
            # Test connection with timeout
            self.client.admin.command("ping")
            st.session_state.db_connected = True
            st.success("✅ MongoDB Atlas Connected Successfully!")
            
        except Exception as e:
            st.warning(f"⚠️ MongoDB connection failed: {str(e)[:100]}...")
            st.info("🔄 Running in offline mode with demo data")
            st.session_state.db_connected = False
            self.setup_demo_data()
    
    def setup_demo_data(self):
        """Setup demo medication data for offline mode"""
        self.demo_medications = [
            {
                "drug": "Acetaminophen",
                "gpt4_form": "Tablet, Oral",
                "similarity_score": 0.95,
                "description": "Pain reliever and fever reducer"
            },
            {
                "drug": "Ibuprofen", 
                "gpt4_form": "Tablet, Oral",
                "similarity_score": 0.92,
                "description": "Anti-inflammatory pain reliever"
            },
            {
                "drug": "Aspirin",
                "gpt4_form": "Tablet, Oral", 
                "similarity_score": 0.88,
                "description": "Pain reliever and blood thinner"
            },
            {
                "drug": "Naproxen",
                "gpt4_form": "Tablet, Oral",
                "similarity_score": 0.85,
                "description": "Long-acting anti-inflammatory"
            },
            {
                "drug": "Diphenhydramine",
                "gpt4_form": "Capsule, Oral",
                "similarity_score": 0.80,
                "description": "Antihistamine for allergies and sleep"
            },
            {
                "drug": "Loratadine",
                "gpt4_form": "Tablet, Oral",
                "similarity_score": 0.78,
                "description": "Non-drowsy antihistamine"
            },
            {
                "drug": "Omeprazole",
                "gpt4_form": "Capsule, Oral",
                "similarity_score": 0.75,
                "description": "Proton pump inhibitor for acid reflux"
            },
            {
                "drug": "Simethicone",
                "gpt4_form": "Tablet, Chewable",
                "similarity_score": 0.72,
                "description": "Anti-gas medication"
            }
        ]
        st.info(f"📊 Loaded {len(self.demo_medications)} demo medications")
    
    def record_audio_upload(self):
        """Handle audio file upload"""
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Upload an audio file containing your drug query"
        )
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format=uploaded_file.type)
            
            if st.button("🎵 Process Audio", key="process_upload"):
                with st.spinner("Processing audio..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name
                    
                    try:
                        # Convert audio to text
                        transcript = self.speech_to_text(temp_path)
                        if transcript:
                            st.session_state.user_query = transcript
                            st.success(f"✅ Transcription: {transcript}")
                            self.process_query(transcript)
                        else:
                            st.error("Failed to transcribe audio. Please try again.")
                    finally:
                        # Clean up temporary file
                        os.unlink(temp_path)
    
    def record_audio_live(self):
        """Handle live audio recording"""
        st.subheader("🎙️ Live Voice Recording")
        
        if not WEBRTC_AVAILABLE:
            st.warning("WebRTC support not available. Please install required package:")
            st.code("pip install streamlit-webrtc", language="bash")
            return
            
        # Audio processor class
        class AudioProcessor:
            def __init__(self):
                self.audio_buffer = []
                self.recording = False
                self.sample_rate = 16000
                
            def receive(self, frame):
                if self.recording:
                    self.audio_buffer.append(frame.to_ndarray())
                return frame
            
            def start_recording(self):
                self.recording = True
                self.audio_buffer = []
            
            def stop_recording(self):
                self.recording = False
                if not self.audio_buffer:
                    return None
                # Concatenate audio chunks
                audio_data = np.concatenate(self.audio_buffer, axis=0)
                return audio_data
        
        # Initialize audio processor
        if 'audio_processor' not in st.session_state:
            st.session_state.audio_processor = AudioProcessor()
        
        # WebRTC Configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="voice-recording",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_configuration,
            audio_receiver_size=1024 * 5,  # 5 seconds buffer
            media_stream_constraints={
                "video": False,
                "audio": True,
                "sampleRate": 16000,
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True
            },
            video_processor_factory=None,
            async_processing=True,
        )
        
        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if webrtc_ctx.state.playing:
                if st.button("🎙️ Start Recording", key="start_rec"):
                    st.session_state.audio_processor.start_recording()
                    st.info("🔴 Recording in progress... Speak your question!")
        
        with col2:
            if webrtc_ctx.state.playing and st.session_state.audio_processor.recording:
                if st.button("⏹️ Stop Recording", key="stop_rec"):
                    audio_data = st.session_state.audio_processor.stop_recording()
                    if audio_data is not None:
                        # Save audio data to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            sf.write(tmp_file.name, audio_data, st.session_state.audio_processor.sample_rate)
                            
                            try:
                                # Process the audio
                                transcript = self.speech_to_text(tmp_file.name)
                                if transcript:
                                    st.success(f"✅ Transcription: {transcript}")
                                    self.process_query(transcript)
                                else:
                                    st.error("Could not transcribe audio. Please try again.")
                            except Exception as e:
                                st.error(f"Error processing audio: {str(e)}")
                            finally:
                                os.unlink(tmp_file.name)
                    else:
                        st.warning("No audio recorded. Please try again.")
        
        # Instructions
        if not webrtc_ctx.state.playing:
            st.info("👆 Click 'START' above to enable your microphone")
        elif not st.session_state.audio_processor.recording:
            st.info("🎙️ Click 'Start Recording' when ready to speak")
            
        # Browser compatibility check
        st.markdown("""
        ℹ️ **Browser Compatibility:**
        - Chrome/Edge (Recommended)
        - Firefox (Supported)
        - Safari (Limited Support)
        """)
        
        # Troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips"):
            st.markdown("""
            1. Make sure you've allowed microphone access in your browser
            2. Try refreshing the page if the microphone isn't working
            3. Check if your microphone is working in other applications
            4. Try using Chrome or Edge for best compatibility
            5. Make sure no other application is using your microphone
            """)
    
    def speech_to_text(self, audio_path: str) -> str:
        """Convert speech to text using Google Cloud Speech-to-Text AI"""
        try:
            return self.google_speech_to_text(audio_path)
        except Exception as e:
            st.error(f"Google Cloud Speech-to-Text error: {e}")
            st.info("💡 Make sure your Google Cloud credentials are configured!")
            return ""
    
    def google_speech_to_text(self, audio_path: str) -> str:
        """Google Cloud Speech-to-Text AI Service"""
        try:
            client = speech.SpeechClient()
            
            # Convert to proper format for Google Cloud
            audio_segment = AudioSegment.from_file(audio_path)
            # Convert to mono, 16kHz WAV for best results
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export to bytes
            wav_bytes = io.BytesIO()
            audio_segment.export(wav_bytes, format="wav")
            wav_bytes = wav_bytes.getvalue()
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=wav_bytes)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                model="medical_conversation",  # Better for medical/drug terms
            )
            
            # Perform recognition
            response = client.recognize(config=config, audio=audio)
            
            if response.results:
                transcript = response.results[0].alternatives[0].transcript
                st.success(f"🎯 Google Cloud AI Transcription: {transcript}")
                return transcript
            else:
                st.warning("No speech detected in the audio file.")
                return ""
                
        except Exception as e:
            st.error(f"Google Cloud Speech-to-Text error: {e}")
            raise e
    
    def store_audio_in_cloud(self, audio_path: str) -> str:
        """Store audio file in Google Cloud Storage for processing"""
        try:
            storage_client = storage.Client()
            bucket_name = "marmar-pillsight-audio"  # You'll need to create this bucket
            blob_name = f"audio-queries/{datetime.now().isoformat()}-{os.path.basename(audio_path)}"
            
            # Create bucket if it doesn't exist (for demo purposes)
            try:
                bucket = storage_client.bucket(bucket_name)
                if not bucket.exists():
                    bucket = storage_client.create_bucket(bucket_name)
            except:
                bucket = storage_client.bucket(bucket_name)
            
            # Upload file
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(audio_path)
            
            st.info(f"📁 Audio stored in Google Cloud Storage: gs://{bucket_name}/{blob_name}")
            return f"gs://{bucket_name}/{blob_name}"
            
        except Exception as e:
            st.warning(f"Cloud storage upload failed: {e}")
            return audio_path  # Return local path as fallback
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for the input text"""
        if self.embedding_model is None:
            st.error("Embedding model not available")
            return np.array([])
        
        try:
            embeddings = self.embedding_model.encode([text])
            return embeddings[0]
        except Exception as e:
            st.error(f"Embedding generation error: {e}")
            return np.array([])
    
    def vector_similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar drugs using vector similarity"""
        if not st.session_state.db_connected:
            # Use demo data when MongoDB is not available
            return self.search_demo_data(query_embedding, top_k)
        
        try:
            # For now, we'll do a simple text-based search since the MongoDB doesn't have vector embeddings stored
            # In a production environment, you'd want to pre-compute and store embeddings for all drugs
            
            # Get all documents and compute similarity on the client side (not ideal for production)
            all_docs = list(self.collection.find())
            
            if not all_docs:
                st.warning("No documents found in database")
                return self.search_demo_data(query_embedding, top_k)
            
            # Compute similarities
            similarities = []
            for doc in all_docs:
                drug_text = f"{doc['drug']} {doc['gpt4_form']}"
                doc_embedding = self.generate_embeddings(drug_text)
                
                if len(doc_embedding) > 0 and len(query_embedding) > 0:
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    similarities.append((similarity, doc))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in similarities[:top_k]]
            
        except Exception as e:
            st.error(f"Vector search error: {e}")
            return self.search_demo_data(query_embedding, top_k)
    
    def search_demo_data(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search demo data when MongoDB is not available"""
        try:
            if not hasattr(self, 'demo_medications'):
                self.setup_demo_data()
            
            # Compute similarities with demo data
            similarities = []
            for med in self.demo_medications:
                drug_text = f"{med['drug']} {med['gpt4_form']} {med.get('description', '')}"
                doc_embedding = self.generate_embeddings(drug_text)
                
                if len(doc_embedding) > 0 and len(query_embedding) > 0:
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    med_copy = med.copy()
                    med_copy['similarity_score'] = similarity
                    similarities.append((similarity, med_copy))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            results = [doc for _, doc in similarities[:top_k]]
            
            if results:
                st.info(f"🔍 Found {len(results)} medications using demo data")
            
            return results
            
        except Exception as e:
            st.error(f"Demo search error: {e}")
            return []
    
    def display_drug_results(self, results: List[Dict[str, Any]]):
        """Display drug search results"""
        if not results:
            st.warning("No results found")
            return
        
        st.subheader(f"🔍 Found {len(results)} Similar Drugs")
        
        for i, drug in enumerate(results, 1):
            with st.expander(f"💊 {drug['drug']} - {drug['gpt4_form']}", expanded=i == 1):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Drug Name:**", drug['drug'])
                    st.write("**Form:**", drug['gpt4_form'])
                    st.write("**Best Match:**", drug.get('best_match', 'N/A'))
                
                with col2:
                    st.write("**Similarity Score:**", f"{drug.get('similarity_score', 0):.3f}")
                    st.write("**Best Match Score:**", f"{drug.get('best_match_score', 0):.3f}")
                    st.write("**Agrees with GPT-4:**", "✅" if drug.get('agrees_with_gpt4') else "❌")
                
                # Add text-to-speech button
                if st.button(f"🔊 Read Aloud", key=f"tts_{i}"):
                    drug_info = f"Drug name: {drug['drug']}. Form: {drug['gpt4_form']}"
                    self.text_to_speech(drug_info)
    
    def text_to_speech(self, text: str):
        """Convert text to speech using Google Cloud Text-to-Speech AI"""
        try:
            self.google_text_to_speech(text)
        except Exception as e:
            st.error(f"Google Cloud Text-to-Speech error: {e}")
            st.info("💡 Make sure your Google Cloud credentials are configured!")
    
    def google_text_to_speech(self, text: str):
        """Google Cloud Text-to-Speech AI Service"""
        try:
            client = texttospeech.TextToSpeechClient()
            
            input_text = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",  # High-quality neural voice
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.9,  # Slightly slower for medical terms
                pitch=0.0
            )
            
            response = client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )
            
            # Play audio in Streamlit
            st.audio(response.audio_content, format="audio/mp3")
            st.success("🔊 Google Cloud AI Voice Generated")
            
        except Exception as e:
            st.error(f"Google Cloud Text-to-Speech error: {e}")
            raise e
    
    def get_google_cloud_status(self):
        """Check Google Cloud authentication status"""
        try:
            credentials, project = google.auth.default()
            return True, project
        except Exception as e:
            return False, str(e)
    
    def process_query(self, query: str):
        """Process the user query end-to-end"""
        if not query.strip():
            st.warning("Please provide a valid query")
            return
        
        with st.spinner("Processing your query..."):
            # Generate embeddings
            embeddings = self.generate_embeddings(query)
            
            if len(embeddings) == 0:
                st.error("Failed to generate embeddings")
                return
            
            # Search for similar drugs
            results = self.vector_similarity_search(embeddings)
            
            # Display results
            self.display_drug_results(results)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Database status
            if st.session_state.get('db_connected', False):
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Disconnected")
                if st.button("🔄 Retry Connection"):
                    self.initialize_database()
                    st.rerun()
            
            # Model status
            if self.embedding_model is not None:
                st.success("✅ AI Model Loaded")
            else:
                st.error("❌ AI Model Failed")
            
            # Google Cloud status (HACKATHON REQUIREMENT)
            gc_status, gc_info = self.get_google_cloud_status()
            if gc_status:
                st.success(f"☁️ Google Cloud Connected")
                st.caption(f"Project: {gc_info}")
            else:
                st.error("☁️ Google Cloud Not Configured")
                st.caption("Set up Google Cloud credentials for full functionality")
            
            # WebRTC status
            if WEBRTC_AVAILABLE:
                st.info("🎙️ Live Recording Available")
            else:
                st.info("📁 File Upload Only")
            
            st.markdown("---")
            
            # Search parameters
            st.subheader("🔍 Search Settings")
            top_k = st.slider("Number of results", 1, 10, 5)
            st.session_state.top_k = top_k
    
    def render_main_interface(self):
        """Render the main application interface"""
        st.title("🔍 MARMAR PillSight")
        st.markdown("### 🏆 AI-Powered Drug Discovery with Google Cloud & MongoDB")
        st.info("🎯 **Hackathon Entry**: MongoDB Track | Voice AI + Vector Search + Public Dataset")
        
        # Create tabs for different input methods
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Conversation", "🎙️ Voice Input", "📝 Text Input", "📊 Database Stats"])
        
        with tab1:
            # REVOLUTIONARY CONVERSATIONAL AI FEATURE!
            self.conversational_ai.render_conversation_interface()
        
        with tab2:
            st.markdown("#### Choose your input method:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                self.record_audio_live()
            
            with col2:
                self.record_audio_upload()
        
        with tab3:
            st.subheader("📝 Text Query")
            text_query = st.text_input(
                "Enter your drug query:",
                placeholder="e.g., 'pain relief medication', 'blood pressure tablets', etc.",
                help="Type your drug-related query here"
            )
            
            if st.button("🔍 Search", key="text_search"):
                if text_query:
                    st.session_state.user_query = text_query
                    self.process_query(text_query)
                else:
                    st.warning("Please enter a query")
        
        with tab4:
            self.render_database_stats()
        
        # Display current query if exists
        if hasattr(st.session_state, 'user_query') and st.session_state.user_query:
            st.info(f"🎯 Current Query: {st.session_state.user_query}")
    
    def render_database_stats(self):
        """Render database statistics"""
        if not st.session_state.get('db_connected', False):
            st.error("Database not connected")
            return
        
        try:
            # Get basic stats
            total_docs = self.collection.count_documents({})
            st.metric("Total Drugs", total_docs)
            
            # Get form distribution
            pipeline = [
                {"$group": {"_id": "$gpt4_form", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            form_stats = list(self.collection.aggregate(pipeline))
            
            if form_stats:
                st.subheader("📊 Drug Forms Distribution")
                for stat in form_stats[:10]:  # Show top 10
                    st.write(f"**{stat['_id']}**: {stat['count']} drugs")
            
        except Exception as e:
            st.error(f"Error fetching database stats: {e}")
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'conversation_active' not in st.session_state:
            st.session_state.conversation_active = False
    
    def run(self):
        """Main application runner"""
        # Initialize session state
        if 'user_query' not in st.session_state:
            st.session_state.user_query = ""
        if 'top_k' not in st.session_state:
            st.session_state.top_k = 5
        
        # Render interface
        self.render_sidebar()
        self.render_main_interface()

# Main app execution
if __name__ == "__main__":
    app = PillSightApp()
    app.run() 