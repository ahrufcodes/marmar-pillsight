"""
Conversational AI Interface for MARMAR PillSight
Enables natural conversation about medications and symptoms
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import tempfile
import os

class StreamlitConversationalInterface:
    """Streamlit interface for conversational AI"""
    
    def __init__(self, pill_sight_app):
        self.app = pill_sight_app
    
    def search_medications_wrapper(self, query: str) -> List[Dict[str, Any]]:
        """Wrapper for the app's medication search function"""
        try:
            embeddings = self.app.generate_embeddings(query)
            if len(embeddings) > 0:
                results = self.app.vector_similarity_search(embeddings, top_k=5)
                return results
            else:
                # Fallback to simple text matching if embeddings fail
                return self.simple_text_search(query)
        except Exception as e:
            st.warning(f"Search error: {e}")
            return self.simple_text_search(query)
    
    def simple_text_search(self, query: str) -> List[Dict[str, Any]]:
        """Simple text-based search fallback"""
        if not hasattr(self.app, 'demo_medications'):
            self.app.setup_demo_data()
        
        query_lower = query.lower()
        results = []
        
        for med in self.app.demo_medications:
            # Simple keyword matching
            drug_text = f"{med['drug']} {med['gpt4_form']} {med.get('description', '')}".lower()
            
            # Check for keyword matches
            if any(word in drug_text for word in query_lower.split()):
                med_copy = med.copy()
                med_copy['similarity_score'] = 0.7  # Default score for text match
                results.append(med_copy)
        
        return results[:5]
    
    def render_conversation_interface(self):
        """Render the conversational AI interface"""
        st.markdown("### 🤖 AI Medical Assistant")
        
        # Initialize conversation state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'conversation_active' not in st.session_state:
            st.session_state.conversation_active = False
        if 'user_input' not in st.session_state:
            st.session_state.user_input = ""
        
        # Minimal Control Buttons
        if not st.session_state.conversation_active:
            if st.button("🚀 Start Conversation", key="start_conv_btn"):
                st.session_state.conversation_active = True
                st.session_state.conversation_history = []
                welcome_msg = "Hello! What symptoms or medication questions do you have today?"
                st.session_state.conversation_history.append({"role": "assistant", "content": welcome_msg})
                st.rerun()
        else:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🔄 Reset", key="clear_history_btn", help="Start a new conversation"):
                    st.session_state.conversation_history = []
                    st.session_state.conversation_active = False
                    st.rerun()
        
        # Active Conversation Interface
        if st.session_state.conversation_active:
            # Display conversation history in a clean chat format
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.conversation_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="background-color: #0066cc; color: white; padding: 12px; border-radius: 15px; margin: 8px 0; margin-left: 20%; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #ffffff; color: #333333; padding: 12px; border-radius: 15px; margin: 8px 0; margin-right: 20%; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            {msg['content']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Simplified Input Section - Now works with Enter key
            user_input = st.text_input(
                "",  # Removed label for cleaner look
                placeholder="Describe your symptoms or ask about medications... (Press Enter to send)",
                key="user_input",
                on_change=self.handle_input_change
            )
            
            # Minimal Quick Response Section
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💊 Describe Pain", key="pain_btn"):
                    self.handle_continuous_conversation("I'm experiencing pain and need medication suggestions")
                    st.rerun()
            with col2:
                if st.button("❌ No Previous Conditions", key="no_conditions_btn"):
                    self.handle_continuous_conversation("I don't have any previous medical conditions")
                    st.rerun()
        
        # Minimal Starter Suggestion
        else:
            st.markdown("---")
            if st.button("💊 'I have a headache'", key="headache_btn"):
                self.start_conversation("I have a headache and need some relief")
    
    def handle_input_change(self):
        """Handle changes to the input field"""
        if st.session_state.user_input and st.session_state.user_input.strip():
            user_message = st.session_state.user_input.strip()
            self.handle_continuous_conversation(user_message)
            # Clear the input
            st.session_state.user_input = ""
            st.rerun()
    
    def handle_continuous_conversation(self, user_message: str):
        """Handle continuous back-and-forth conversation"""
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_message})
        
        # Get AI response based on conversation context
        ai_response = self.get_contextual_ai_response(user_message, st.session_state.conversation_history)
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Check if we should search for medications
        should_search = any(word in user_message.lower() for word in [
            "need", "find", "looking for", "help", "pain", "headache", "medication", 
            "medicine", "drug", "relief", "treatment", "sharp", "dull", "throbbing"
        ])
        
        if should_search:
            # Combine user message with recent context for better search
            context_messages = [msg['content'] for msg in st.session_state.conversation_history[-4:] if msg['role'] == 'user']
            search_query = " ".join(context_messages)
            
            results = self.search_medications_wrapper(search_query)
            if results:
                med_response = self.create_medication_response(results)
                st.session_state.conversation_history.append({"role": "assistant", "content": med_response})
    
    def get_contextual_ai_response(self, user_message: str, conversation_history: list) -> str:
        """Get AI response based on conversation context"""
        user_lower = user_message.lower()
        
        # Check recent conversation context
        recent_messages = [msg['content'].lower() for msg in conversation_history[-3:]]
        context = " ".join(recent_messages)
        
        # Responses to follow-up questions
        if "sharp" in user_lower or "throbbing" in user_lower:
            return "Sharp, throbbing pain often indicates a migraine. These can be more severe and may require specific treatments. Do you experience nausea, sensitivity to light, or visual disturbances with these headaches? This will help me recommend the most appropriate medication."
        
        elif "dull" in user_lower or "constant" in user_lower or "tension" in user_lower:
            return "A dull, constant ache sounds like a tension headache. These are often caused by stress, poor posture, or muscle tension. Over-the-counter pain relievers are usually effective. How often do you get these headaches, and do you have any medication allergies I should know about?"
        
        elif "no" in user_lower and ("allerg" in context or "aspirin" in context):
            return "Great! Since you don't have any allergies, we have more options available. Let me search for the most appropriate medications for your type of headache. I'll look for both fast-acting and longer-lasting options."
        
        elif "yes" in user_lower and ("allerg" in context or "medication" in context):
            return "It's important to know about your allergies for safety. Can you tell me which medications you're allergic to? Common ones include aspirin, ibuprofen, acetaminophen, or any antibiotics?"
        
        elif "aspirin" in user_lower and "allerg" in user_lower:
            return "Thank you for letting me know about your aspirin allergy. I'll avoid recommending aspirin and will focus on other pain relievers like acetaminophen or non-aspirin NSAIDs, unless you're allergic to those as well."
        
        elif "ibuprofen" in user_lower and "allerg" in user_lower:
            return "I'll note your ibuprofen allergy. We can look at acetaminophen-based pain relievers or other alternatives. Are you also allergic to other NSAIDs like naproxen?"
        
        # Handle general pain/symptom responses
        elif any(word in user_lower for word in ["pain", "hurt", "ache"]):
            return "I understand you're experiencing pain. Based on what you've told me, let me search for appropriate pain relief options that would be safe and effective for your situation."
        
        elif any(word in user_lower for word in ["nausea", "sick", "stomach"]):
            return "Nausea can make headaches worse. I'll look for medications that can help with both the pain and nausea, or at least won't upset your stomach further."
        
        elif any(word in user_lower for word in ["often", "frequent", "always"]):
            return "Frequent headaches might benefit from different treatment approaches. I'll look for both immediate relief options and potentially preventive medications you could discuss with your doctor."
        
        else:
            return "Thank you for providing that information. It helps me understand your situation better. Let me search for the most appropriate medications based on everything you've told me."
    
    def start_conversation(self, initial_message: str):
        """Start a conversation with an initial message"""
        # Activate conversation mode
        st.session_state.conversation_active = True
        st.session_state.conversation_history = []
        
        # Welcome message first
        welcome_msg = "Hello! I'm your AI medication assistant. I see you want to discuss your symptoms."
        st.session_state.conversation_history.append({"role": "assistant", "content": welcome_msg})
        
        # Add user message
        st.session_state.conversation_history.append({"role": "user", "content": initial_message})
        
        # Get AI response using contextual method
        ai_response = self.get_contextual_ai_response(initial_message, st.session_state.conversation_history)
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        st.rerun()
    
    def get_ai_response(self, user_message: str) -> str:
        """Get AI response (enhanced conversational logic)"""
        user_lower = user_message.lower()
        
        # Enhanced conversational responses
        if any(word in user_lower for word in ["headache", "head pain", "migraine"]):
            return "I understand you're experiencing head pain. To help you find the best medication, I'd like to know: Is this a sharp, throbbing pain (possibly migraine) or a dull, constant ache (tension headache)? Also, do you have any allergies to common pain relievers like aspirin, ibuprofen, or acetaminophen?"
            
        elif any(word in user_lower for word in ["blood pressure", "hypertension", "bp"]):
            return "I can help you find blood pressure medications. Are you looking for information about starting blood pressure treatment, or do you need alternatives to current medications? It's important to know if you have any heart conditions or drug allergies, as this affects which medications are safe for you."
            
        elif any(word in user_lower for word in ["nausea", "nauseous", "sick", "vomit"]):
            return "Nausea can be very uncomfortable. To recommend the right medication, can you tell me: Is this related to motion sickness, morning sickness, food poisoning, or something else? Are you currently taking any other medications that might be causing this?"
            
        elif any(word in user_lower for word in ["pain", "hurt", "ache", "sore"]):
            return "I can help you find pain relief options. Can you describe where the pain is located and what type of pain it is? Also, do you have any allergies to pain medications like NSAIDs (ibuprofen, naproxen) or other pain relievers?"
            
        elif any(word in user_lower for word in ["cold", "flu", "cough", "congestion"]):
            return "Cold and flu symptoms can be treated with various medications. Are you primarily dealing with congestion, cough, fever, or body aches? Do you have any health conditions or take medications that might interact with cold remedies?"
            
        elif any(word in user_lower for word in ["allergy", "allergic", "sneezing", "runny nose"]):
            return "Allergy symptoms can be managed effectively. Are you experiencing seasonal allergies, food allergies, or reactions to specific triggers? Do you prefer non-drowsy options, or is sedation okay for nighttime use?"
        
        elif any(word in user_lower for word in ["sharp", "throbbing", "migraine"]):
            return "It sounds like you might be experiencing a migraine. Migraines often require specific treatments. Have you been diagnosed with migraines before? Do you have any triggers you've noticed, and are you allergic to any medications?"
            
        elif any(word in user_lower for word in ["dull", "constant", "tension"]):
            return "That sounds like a tension headache. These are often caused by stress, muscle tension, or dehydration. Over-the-counter pain relievers can be effective. Do you have any allergies to medications like acetaminophen, ibuprofen, or aspirin?"
            
        elif any(word in user_lower for word in ["aspirin", "allergy", "allergic"]):
            return "Thank you for letting me know about your aspirin allergy. That's important information! I'll look for aspirin-free alternatives. Are you allergic to any other medications or NSAIDs like ibuprofen?"
            
        else:
            return "I'm here to help you find the right medications. Can you tell me more about your symptoms or what type of medication you're looking for? Also, please let me know about any drug allergies or current medications, as this helps me recommend safe options."
    
    def handle_text_message(self, message: str):
        """Handle text message input"""
        if not hasattr(st.session_state, 'conversation_history'):
            st.session_state.conversation_history = []
        
        # Add user message
        st.session_state.conversation_history.append({"role": "user", "content": message})
        
        # Get AI response
        ai_response = self.get_ai_response(message)
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Search for medications if the message seems like a request
        if any(word in message.lower() for word in ["need", "find", "looking for", "help", "pain", "headache", "medication", "medicine", "drug"]):
            results = self.search_medications_wrapper(message)
            if results:
                med_count = len(results)
                follow_up = f"I found {med_count} medications that might help. Here are the most relevant ones:"
                st.session_state.conversation_history.append({"role": "assistant", "content": follow_up})
                
                # Display results
                self.display_conversation_results(results[:3])
        
        st.rerun()
    
    def display_conversation_results(self, results: List[Dict[str, Any]]):
        """Display medication results in a conversational format"""
        st.markdown("### 💊 Recommended Medications")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"Option {i}: {result['drug']} - {result['gpt4_form']}", expanded=i==1):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Drug Name:** {result['drug']}")
                    st.write(f"**Form:** {result['gpt4_form']}")
                    st.write(f"**Match Score:** {result.get('similarity_score', 0):.3f}")
                
                with col2:
                    if st.button(f"🔊 Read about {result['drug']}", key=f"tts_conv_{i}"):
                        description = f"Option {i}: {result['drug']}, available as {result['gpt4_form']}. This medication has a similarity score of {result.get('similarity_score', 0):.2f} for your needs."
                        self.app.text_to_speech(description)
                
                # Add to conversation history
                med_info = f"**{result['drug']}** ({result['gpt4_form']}) - Match score: {result.get('similarity_score', 0):.3f}"
                if not any(med_info in msg.get('content', '') for msg in st.session_state.conversation_history):
                    st.session_state.conversation_history.append({"role": "assistant", "content": f"Recommendation: {med_info}"})
    
    def handle_audio_message(self, audio_file):
        """Handle audio message input"""
        with st.spinner("🎙️ Processing your audio message..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_path = tmp_file.name
            
            try:
                # Convert to text using existing functionality
                transcript = self.app.speech_to_text(temp_path)
                if transcript:
                    st.success(f"🎯 You said: {transcript}")
                    # Handle as text message
                    self.handle_text_message(transcript)
                else:
                    st.error("Could not transcribe audio. Please try again.")
            except Exception as e:
                st.error(f"Audio processing error: {e}")
            finally:
                os.unlink(temp_path)
    
    def start_live_conversation(self):
        """Start a live conversation session"""
        st.markdown("### 🔴 **LIVE CONVERSATION MODE**")
        st.info("🎙️ **NEW**: Real-time voice conversation with AI medication assistant!")
        
        # Initialize conversation state
        if 'live_conversation_active' not in st.session_state:
            st.session_state.live_conversation_active = True
            st.session_state.conversation_turn = 0
        
        # Live conversation interface
        conversation_container = st.container()
        
        with conversation_container:
            st.markdown("**🤖 AI:** Hello! I'm your AI medication assistant. What symptoms or medication questions do you have today?")
            
            # Auto-play welcome message
            if st.session_state.conversation_turn == 0:
                welcome_msg = "Hello! I'm your AI medication assistant. What symptoms or medication questions do you have today?"
                self.app.text_to_speech(welcome_msg)
                st.session_state.conversation_turn += 1
            
            # Voice input section
            col1, col2 = st.columns(2)
            
            with col1:
                # Simulated live microphone (using file upload for now)
                st.markdown("**🎙️ Speak to the AI:**")
                live_audio = st.file_uploader("Record your response:", type=['wav', 'mp3', 'm4a'], key=f"live_audio_{st.session_state.conversation_turn}")
                
                if live_audio:
                    self.handle_live_audio_response(live_audio)
            
            with col2:
                # Quick response buttons
                st.markdown("**⚡ Quick Responses:**")
                if st.button("💊 I need pain relief", key="quick_pain"):
                    self.handle_live_text_response("I need pain relief medication")
                
                if st.button("🤧 I have allergies", key="quick_allergy"):
                    self.handle_live_text_response("I have allergy symptoms")
                
                if st.button("😴 I can't sleep", key="quick_sleep"):
                    self.handle_live_text_response("I'm having trouble sleeping")
            
            # End conversation
            if st.button("🛑 End Conversation", key="end_live"):
                st.session_state.live_conversation_active = False
                st.success("✅ Conversation ended. Thank you for using MARMAR PillSight!")
                st.rerun()
    
    def handle_live_audio_response(self, audio_file):
        """Handle live audio response in conversation"""
        with st.spinner("🎙️ Processing your voice..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.read())
                temp_path = tmp_file.name
            
            try:
                # Convert to text
                transcript = self.app.speech_to_text(temp_path)
                if transcript:
                    st.success(f"🎯 You said: {transcript}")
                    self.handle_live_text_response(transcript)
                else:
                    st.error("Could not understand. Please try again.")
            except Exception as e:
                st.error(f"Audio processing error: {e}")
            finally:
                os.unlink(temp_path)
    
    def handle_live_text_response(self, user_message: str):
        """Handle live text response in conversation"""
        # Add to conversation history
        if not hasattr(st.session_state, 'live_conversation_history'):
            st.session_state.live_conversation_history = []
        
        st.session_state.live_conversation_history.append({"role": "user", "content": user_message})
        
        # Get AI response
        ai_response = self.get_enhanced_ai_response(user_message)
        st.session_state.live_conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Display AI response
        st.markdown(f"**🤖 AI:** {ai_response}")
        
        # Speak the response
        self.app.text_to_speech(ai_response)
        
        # Search for medications and integrate into conversation
        results = self.search_medications_wrapper(user_message)
        if results:
            medication_response = self.create_medication_response(results[:3])
            st.session_state.live_conversation_history.append({"role": "assistant", "content": medication_response})
            
            # Display medication results
            st.markdown(f"**🤖 AI:** {medication_response}")
            self.app.text_to_speech(medication_response)
            
            # Show medication cards
            self.display_live_medication_results(results[:3])
        
        # Increment conversation turn
        st.session_state.conversation_turn += 1
        st.rerun()
    
    def get_enhanced_ai_response(self, user_message: str) -> str:
        """Get enhanced AI response for live conversation"""
        user_lower = user_message.lower()
        
        # More natural, conversational responses
        if any(word in user_lower for word in ["headache", "head pain", "migraine"]):
            return "I understand you're dealing with head pain. That can be really uncomfortable. Can you tell me - is it a sharp, throbbing pain like a migraine, or more of a dull, constant ache? Also, are you allergic to any pain medications?"
            
        elif any(word in user_lower for word in ["pain relief", "pain", "hurt", "ache"]):
            return "I'm sorry you're in pain. Let me help you find the right relief. Where is the pain located, and how would you describe it - sharp, dull, throbbing? Any allergies to pain medications I should know about?"
            
        elif any(word in user_lower for word in ["allergy", "allergic", "sneezing", "runny nose"]):
            return "Allergies can be really bothersome! Are you dealing with seasonal allergies, or is this a reaction to something specific? Do you prefer non-drowsy options, or would a medication that might make you sleepy be okay?"
            
        elif any(word in user_lower for word in ["sleep", "insomnia", "can't sleep"]):
            return "Sleep troubles can affect everything. How long have you been having difficulty sleeping? Are you looking for something to help you fall asleep, or do you wake up during the night? Any medications you're currently taking?"
            
        elif any(word in user_lower for word in ["nausea", "nauseous", "sick", "stomach"]):
            return "Nausea is never pleasant. Is this related to motion sickness, something you ate, or perhaps medication side effects? Are you able to keep fluids down?"
            
        else:
            return "I'm here to help you find the right medication for your needs. Can you tell me more about what you're experiencing? Also, please let me know about any drug allergies - that's really important for your safety."
    
    def create_medication_response(self, results: List[Dict[str, Any]]) -> str:
        """Create a natural response about found medications"""
        if not results:
            return "I didn't find any specific medications for your symptoms, but let me ask you some more questions to help narrow it down."
        
        med_names = [result['drug'] for result in results[:3]]
        
        if len(results) == 1:
            return f"I found {results[0]['drug']} which comes as {results[0]['gpt4_form']}. This could be a good option for your symptoms."
        else:
            return f"I found several options that might help: {', '.join(med_names[:-1])}, and {med_names[-1]}. Let me show you the details for each one."
    
    def display_live_medication_results(self, results: List[Dict[str, Any]]):
        """Display medication results in live conversation format"""
        st.markdown("### 💊 **Medication Recommendations**")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"💊 {result['drug']} ({result['gpt4_form']})", expanded=i==1):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**💊 Drug:** {result['drug']}")
                    st.write(f"**📋 Form:** {result['gpt4_form']}")
                    st.write(f"**🎯 Match:** {result.get('similarity_score', 0):.1%}")
                    if 'description' in result:
                        st.write(f"**ℹ️ Info:** {result['description']}")
                
                with col2:
                    if st.button(f"🔊 Tell me about {result['drug']}", key=f"speak_med_{i}"):
                        med_description = f"{result['drug']} is available as {result['gpt4_form']}. {result.get('description', 'This medication could help with your symptoms.')}"
                        self.app.text_to_speech(med_description)
                    
                    if st.button(f"❓ Ask about {result['drug']}", key=f"ask_med_{i}"):
                        question = f"Tell me more about {result['drug']} and how it works"
                        self.handle_live_text_response(question) 