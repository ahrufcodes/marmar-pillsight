# 🏆 MARMAR PillSight - Hackathon Setup Guide

## 📋 **Hackathon Compliance Checklist**

### ✅ **MongoDB Track Requirements**
- [x] **Public Dataset**: Using `drugformdb` open-source pharmaceutical dataset
- [x] **MongoDB Vector Search**: Advanced similarity matching for drug identification  
- [x] **Google Cloud Integration**: Speech-to-Text + Text-to-Speech AI services
- [x] **AI Analysis**: Voice queries → embeddings → vector search → results

### 🎯 **Key Features for Judges**
- **Voice-First Interface**: Natural conversation with AI
- **Medical AI Model**: Specialized for pharmaceutical terms
- **Real-time Processing**: Google Cloud AI services
- **Healthcare Impact**: Medication safety & accessibility

---

## 🚀 **Quick Start (5 Minutes)**

### 1. **Google Cloud Setup**
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login and create project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable speech.googleapis.com
gcloud services enable texttospeech.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Create service account and key
gcloud iam service-accounts create pillsight-hackathon
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pillsight-hackathon@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/owner"

gcloud iam service-accounts keys create key.json \
    --iam-account=pillsight-hackathon@YOUR_PROJECT_ID.iam.gserviceaccount.com

export GOOGLE_APPLICATION_CREDENTIALS="./key.json"
```

### 2. **Run the App**
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### 3. **Test Voice Features**
1. Upload an audio file saying: *"I need something for headaches"*
2. AI will transcribe using Google Cloud Speech-to-Text
3. MongoDB vector search finds similar medications
4. Results read aloud using Google Cloud Text-to-Speech

---

## 🏅 **Competitive Advantages**

### **Technical Innovation**
- **Google Cloud AI**: Latest speech recognition models
- **MongoDB Vector Search**: Advanced semantic similarity
- **Medical Specialization**: Trained for pharmaceutical terminology
- **Cloud-Native**: Fully scalable architecture

### **Real-World Impact**
- **Accessibility**: Voice interface for visually impaired
- **Safety**: Prevents medication errors
- **Global Health**: Works with international drug databases
- **Emergency Use**: Quick medication identification

### **Hackathon Fit**
- **Google Cloud**: Prominent use of multiple GCP services
- **MongoDB**: Advanced vector search capabilities  
- **Public Data**: Open pharmaceutical dataset
- **AI Innovation**: Voice + vector search combination

---

## 📊 **Architecture Overview**

```
🗣️ Voice Input → 🤖 Google Cloud Speech → 🔍 MongoDB Vector Search → 📢 Google Cloud TTS
```

### **Google Cloud Services Used:**
- **Speech-to-Text**: Medical conversation model
- **Text-to-Speech**: Neural voices for responses  
- **Cloud Storage**: Audio file processing
- **AI Platform**: Model hosting & inference

### **MongoDB Features:**
- **Vector Search**: Semantic similarity matching
- **Atlas**: Cloud-native database
- **Aggregation Pipeline**: Complex query processing
- **Full-text Search**: Combined text + vector search

---

## 🎬 **Demo Script (3-minute video)**

### **Minute 1: Problem & Solution**
- "Medication errors affect millions worldwide"
- "Voice-enabled AI can help identify pills safely"
- "Using Google Cloud + MongoDB for real-time processing"

### **Minute 2: Technical Demo**
- Upload audio: *"Show me blood pressure medications"*
- Watch Google Cloud Speech-to-Text transcription
- See MongoDB vector search results
- Hear Google Cloud Text-to-Speech response

### **Minute 3: Impact & Innovation**
- Healthcare accessibility improvements
- Technical architecture explanation
- Scalability and global deployment potential

---

## 🎯 **Judging Criteria Alignment**

### **Technological Implementation (25%)**
- Google Cloud Speech APIs with medical models
- MongoDB vector search with custom embeddings
- Streamlit real-time web interface
- Cloud-native, production-ready architecture

### **Design (25%)**
- Intuitive voice-first interface
- Accessible for users with disabilities  
- Clean, medical-professional UI
- Mobile-responsive design

### **Potential Impact (25%)**  
- Global medication safety improvement
- Healthcare accessibility for visually impaired
- Reduced medication errors in hospitals
- Open-source for worldwide deployment

### **Quality of Idea (25%)**
- Novel combination of voice AI + drug identification
- Addresses real healthcare challenges
- Leverages cutting-edge AI technologies
- Scalable business model

---

## 🔧 **Environment Variables**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="./key.json"
export ATLAS_URI="mongodb+srv://..."
export DB_NAME="marmar-pillsight"
export COLLECTION_NAME="drug_forms"
```

---

## 📝 **Submission Checklist**
- [ ] GitHub repository with clean code
- [ ] Demo video (3 minutes max) on YouTube/Vimeo
- [ ] Live hosted app URL (Streamlit Cloud/GCP)
- [ ] Google Cloud services documentation
- [ ] MongoDB integration writeup
- [ ] Devpost submission form completed

**Ready to win! 🏆** 