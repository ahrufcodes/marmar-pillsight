# MARMAR PillSight

An AI-powered medical assistant that transforms medication discovery through natural conversations. Built for the AI in Action Hackathon with MongoDB and Google Cloud.

## Features

- Natural voice/text conversations about symptoms and medical needs
- Real-time medication recommendations using MongoDB's vector search
- Voice processing through Google Cloud Speech-to-Text
- Intelligent drug matching using SentenceTransformers
- User-friendly interface built with Streamlit

## Tech Stack

- Python
- Streamlit
- MongoDB Atlas (with Vector Search)
- Google Cloud (Speech-to-Text & Text-to-Speech)
- SentenceTransformers
- PyAudio
- streamlit-webrtc

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ahrufcodes/marmar-pillsight.git
cd marmar-pillsight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
ATLAS_URI=your_mongodb_connection_string
GOOGLE_APPLICATION_CREDENTIALS=path_to_google_cloud_credentials.json
```

4. Run the application:
```bash
streamlit run app.py
```

## License

MIT License 