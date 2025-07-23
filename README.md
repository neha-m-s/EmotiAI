# EmotiAI: Emotion-Aware Mental Health Chatbot 

EmotiAI is an AI-powered mental health chatbot designed to engage users in empathetic conversations using Cohere’s Command LLM and real-time emotion detection. Built with Streamlit, LangChain, and Hugging Face Transformers, EmotiAI offers emotionally intelligent responses for well-being support.

## Features

-  **LLM-Powered Conversations** using Cohere’s `command-xlarge` via LangChain
-  **Emotion Detection** via Hugging Face facial expression classification
-  **Intent-Aware Responses** using a custom JSON-based knowledge base
-  **Interactive UI** built with Streamlit
-  **Lightweight and Fully Local** (no backend required)

---

##  Tech Stack

- **Frontend**: Streamlit
- **LLM**: Cohere Command (via LangChain)
- **Emotion Detection**: Hugging Face + OpenCV
- **Other Libraries**: Transformers, Torch, Sentence Transformers, Pillow, NumPy

---

##  Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/emotiAI.git
cd emotiAI

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

#Run the app
streamlit run app.py
