import streamlit as st
import json
import time
import os
from langchain.llms import Cohere
import datetime
from transformers import pipeline

# Load Hugging Face emotion model
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

emotion_classifier = load_emotion_model()

def detect_emotion(text):
    result = emotion_classifier(text)
    return result[0]["label"] if result else "neutral"

# Set API Key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "1e7a5e10482ad3bce271180e403c1b4e9a785a00ec66c9821621d036d354ae72")

@st.cache_data
def load_mental_health_data():
    with open(r"C:/Users/NEHA M S/Downloads/Mental_Health_Chatbot-main/ai.json", "r") as file:
        return json.load(file)

SYSTEM_PROMPT = """
You are a kind, compassionate, and supportive mental health assistant.  
Your goal is to **uplift, encourage, and provide clear, practical advice** to users in distress.

**How to Respond:**
- **Start every response with a strong, reassuring sentence in CAPITALS and bold.**  
- Focus on **empowering solutions** rather than just acknowledging distress.  
- Use a **warm, hopeful tone**, reminding them that **things can improve and they are capable**.  
- Offer **small, achievable steps** for self-care, deep breathing, and positive self-talk.  
- If a user feels **overwhelmed, remind them of their inner strength**.  
"""

models = {
    "Cohere Command": Cohere(model="command-xlarge", cohere_api_key="sWmE1lyhhw4XomK8LVSW58LlX0fe4ke89B1fxFvz")
}

def get_response(model_name, user_query, dataset, emotion):
    for keyword, advice in dataset.items():
        if keyword in user_query.lower():
            user_query += f"\n[Additional Context: {advice}]"

    # Include detected emotion in the system prompt
    user_query = f"[Detected Emotion: {emotion}]\n" + user_query

    modified_query = SYSTEM_PROMPT + "\nUser: " + user_query + "\nAI:"

    try:
        response = models[model_name].invoke(modified_query, max_tokens=1024)
        response = response.strip()

        retry_attempts = 3
        while not response.endswith(".") and retry_attempts > 0:
            additional_response = models[model_name].invoke(modified_query, max_tokens=512)
            response += " " + additional_response.strip()
            retry_attempts -= 1

        return response if isinstance(response, str) else str(response)

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="EmotiCare AI", layout="wide")

st.markdown("""
    <style>
        body { background-color: #121212; color: #ffffff; }
        .chat-container {
            max-width: 600px; margin: auto; background: rgba(255, 255, 255, 0.1);
            padding: 20px; border-radius: 15px; box-shadow: 0px 4px 15px rgba(0, 255, 200, 0.3);
            backdrop-filter: blur(8px);
        }
        .chat-message { padding: 12px; border-radius: 8px; margin: 10px 0; font-size: 16px;
            animation: fadeIn 0.5s forwards; display: inline-block; }
        .user-message { background: linear-gradient(135deg, #007aff, #00c6ff); color: white;
            text-align: right; border-radius: 15px 15px 0 15px; padding: 10px 15px; }
        .ai-message { background: linear-gradient(135deg, #00e676, #1de9b6); color: black;
            text-align: left; border-radius: 15px 15px 15px 0; padding: 10px 15px; }
        .typing-indicator { font-size: 14px; color: #00e676; display: flex; align-items: center; }
        .dot { width: 6px; height: 6px; margin: 0 3px; background: #00e676;
            border-radius: 50%; animation: blink 1.5s infinite; }
        .dot:nth-child(2) { animation-delay: 0.3s; }
        .dot:nth-child(3) { animation-delay: 0.6s; }
        @keyframes blink { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
        .send-button { background: linear-gradient(135deg, #007aff, #00c6ff); color: white;
            border: none; padding: 10px 20px; border-radius: 25px; font-size: 16px; transition: 0.3s; }
        .send-button:hover { background: linear-gradient(135deg, #00c6ff, #007aff); transform: scale(1.05); }
        .send-button:active { transform: scale(0.95); }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #00e676;'>💬 EmotiCare AI </h1>", unsafe_allow_html=True)
st.write("🌿 **A calming AI companion for your mental well-being.**")

if "messages" not in st.session_state:
    st.session_state.messages = [("ai-message", "<strong>Mental Health Chatbot:</strong> Hello! I am here to support you. How are you feeling right now?", datetime.datetime.now().strftime("%H:%M:%S"))]

model_choice = st.selectbox("🛠 Select AI Model:", list(models.keys()))

chat_container = st.container()
with chat_container:
    for role, text, timestamp in st.session_state.messages:
        st.markdown(f'<div class="chat-message {role}">{text} <br><small style="color:gray;">🕒 {timestamp}</small></div>', unsafe_allow_html=True)

user_input = st.text_input("💬 Type your message here...")

col1, col2 = st.columns([4, 1])
with col1:
    send_btn = st.button("Send", key="send-btn", help="Click to send message")
with col2:
    clear_btn = st.button("🗑 Clear Chat", key="clear-btn", help="Reset the conversation")

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if send_btn:
    if user_input.strip():
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        detected_emotion = detect_emotion(user_input)

        st.session_state.messages.append((
            "user-message", 
            f"<strong>You:</strong> {user_input} <br><em>🧠 Detected Emotion: <strong>{detected_emotion}</strong></em>", 
            timestamp
        ))

        with st.spinner("Analyzing..."):
            time.sleep(1.5)
            dataset = load_mental_health_data()
            response = get_response(model_choice, user_input, dataset, detected_emotion)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append(("ai-message", f"<strong>{model_choice}:</strong> {response}", timestamp))

        st.rerun()
    else:
        st.warning("⚠️ Please type a message before sending.")
