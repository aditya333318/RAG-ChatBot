"""
Streamlit web application for the RAG Chatbot.
Provides a user-friendly interface for uploading PDFs and chatting with them.
"""

import os
import sys

# Ensure src is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.chatbot import RAGChatbot
from src.config import Config

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS STYLING
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .source-header {
        font-weight: 600;
        color: #1f77b4;
        font-size: 0.9rem;
    }
    .source-content {
        color: #555;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 12px 12px 2px 12px;
        margin: 8px 0;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 2px;
        margin: 8px 0;
    }
    .stSpinner > div {
        border-top-color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────

if "chatbot" not in st.session_state:
    try:
        st.session_state.chatbot = RAGChatbot()
        st.session_state.ready = False
    except ValueError as e:
        st.error(f"⚠️ {e}")
        st.info("Please create a `.env` file with your `OPENAI_API_KEY` and restart the app.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources_visible" not in st.session_state:
    st.session_state.sources_visible = {}

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to chat with",
    )

    if uploaded_files:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")

                # Save to temp file
                temp_path = os.path.join(Config.SAMPLE_PDFS_PATH, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Ingest
                try:
                    result = st.session_state.chatbot.ingest_pdf(temp_path)
                    progress_bar.progress((i + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                    continue

            status_text.text("✅ All documents processed!")
            st.session_state.ready = True
            st.success(f"Indexed {len(uploaded_files)} document(s). Ready to chat!")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    model = st.selectbox(
        "LLM Model",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        index=0,
    )

    top_k = st.slider("Retrieved Chunks (Top-K)", 1, 10, 5)

    st.markdown("---")
    st.markdown("### 💾 Index")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", use_container_width=True):
            try:
                st.session_state.chatbot.save_index()
                st.success("Index saved!")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with col2:
        if st.button("📂 Load", use_container_width=True):
            try:
                st.session_state.chatbot.load_index()
                st.session_state.ready = True
                st.success("Index loaded!")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chatbot.clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("• LangChain + OpenAI + FAISS")
    st.markdown("• Streamlit")

# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────

st.markdown('<div class="main-header"> RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Upload PDFs and chat with them. Ask questions, get answers with source citations.'
    '</div>',
    unsafe_allow_html=True
)

# Status indicator
if st.session_state.ready:
    st.success("✅ Chatbot is ready! Ask your questions below.")
else:
    st.info("⬅️ Upload PDFs from the sidebar to get started.")

# ──────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ──────────────────────────────────────────────────────────────

st.markdown("---")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.container():
            st.markdown(
                f'<div class="chat-message-user"><b>You:</b><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
    else:
        with st.container():
            st.markdown(
                f'<div class="chat-message-assistant"><b>Assistant:</b><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )

            # Show sources if available
            if "sources" in message:
                key = f"source_{i}"
                if key not in st.session_state.sources_visible:
                    st.session_state.sources_visible[key] = False

                if st.toggle(
                    f"📚 Show Sources ({len(message['sources'])})",
                    key=key,
                    value=st.session_state.sources_visible[key],
                ):
                    for src in message["sources"]:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<div class="source-header">📄 {src["source"]} — Page {src["page"]}</div>'
                            f'<div class="source-content">{src["content"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

# ──────────────────────────────────────────────────────────────
# INPUT
# ──────────────────────────────────────────────────────────────

st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="e.g., What are the key findings in the report?",
            label_visibility="collapsed",
        )
    with col2:
        submit = st.form_submit_button("Send", use_container_width=True, type="primary")

if submit and user_input.strip():
    if not st.session_state.ready:
        st.warning("Please upload and process PDFs first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chatbot.ask(user_input)

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })
            except Exception as e:
                st.error(f"Error: {e}")

        st.rerun()
