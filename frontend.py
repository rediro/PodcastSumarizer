import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("RAG Model Web Interface")

# Process YouTube Video
st.header("Transcribe YouTube Video")
youtube_url = st.text_input("Enter YouTube URL:")
if st.button("Transcribe"):
    if youtube_url:
        response = requests.post(f"{BACKEND_URL}/process_youtube", data={"video_url": youtube_url})
        st.write("**Transcription:**", response.json().get("transcript", "Error processing video"))

# Upload Document
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=["txt"])
if st.button("Process Document"):
    if uploaded_file:
        files = {"file": uploaded_file}
        response = requests.post(f"{BACKEND_URL}/process_document", files=files)
        st.write(response.json().get("message", "Error processing document"))

# Query RAG Model
st.header("Ask a Question")
query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if query:
        response = requests.post(f"{BACKEND_URL}/query_rag", data={"query": query})
        st.write("**Answer:**", response.json().get("answer", "No response"))
