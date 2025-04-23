import streamlit as st
import fitz  # PyMuPDF
import requests
import re
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="RAG PDF QA with Topics", layout="wide")
st.title("📄🔍 ChatBot for Pakistan Constitution")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract headings and map them to content
def extract_topics(text):
    lines = text.split("\n")
    topics = {}
    current_topic = "Introduction"
    topics[current_topic] = ""

    for line in lines:
        # Match headings (very simple heuristic: all caps or numbered)
        if re.match(r"^[A-Z][A-Z\s\d\-\.]{3,}$", line.strip()) or re.match(r"^\d+(\.\d+)*\s.+", line.strip()):
            current_topic = line.strip()
            topics[current_topic] = ""
        else:
            topics[current_topic] += line.strip() + " "

    return topics

def query_groq_llm(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that only answers using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Groq API Error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded successfully!")

    topics_dict = extract_topics(text)
    topic_list = list(topics_dict.keys())

    selected_topic = st.selectbox("📚 Select a topic", topic_list)

    if selected_topic:
        topic_text = topics_dict[selected_topic]
        st.subheader(f"📄 Content under: {selected_topic}")
        st.write(topic_text[:2000] + "..." if len(topic_text) > 2000 else topic_text)

        user_query = st.text_input("💬 Ask a question about this topic:")
        if user_query:
            with st.spinner("Thinking..."):
                chunks = [topic_text]
                chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
                query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
                similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
                top_chunks = [chunks[i] for i in torch.topk(similarities, k=1).indices]

                context = "\n\n".join(top_chunks)
                prompt = f"Context:\n{context}\n\nQuestion: {user_query}"

                try:
                    answer = query_groq_llm(prompt, GROQ_API_KEY)
                    st.markdown("### 💡 Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
