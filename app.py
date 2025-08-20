import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load QA model
qa_model = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# Create FAISS index
dimension = 384  # embedding dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
documents = []

# Sidebar
st.sidebar.title("📘 RAG App")
st.sidebar.markdown("Upload notes or paste raw text, then ask questions!")

# Tabs for PDF or Raw Text
tab1, tab2 = st.tabs(["📄 Upload PDF", "📝 Raw Text"])

raw_text = ""

with tab1:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in pdf_reader.pages:
            raw_text += page.extract_text() + "\n"

with tab2:
    raw_text = st.text_area("Paste your notes here", height=300)

if raw_text:
    # Show summary
    summary = summarizer(raw_text, max_length=150, min_length=50, do_sample=False)
    st.subheader("📝 Summary of Notes")
    st.write(summary[0]['summary_text'])

    # Build index ONLY from raw text
    sentences = raw_text.split(". ")
    embeddings = embedder.encode(sentences)
    index.add(np.array(embeddings, dtype="float32"))
    documents.extend(sentences)

    # Question answering
    st.subheader("🔎 Ask a Question")
    query = st.text_input("Type your question here...")
    if query:
        query_emb = embedder.encode([query])
        D, I = index.search(np.array(query_emb, dtype="float32"), k=3)
        retrieved = [documents[i] for i in I[0]]

        st.write("**🔎 Retrieved Context**")
        st.write(" ".join(retrieved))

        answer = qa_model(f"question: {query} context: {' '.join(retrieved)}", max_length=150)
        st.write("**💡 Answer**")
        st.write(answer[0]['generated_text'])
