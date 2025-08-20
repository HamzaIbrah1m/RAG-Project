# 📘 RAG App (Retrieval-Augmented Generation)

This project is a simple **RAG pipeline** built with:
- Streamlit (for the UI)
- Hugging Face Transformers (for summarization & Q/A)
- FAISS (for retrieval)
- SentenceTransformers (for embeddings)

## 🚀 Features
- Upload a **PDF** or paste **raw text**
- Automatically **summarizes** the notes
- Build a **vector index** with FAISS
- Ask **questions** about the uploaded content

## 📂 How to Run
```bash
git clone https://github.com/your-username/rag_project.git
cd rag_project
pip install -r requirements.txt
streamlit run app.py
