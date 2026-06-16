# IKC AI Student Assistant

An AI-powered university chatbot designed to answer student inquiries using Retrieval-Augmented Generation (RAG), semantic search, and Natural Language Processing (NLP).

## Features

- AI-powered question answering
- Semantic search with vector embeddings
- FAQ management dashboard
- Unanswered questions tracking
- Arabic language support
- ChromaDB vector database
- Ollama integration

## Technologies

- Python
- Flask
- ChromaDB
- Ollama
- LangChain
- SQLite

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text
python rag_index.py
python app.py
```

## Project Structure

- `app.py` – Main application
- `rag_engine.py` – Retrieval engine
- `rag_index.py` – Vector index builder
- `admin_routes.py` – Admin dashboard routes
- `faqs.db` – Knowledge base
