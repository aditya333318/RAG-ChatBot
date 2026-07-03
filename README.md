# RAG Chatbot — Chat with Your PDFs

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **FAISS**, and **OpenAI**. Upload PDFs (even 500+ pages) and ask questions in natural language. The bot retrieves the most relevant chunks and generates accurate, context-aware answers.

---

## Features

| Feature | Description |
|---------|-------------|
| PDF Ingestion | Load single or multiple PDFs with robust error handling |
| Smart Chunking | Recursive + semantic chunking with configurable overlap |
| Vector Search | FAISS for lightning-fast similarity search |
| OpenAI Integration | GPT-4o / GPT-3.5-turbo with custom prompt engineering |
| Streaming Responses | Real-time token streaming for better UX |
| Chat Memory | Conversation history across multiple turns |
| Source Citations | Every answer cites the source document and page |
| Web UI | Beautiful Streamlit interface with drag-and-drop upload |

---

## Tech Stack

- **LangChain** — Orchestration framework for LLM + retrieval pipelines
- **FAISS** — Facebook AI Similarity Search (in-memory vector DB)
- **OpenAI** — Embeddings (`text-embedding-3-small`) + Chat models
- **PyPDF / pdfplumber** — Robust PDF text extraction
- **Streamlit** — Interactive web application
- **Python 3.10+**

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <repo-url>
cd rag-chatbot
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
rag-chatbot/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
├── app.py                    # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration & constants
│   ├── document_loader.py    # PDF loading & parsing
│   ├── chunking.py           # Text chunking strategies
│   ├── embeddings.py         # Embedding model wrapper
│   ├── vector_store.py       # FAISS index management
│   ├── prompt_templates.py   # Prompt engineering
│   ├── rag_chain.py          # Core RAG pipeline
│   └── chatbot.py            # Conversational agent
├── tests/
│   ├── test_loader.py
│   ├── test_chunking.py
│   └── test_rag_chain.py
├── notebooks/
│   └── 01_rag_baseline.ipynb # Research / experimentation
└── sample_pdfs/
    └── .gitkeep
```

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF File  │────▶│ PDF Loader   │────▶│ Raw Text    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
              ┌─────────────────┐
              │ Text Chunking   │  ◄── RecursiveCharacterTextSplitter
              │  (Overlap)      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  OpenAI Embed   │  ◄── text-embedding-3-small
              │                 │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  FAISS Index    │  ◄── In-memory vector store
              │                 │
              └────────┬────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐        ┌─────────────────┐
│ User Question   │        │ Similarity      │
│                 │───────▶│ Search (Top-k)  │
└─────────────────┘        └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Retrieved       │
                           │ Chunks          │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Prompt Engineer │
                           │ (System + Context)
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  OpenAI LLM     │
                           │ (GPT-4o/3.5)    │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Answer + Sources│
                           └─────────────────┘
```

---

## Chunking Strategy

This project uses **recursive character text splitting** with sensible defaults optimized for technical documents:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk Size | 1000 tokens | Fits within embedding context + 3-4 paragraphs |
| Chunk Overlap | 200 tokens | Preserves sentence boundaries across chunks |
| Separators | `["\n\n", "\n", ". ", " ", ""]` | Prioritizes paragraph → sentence → word boundaries |

For 500-page PDFs (~125K words), this produces ~180 chunks — easily searchable in FAISS in <50ms.

---

## Prompt Engineering

The system uses a carefully crafted prompt template with:

1. **System instructions** — Define the assistant's role and constraints
2. **Context injection** — Retrieved chunks inserted with source metadata
3. **Few-shot examples** — Guide answer format (bulleted, concise, cited)
4. **Safety guardrails** — "If the answer is not in the context, say 'I don't know'"

See `src/prompt_templates.py` for full templates.

---

## Example Usage

### Web Interface
```bash
streamlit run app.py
```
Drag-and-drop PDFs, then ask questions like:
- "What are the key findings in section 3?"
- "Summarize the conclusion."
- "Compare the approaches on pages 45 and 78."

### Programmatic API
```python
from src.chatbot import RAGChatbot

bot = RAGChatbot()
bot.ingest_pdf("research_paper.pdf")

response = bot.ask("What is the main contribution of this paper?")
print(response["answer"])
print(response["sources"])  # [{'page': 3, 'source': 'research_paper.pdf', 'content': '...'}]
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Your OpenAI API key |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model name |
| `LLM_MODEL` | No | `gpt-3.5-turbo` | Chat model name |
| `CHUNK_SIZE` | No | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | No | `200` | Chunk overlap |
| `TOP_K` | No | `5` | Number of chunks to retrieve |
| `TEMPERATURE` | No | `0.0` | LLM temperature |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Future Enhancements

- [ ] Hybrid search (BM25 + vector) for better keyword matching
- [ ] Re-ranking with cross-encoders (e.g., Cohere Rerank)
- [ ] Persistent vector DB (Pinecone / Weaviate / Chroma)
- [ ] Multi-modal support (images, tables within PDFs)
- [ ] Evaluation framework (RAGAS, LLM-as-a-judge)
- [ ] Docker containerization
- [ ] REST API with FastAPI

---

## License

MIT

---

## Author

Built for resume portfolio — demonstrates production RAG system design with LangChain, FAISS, and OpenAI.
