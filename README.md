# RAG EDF Insight

Retrieval-Augmented Generation system built on EDF annual reports (2023, 2024, 2025). Ask financial questions in French and get sourced answers from official EDF documents.

## Stack

- PDF extraction: PyMuPDF (fitz)
- Chunking: LangChain RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
- Embeddings: Sentence-Transformers (paraphrase-multilingual-mpnet-base-v2)
- Vector store: ChromaDB (HNSW, cosine distance, 22 661 vectors)
- LLM: Ollama with qwen2.5:7b (primary) and llama3 (fallback)
- API: FastAPI with /health and /ask endpoints
- Cloud storage: Azure Blob Storage
- Containerization: Docker

## Project structure

    rag-edf-insight/
    ├── src/
    │   ├── extract.py    # PDF text extraction from Azure Blob
    │   ├── index.py      # Chunking + embedding + ChromaDB indexing
    │   ├── rag.py        # RAG pipeline (retrieve + generate)
    │   └── api.py        # FastAPI REST API
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── .gitignore
    └── .env              # Azure credentials (not in repo)

## Run locally

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ollama serve &
    ollama pull qwen2.5:7b
    python src/extract.py
    python src/index.py
    uvicorn src.api:app --host 0.0.0.0 --port 8000

## Run with Docker

    docker build -t rag-edf-insight .
    docker run --network host \
      -v /path/to/vectorstore:/app/vectorstore \
      --env-file .env \
      -e CHROMA_PATH=/app/vectorstore \
      rag-edf-insight

## API usage

Health check:

    curl http://localhost:8000/health

Ask a question:

    curl -X POST http://localhost:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "Quel est le chiffre d affaires d EDF en 2024 ?", "year": "2024"}'

## Author

Hamza Zaroual — Data Science / AI Engineer
