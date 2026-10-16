# RAG EDF Insight

Retrieval-Augmented Generation system built on EDF annual reports (2023, 2024, 2025). Ask financial questions in French and get sourced answers from official EDF documents.

## Stack

- PDF extraction: PyMuPDF (fitz)
- Chunking: LangChain RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
- Embeddings: sentence-transformers (paraphrase-multilingual-mpnet-base-v2)
- Vector store: ChromaDB (persistent, cosine distance, HNSW)
- LLM: Ollama (qwen2.5:7b local, llama3 fallback)
- API: FastAPI (REST)
- Cloud: Azure Blob Storage, Azure Container Registry, Azure Container Apps
- Containerization: Docker + docker-compose

## Project structure

    rag-edf-insight/
    ├── src/
    │   ├── extract.py       # Download PDFs from Azure Blob, extract text
    │   ├── index.py         # Chunk text, build ChromaDB vector index
    │   ├── rag.py           # RAG pipeline: retrieve + generate with Ollama
    │   └── api.py           # FastAPI REST API (/health, /ask)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md

## Run locally

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Set up Azure credentials in .env:

    AZURE_STORAGE_CONNECTION_STRING=your_connection_string
    AZURE_CONTAINER_PDFS=pdfs
    AZURE_CONTAINER_PROCESSED=processed

Run the pipeline:

    python src/extract.py
    python src/index.py

Start Ollama and pull the model:

    ollama pull qwen2.5:7b

Start the API:

    uvicorn src.api:app --host 0.0.0.0 --port 8000

## Run with Docker

    docker compose up --build

## API usage

Health check:

    curl http://localhost:8000/health

Ask a question:

    curl -X POST http://localhost:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "Quel est le chiffre d affaires d EDF en 2024 ?"}'

## Cloud deployment

The API is deployed on Azure Container Apps:

    https://app-rag-edf.agreeablerock-08d8729a.francecentral.azurecontainerapps.io/health

The /health endpoint is live. The /ask endpoint requires Ollama which runs locally — the cloud deployment demonstrates containerization and Azure infrastructure skills.

Infrastructure: Azure Blob Storage (PDF hosting) → Azure Container Registry (Docker image) → Azure Container Apps (API hosting).

## Author

Hamza Zaroual — Data Scientist / AI Engineer