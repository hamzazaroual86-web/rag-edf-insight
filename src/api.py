import os
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.expanduser("~/rag-edf-insight/vectorstore"))
COLLECTION_NAME = "edf_reports"
OLLAMA_URL = "http://localhost:11434/api/generate"
PRIMARY_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "llama3:latest"
TOP_K = 5
MAX_DISTANCE = 0.5

app = FastAPI(title="RAG EDF Insight", version="1.0.0")

embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


class AskRequest(BaseModel):
    question: str
    year: str | None = None  # optional: "2023", "2024", "2025"


class Source(BaseModel):
    year: str
    page: int
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    duration_seconds: float


def get_available_model():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if PRIMARY_MODEL in models:
            return PRIMARY_MODEL
        if FALLBACK_MODEL in models:
            return FALLBACK_MODEL
        raise HTTPException(status_code=503, detail="No Ollama model available.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama is not running.")


def retrieve_chunks(question, year=None):
    embedding = embedder.encode(question).tolist()
    filters = {"year": year} if year else None
    results = collection.query(
        query_embeddings=[embedding],
        n_results=TOP_K,
        where=filters
    )
    chunks = []
    for doc, distance, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        if distance <= MAX_DISTANCE:
            chunks.append({"text": doc, "meta": meta, "distance": distance})
    return chunks


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[EDF {c['meta']['year']} — page {c['meta']['page']}]\n{c['text']}"
        for c in chunks
    )
    return f"""Tu es un expert financier spécialisé sur EDF. Réponds uniquement à partir du contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""


def query_ollama(prompt, model):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300
    )
    return resp.json()["response"].strip()


@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_docs": collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if body.year and body.year not in ["2023", "2024", "2025"]:
        raise HTTPException(status_code=400, detail="year must be 2023, 2024 or 2025.")

    model = get_available_model()
    chunks = retrieve_chunks(body.question, body.year)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant document found.")

    start = time.time()
    prompt = build_prompt(body.question, chunks)
    answer = query_ollama(prompt, model)
    duration = round(time.time() - start, 2)

    sources = [
        Source(year=c["meta"]["year"], page=c["meta"]["page"], source=c["meta"]["source"])
        for c in chunks
    ]
    return AskResponse(answer=answer, sources=sources, model=model, duration_seconds=duration)