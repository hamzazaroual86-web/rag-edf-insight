import os
import zipfile
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- Config ---
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.expanduser("~/rag-edf-insight/vectorstore"))
COLLECTION_NAME = "edf_reports"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TOP_K = 5
MAX_DISTANCE = 0.5
TIMEOUT = 300


def download_vectorstore():
    """Fetch the vectorstore from Azure Blob when it is missing on local disk.
    Runs in the cloud, where the store is not shipped inside the image."""
    if os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
        return
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        return
    from azure.storage.blob import BlobServiceClient
    service = BlobServiceClient.from_connection_string(conn)
    blob = service.get_blob_client(container="vectorstore", blob="vectorstore.zip")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    zip_path = os.path.join(CHROMA_PATH, "vectorstore.zip")
    with open(zip_path, "wb") as f:
        f.write(blob.download_blob().readall())
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(CHROMA_PATH)
    os.remove(zip_path)


# --- Load vectorstore (download from Blob in the cloud) and embedding model ---
download_vectorstore()
embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


def get_available_model():
    """Check which Ollama model is available. Tries qwen2.5:7b first, then llama3."""
    for model in ["qwen2.5:7b", "llama3:latest"]:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": "test", "stream": False},
                timeout=60
            )
            if resp.status_code == 200:
                return model
        except requests.exceptions.RequestException:
            continue
    return None


def retrieve_chunks(query, top_k=TOP_K):
    """Embed the query and retrieve the closest chunks from ChromaDB."""
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        if distance > MAX_DISTANCE:
            continue
        metadata = results["metadatas"][0][i]
        chunks.append({
            "text": doc,
            "source": metadata.get("source", "unknown"),
            "year": metadata.get("year", "unknown"),
            "page": metadata.get("page", "unknown"),
            "distance": round(distance, 4)
        })
    return chunks


def build_prompt(query, chunks):
    """Build the LLM prompt from retrieved chunks."""
    if not chunks:
        return f"Question: {query}\n\nNo relevant context found in EDF reports. Answer based on general knowledge and state that no specific EDF data was found."
    context_parts = []
    for c in chunks:
        header = f"[Source: {c['source']} | Year: {c['year']} | Page: {c['page']} | Distance: {c['distance']}]"
        context_parts.append(f"{header}\n{c['text']}")
    context_block = "\n\n---\n\n".join(context_parts)
    return f"""You are an expert analyst on EDF (Electricite de France) annual reports.
Use ONLY the context below to answer. If the context is insufficient, say so.
Always cite the source document and year.
Answer in the same language as the question.

Context:
{context_block}

Question: {query}

Answer:"""


def query_ollama(prompt, model):
    """Send the prompt to Ollama and return the response text."""
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024}
        },
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()["response"]