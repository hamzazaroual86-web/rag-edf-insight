import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHROMA_PATH = os.path.expanduser("~/rag-edf-insight/vectorstore")
COLLECTION_NAME = "edf_reports"
OLLAMA_URL = "http://localhost:11434/api/generate"
PRIMARY_MODEL = "qwen2.5:7b"
FALLBACK_MODEL = "llama3:latest"
TOP_K = 5
MAX_DISTANCE = 0.5

embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def get_available_model():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if PRIMARY_MODEL in models:
            return PRIMARY_MODEL
        if FALLBACK_MODEL in models:
            print(f"Primary model unavailable, using fallback: {FALLBACK_MODEL}")
            return FALLBACK_MODEL
        raise RuntimeError("No Ollama model available.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")

def retrieve_chunks(question, year=None):
    embedding = embedder.encode(question).tolist()
    filters = {"year": year} if year else None
    results = collection.query(query_embeddings=[embedding], n_results=TOP_K, where=filters)
    chunks = []
    for doc, distance, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        if distance <= MAX_DISTANCE:
            chunks.append({"text": doc, "meta": meta, "distance": distance})
    return chunks

def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[EDF {c['meta']['year']} — page {c['meta']['page']}]\n{c['text']}" for c in chunks
    )
    return f"""Tu es un expert financier spécialisé sur EDF. Réponds uniquement à partir du contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}

Réponse :"""

def query_ollama(prompt, model):
    resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=300)
    return resp.json()["response"].strip()

def ask(question, year=None):
    model = get_available_model()
    chunks = retrieve_chunks(question, year)
    if not chunks:
        return {"answer": "Aucun document pertinent trouvé.", "sources": [], "model": model}
    prompt = build_prompt(question, chunks)
    answer = query_ollama(prompt, model)
    sources = [{"year": c["meta"]["year"], "page": c["meta"]["page"], "source": c["meta"]["source"]} for c in chunks]
    return {"answer": answer, "sources": sources, "model": model}

if __name__ == "__main__":
    print("RAG EDF Insight — prêt. Tape 'exit' pour quitter.")
    print("Préfixe optionnel : '2023:', '2024:', '2025:' pour filtrer par année\n")
    while True:
        user_input = input("Question : ").strip()
        if user_input.lower() == "exit":
            break
        year = None
        question = user_input
        for y in ["2023", "2024", "2025"]:
            if user_input.startswith(f"{y}:"):
                year = y
                question = user_input[5:].strip()
                break
        result = ask(question, year)
        print(f"\nModèle : {result['model']}")
        print(f"Réponse : {result['answer']}")
        print(f"Sources : {result['sources']}\n")