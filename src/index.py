import json
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
BATCH_SIZE = 2000
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHROMA_PATH = os.path.expanduser("~/rag-edf-insight/vectorstore")
COLLECTION_NAME = "edf_reports"

YEARS = ["2023", "2024", "2025"]


def get_blob_client():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(conn_str)


def load_pages(client, year):
    container = os.getenv("AZURE_CONTAINER_PROCESSED")
    blob = client.get_blob_client(container=container, blob=f"edf_{year}.json")
    data = blob.download_blob().readall()
    return json.loads(data)


def chunk_pages(pages, year):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks, metadatas, ids = [], [], []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunk_id = f"edf_{year}_p{page['page']}_c{i}"
            chunks.append(f"EDF {year} — {split}")
            metadatas.append({
                "company": "EDF",
                "year": year,
                "page": page["page"],
                "source": page["source"]
            })
            ids.append(chunk_id)
    return chunks, metadatas, ids


def run():
    client = get_blob_client()
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for year in YEARS:
        print(f"\nIndexing EDF {year}...")
        pages = load_pages(client, year)
        chunks, metadatas, ids = chunk_pages(pages, year)
        print(f"{len(chunks)} chunks created for {year}")

        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()

        for i in range(0, len(chunks), BATCH_SIZE):
            collection.add(
                documents=chunks[i:i+BATCH_SIZE],
                embeddings=embeddings[i:i+BATCH_SIZE],
                metadatas=metadatas[i:i+BATCH_SIZE],
                ids=ids[i:i+BATCH_SIZE]
            )
        print(f"Done indexing {year}")

    print(f"\nTotal documents in vectorstore: {collection.count()}")


if __name__ == "__main__":
    run()