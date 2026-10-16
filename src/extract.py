import fitz  # PyMuPDF
import json
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

PDFS = ["edf_2023.pdf", "edf_2024.pdf", "edf_2025.pdf"]
MIN_CHARS = 80  # skip blank or image-only pages


def get_blob_client():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(conn_str)


def download_pdf(client, pdf_name):
    container = os.getenv("AZURE_CONTAINER_PDFS")
    blob = client.get_blob_client(container=container, blob=pdf_name)
    return blob.download_blob().readall()


def extract_text(pdf_bytes, year):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < MIN_CHARS:
            continue
        pages.append({
            "text": text,
            "page": page_num,
            "year": year,
            "source": f"edf_{year}.pdf"
        })
    return pages


def save_to_processed(client, year, pages):
    container = os.getenv("AZURE_CONTAINER_PROCESSED")
    blob_name = f"edf_{year}.json"
    data = json.dumps(pages, ensure_ascii=False, indent=2)
    blob = client.get_blob_client(container=container, blob=blob_name)
    blob.upload_blob(data, overwrite=True)
    print(f"Saved {blob_name} ({len(pages)} pages) to processed/")


def run():
    client = get_blob_client()
    for pdf_name in PDFS:
        year = pdf_name.replace("edf_", "").replace(".pdf", "")
        print(f"Processing {pdf_name}...")
        pdf_bytes = download_pdf(client, pdf_name)
        pages = extract_text(pdf_bytes, year)
        save_to_processed(client, year, pages)
        print(f"Done: {len(pages)} pages extracted for {year}")


if __name__ == "__main__":
    run()