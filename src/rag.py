import sys
from core import retrieve_chunks, build_prompt, query_ollama, get_available_model


def ask(query):
    """Run the full RAG pipeline: retrieve → build prompt → query LLM."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Step 1 — retrieve relevant chunks
    chunks = retrieve_chunks(query)
    print(f"Retrieved {len(chunks)} relevant chunks")
    for i, c in enumerate(chunks):
        print(f"  [{i+1}] {c['source']} (year={c['year']}, page={c['page']}, dist={c['distance']})")

    # Step 2 — check Ollama
    model = get_available_model()
    if not model:
        print("\nError: no Ollama model available. Is Ollama running?")
        print("Start it with: ollama serve")
        return None

    print(f"\nUsing model: {model}")

    # Step 3 — build prompt and query LLM
    prompt = build_prompt(query, chunks)
    print("Querying LLM...")
    response = query_ollama(prompt, model)

    print(f"\n{'='*60}")
    print("Response:")
    print(f"{'='*60}")
    print(response)
    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Quel est le chiffre d'affaires d'EDF en 2023 ?"

    ask(user_query)