import time
from fastapi import FastAPI
from pydantic import BaseModel
from core import retrieve_chunks, build_prompt, query_ollama, get_available_model

app = FastAPI(title="RAG EDF Insight API", version="1.0.0")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    chunks_used: int
    response_time: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    # Retrieve chunks
    chunks = retrieve_chunks(req.question)

    # Check Ollama
    model = get_available_model()
    if not model:
        return AskResponse(
            question=req.question,
            answer="Error: Ollama is not running or no model available.",
            model="none",
            chunks_used=0,
            response_time=round(time.time() - start, 2)
        )

    # Build prompt and query LLM
    prompt = build_prompt(req.question, chunks)
    response = query_ollama(prompt, model)

    return AskResponse(
        question=req.question,
        answer=response,
        model=model,
        chunks_used=len(chunks),
        response_time=round(time.time() - start, 2)
    )