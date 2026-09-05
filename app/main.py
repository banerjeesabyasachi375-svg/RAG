from fastapi import FastAPI,UploadFile,File
from app.ingest import ingest_document
from pydantic import BaseModel
from app.query import query_rag

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Enterprise RAG API"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/documents/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunk_count = ingest_document(file_path)

    return {
        "filename": file.filename,
        "chunks_created": chunk_count
    }


class QueryRequest(BaseModel):
    question: str
    session_id: str


@app.post("/query")
async def query(request: QueryRequest):
    answer = query_rag(request.question, request.session_id)

    return {
        "question": request.question,
        "answer": answer
    }