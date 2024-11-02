from contextlib import asynccontextmanager

from fastapi import FastAPI
from searchagent.colpali.base import ColPaliRag
from searchagent.utils import project_paths

pdfs_dir = project_paths.PDF_DIR
single_file_dir = project_paths.SINGLE_FILE_DIR
embeddings_filepath = f"{project_paths}/embeddings_metadata.json"

rag = ColPaliRag(input_dir=pdfs_dir, store_locally=False)
app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = rag.model
    _ = rag.processor


@app.get("/")
async def root():
    return {"message": "ColPali Search API", "docs": "/docs", "health": "/health"}


@app.post("/search")
async def search(query: str):
    response = await rag.search(query=query)
    return {"response": response}
