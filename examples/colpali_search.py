from contextlib import asynccontextmanager

from fastapi import FastAPI
from searchagent.colpali.base import ColPaliRag
from searchagent.db_connection import enable_pgvector_extension
from searchagent.ragmetrics.metrics import average_recall, fetch_dataset_column
from searchagent.utils import project_paths

pdfs_dir = project_paths.PDF_DIR
single_file_dir = project_paths.SINGLE_FILE_DIR


rag = ColPaliRag(benchmark=False)
_ = rag.model
_ = rag.processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    await enable_pgvector_extension()
    print("Starting up")
    yield
    print("Shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "ColPali Search API", "docs": "/docs", "health": "/health"}


@app.post("/search")
async def search(query: str):
    response = await rag.search(query=query)
    return {"response": response}


@app.post("/benchmark")
async def benchmark():
    k = 10
    pages = fetch_dataset_column(column_name="page")
    queries = fetch_dataset_column(column_name="query")
    actual_pages = [int(page_number) for page_number in pages]

    score = await average_recall(rag, queries, actual_pages, k)

    return {"score": score}
