import asyncio

from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.search_service import SearchService
from fastapi import APIRouter, FastAPI, HTTPException

from .routers import embeddings, files, index

description = """
ColPali Search API for searching local PDF files using natural language. 🚀

## Default

You can **search local files** or **benchmark the RAG system**.


## Files
You will be able to:

* **Get all files**
* **Get file by path**
* **Delete file by path**


## Embeddings

You will be able to:

* **Generate embeddings for file**
* **Generate embeddings for files**
* **Get embedding job status**

## Index

You will be able to:

* **List all supported index strategies**
* **Configure index strategy**
* **Reset index strategy**
"""


app = FastAPI(
    title="ColPali Search",
    description=description,
    version="0.0.1",
    contact={
        "name": "Jingying Huang",
        "email": "jingyingfhuang@gmail.com",
    },
)


api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(embeddings.embeddings_router)
api_v1_router.include_router(files.files_router)
api_v1_router.include_router(index.index_router)


@api_v1_router.post("/search/{query}")
async def search(query: str, top_k: int):
    # Run similarity search over the page embeddings for all the pages in the collection
    # top_indices has the shape of
    # tensor([[12,  0, 14],
    # [15, 14, 11]])
    try:
        model_service = ColPaliModelService()
        embedding_service = EmbeddingSerivce()
        search_service = SearchService()

        query_embeddings = await asyncio.to_thread(model_service.embed_query, query)
        await embedding_service.upsert_query_embeddings(query, query_embeddings)

        result = search_service.search(query_embeddings, top_k)

        return {"results": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_v1_router.post("/benchmark")
async def benchmark(top_k: int):
    pass


app.include_router(api_v1_router)


@app.get("/")
async def info():
    return {
        "service_name": "Search2.0 API",
        "version": "1.0.0",
        "description": "API for searching local files using natural language",
    }
