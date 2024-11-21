import asyncio
from contextlib import asynccontextmanager

from colpali_search.context import app_context, initialize_context
from colpali_search.database import async_session
from colpali_search.dependencies import (
    EmbeddingSerivceDep,
    ModelServiceDep,
    SearchSerivceDep,
)
from colpali_search.models import User
from colpali_search.routers import embeddings, files, index
from colpali_search.schemas.endpoints.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from sqlalchemy import select

description = """
ColPali Search API for searching local PDF files using natural language. 🚀

## Default

You can **search local files** or **benchmark the RAG system**.


## Files
You will be able to:

* **Get all files**
* **Get file by id**
* **Delete file by id**


## Embeddings

You will be able to:

* **Generate embeddings for file**
* **Generate embeddings for files**


## Index

You will be able to:

* **List all supported index strategies**
* **Configure index strategy**
* **Reset index strategy**
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_context()
    _ = app_context.model_service.model
    _ = app_context.model_service.processor
    yield
    app_context.model_service = None
    app_context.embedding_service = None
    app_context.pdf_conversion_service = None
    app_context.search_service = None


app = FastAPI(
    lifespan=lifespan,
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


async def get_current_user(email: str = "colpalisearch@gmail.com") -> int:
    async with async_session.begin() as session:
        stmt = select(User).where(User.email == email)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user.id


@api_v1_router.post("/search/{query}")
async def search(
    body: SearchRequest,
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    search_service: SearchSerivceDep,
    user_id: int = Depends(get_current_user),
) -> SearchResponse:
    # Run similarity search over the page embeddings for all the pages in the collection
    # top_indices has the shape of
    # tensor([[12,  0, 14],
    # [15, 14, 11]])
    query, top_k = body.query, body.top_k

    try:
        query_embeddings = await asyncio.to_thread(model_service.embed_query, query)
        await embedding_service.upsert_query_embeddings(
            user_id, query, query_embeddings
        )

        result = await search_service.search(query_embeddings, top_k)
        return SearchResponse(result=[SearchResult(**item) for item in result])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_v1_router.post("/benchmark")
async def benchmark(top_k: int):
    pass


@app.get("/")
async def info():
    return {
        "service_name": "Search2.0 API",
        "version": "1.0.0",
        "description": "API for searching local files using natural language",
    }


app.include_router(api_v1_router)
