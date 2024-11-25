from contextlib import asynccontextmanager

from colpali_search.context import app_context, initialize_context
from colpali_search.database import async_session, get_session
from colpali_search.dependencies import BenchmarkServiceDep, SearchSerivceDep
from colpali_search.models import User
from colpali_search.routers import embeddings, files, index
from colpali_search.schemas.endpoints.benchmark import BenchmarkResponse
from colpali_search.schemas.endpoints.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
* **Generate embeddings for benchmarking**


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


@api_v1_router.post("/search")
async def search(
    body: SearchRequest,
    search_service: SearchSerivceDep,
    user_id: int = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> SearchResponse:
    logger.info(
        f"Received search request from user_id={user_id} for query='{body.query}'"
    )

    try:
        result = await search_service.search(user_id, body.query, body.top_k, session)
        return SearchResponse(result=[SearchResult(**item) for item in result])

    except Exception as e:
        logger.error(f"Error during search for user_id={user_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@api_v1_router.post("/benchmark")
async def benchmark(
    top_k: int,
    benchmark_service: BenchmarkServiceDep,
    user_id: int = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> BenchmarkResponse:
    average_recall_score = await benchmark_service.average_recall(
        top_k, user_id, session
    )
    precision_score = await benchmark_service.precision(top_k, user_id, session)
    mrr_score = await benchmark_service.mrr(top_k, user_id, session)

    return BenchmarkResponse(
        average_recall_score=average_recall_score,
        precision_score=precision_score,
        mrr_score=mrr_score,
    )


@app.get("/")
async def info():
    return {
        "service_name": "Search2.0 API",
        "version": "1.0.0",
        "description": "API for searching local files using natural language",
    }


app.include_router(api_v1_router)
