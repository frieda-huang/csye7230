from colpali_search.dependencies import IndexingServiceDep
from colpali_search.schemas.endpoints.index import (
    ConfigureIndexResponse,
    ResetIndexResponse,
)
from colpali_search.types import IndexingStrategyType
from colpali_search.utils import get_now
from fastapi import APIRouter, HTTPException

index_router = APIRouter(prefix="/index", tags=["index"])


@index_router.get("/")
async def list_supported_index_strategies():
    strategies = [
        {
            "name": "hnsw-cs",
            "summary": "HNSW with Cosine Similarity",
        },
        {
            "name": "exact",
            "summary": "Exact with MaxSim",
        },
        {
            "name": "hnsw-bq-hd",
            "summary": "HNSW with Binary Quantization and Hamming Distance",
        },
    ]
    return {"strategies": strategies}


@index_router.post("/{strategy}", description="Configure an index strategy")
async def configure_index_strategy(
    strategy: IndexingStrategyType, indexing_service: IndexingServiceDep
):
    try:
        indexing_strategy = await indexing_service.configure_strategy(strategy)
        return ConfigureIndexResponse.model_validate(indexing_strategy)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@index_router.post("/reset", description="Reset index strategy")
async def reset_index_strategy(indexing_service: IndexingServiceDep):
    try:
        await indexing_service.reset_strategy()
        return ResetIndexResponse(
            status="success",
            message="Index strategy has been reset successfully.",
            reset_time=get_now(),
        )
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))
