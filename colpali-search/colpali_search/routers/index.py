from enum import Enum

from fastapi import APIRouter

index_router = APIRouter(prefix="/index", tags=["index"])


class IndexStrategy(str, Enum):
    hnsw_cosine_similarity = "hnsw-cs"
    exact_maxsim = "exact"
    hnsw_binary_quantization_hamming_distance = "hnsw-bq-hd"


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
async def configure_index_strategy(strategy: IndexStrategy):
    return {"success": strategy}


@index_router.post("/reset", description="Reset index strategy")
async def reset_index_strategy():
    pass
