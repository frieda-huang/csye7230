from typing import Any, List, TypeAlias
from enum import Enum

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

VectorList: TypeAlias = List[NDArray]

QueryEmbeddingList: TypeAlias = List[NDArray[Any]]


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)


class IndexingStrategyType(str, Enum):
    hnsw_cosine_similarity = "hnsw-cs"
    exact_maxsim = "exact"
    hnsw_binary_quantization_hamming_distance = "hnsw-bq-hd"

    @property
    def alias(self):
        aliases = {
            "hnsw-cs": "HNSWCosineSimilarity",
            "exact": "ExactMaxSim",
            "hnsw-bq-hd": "HNSWBQHamming",
        }
        return aliases[self.value]
