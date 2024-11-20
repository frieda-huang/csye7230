from typing import List

from colpali_search.schemas.common import ImageMetadata
from pydantic import BaseModel, ConfigDict


class EmbeddingsFileResponse(BaseModel):
    id: str
    embeddings: List[List[float]]
    metadata: List[ImageMetadata]

    model_config = ConfigDict(arbitrary_types_allowed=True)
