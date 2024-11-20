from typing import Any, List, TypeAlias

from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

VectorList: TypeAlias = List[NDArray]

QueryEmbeddingList: TypeAlias = List[NDArray[Any]]


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
