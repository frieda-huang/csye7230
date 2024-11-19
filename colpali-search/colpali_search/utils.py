from datetime import datetime, timezone
from typing import Any, List, TypeAlias

from numpy.typing import NDArray

VectorList: TypeAlias = List[NDArray]


QueryEmbeddingList: TypeAlias = List[NDArray[Any]]


def get_now():
    return datetime.now(timezone.utc).isoformat()
