from typing import List

from psycopg.cursor import Row
from searchagent.colpali.search_engine.search_strategies import SearchStrategy
from searchagent.utils import VectorList


class Context:
    def __init__(self, strategy: SearchStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> SearchStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: SearchStrategy):
        self._strategy = strategy

    async def execute_strategy(
        self, query_embeddings: VectorList, top_k: int
    ) -> List[Row]:
        return await self._strategy.search(query_embeddings, top_k)
