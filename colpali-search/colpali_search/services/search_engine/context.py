from typing import List

from colpali_search.services.search_engine.indexing_strategies import IndexingStrategy
from colpali_search.services.search_engine.search_strategies import SearchStrategy
from colpali_search.types import VectorList
from loguru import logger
from psycopg.cursor import Row


class SearchContext:
    def __init__(self, strategy: SearchStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> SearchStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: SearchStrategy):
        self._strategy = strategy

    async def execute_search_strategy(
        self, query_embeddings: VectorList, top_k: int
    ) -> List[Row]:
        return await self._strategy.search(query_embeddings, top_k)


class IndexingContext:
    def __init__(self, strategy: IndexingStrategy):
        self._strategy = strategy

    @property
    def strategy(self, strategy: IndexingStrategy) -> IndexingStrategy:
        self._strategy = strategy

    async def execute_indexing_strategy(self):
        logger.info("Building index...")
        return await self._strategy.build_index()