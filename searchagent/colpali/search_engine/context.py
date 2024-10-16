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

    def execute_strategy(self, query_embeddings: VectorList, top_k: int):
        self._strategy.search(query_embeddings, top_k)
