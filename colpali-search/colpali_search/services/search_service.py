from colpali_search.services.search_engine.context import SearchContext
from colpali_search.services.search_engine.strategy_factory import SearchStrategyFactory
from colpali_search.types import QueryEmbeddingList


class SearchService:
    async def search(self, query_embeddings: QueryEmbeddingList, top_k: int):

        ctx = SearchContext(
            SearchStrategyFactory.create_strategy("ANNHNSWCosineSimilarity")
        )
        return await ctx.execute_search_strategy(query_embeddings, top_k)
