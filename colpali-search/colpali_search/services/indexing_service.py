from colpali_search.services.search_engine.context import IndexingContext
from colpali_search.services.search_engine.strategy_factory import (
    IndexingStrategyFactory,
)
from colpali_search.types import IndexingStrategyType


class IndexingService:
    async def configure_index(
        self,
        strategy_type: IndexingStrategyType = IndexingStrategyType.hnsw_cosine_similarity,
    ):
        ctx = IndexingContext(
            IndexingStrategyFactory.create_strategy(strategy_type.alias)
        )
        await ctx.execute_indexing_strategy()
