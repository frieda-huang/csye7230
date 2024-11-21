from colpali_search.models import IndexingStrategy
from colpali_search.repository.repositories import IndexingStrategyRepository
from colpali_search.services.search_engine.context import IndexingContext
from colpali_search.services.search_engine.strategy_factory import (
    IndexingStrategyFactory,
)
from colpali_search.types import IndexingStrategyType
from sqlalchemy.ext.asyncio import AsyncSession


class IndexingService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.indexing_strategy_repository = IndexingStrategyRepository(
            IndexingStrategy, session=session
        )

    async def get_current_strategy(self):
        await self.indexing_strategy_repository.get_current_strategy()

    async def build_index(
        self,
        strategy_type: IndexingStrategyType = IndexingStrategyType.hnsw_cosine_similarity,
    ):
        ctx = IndexingContext(
            IndexingStrategyFactory.create_strategy(strategy_type.alias)
        )
        await ctx.execute_indexing_strategy()

    async def configure_strategy(
        self, strategy_name: IndexingStrategyType
    ) -> IndexingStrategy:
        return await self.indexing_strategy_repository.configure_strategy(strategy_name)

    async def reset_strategy(self):
        await self.indexing_strategy_repository.reset_strategy()
