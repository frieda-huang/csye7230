from colpali_search.models import IndexingStrategy
from colpali_search.repository.repositories import IndexingStrategyRepository
from colpali_search.services.search_engine.context import IndexingContext
from colpali_search.services.search_engine.strategy_factory import (
    IndexingStrategyFactory,
)
from colpali_search.types import IndexingStrategyType
from loguru import logger


class IndexingService:
    def __init__(self):
        self.indexing_strategy_repository = IndexingStrategyRepository(IndexingStrategy)

    def _create_context(self, strategy_type: IndexingStrategyType) -> IndexingContext:
        if not isinstance(strategy_type, IndexingStrategyType):
            raise ValueError(f"Invalid strategy_type: {strategy_type}")

        return IndexingContext(
            IndexingStrategyFactory.create_strategy(strategy_type.alias)
        )

    async def get_current_strategy(self):
        await self.indexing_strategy_repository.get_current_strategy()

    async def build_index(
        self,
        strategy_type: IndexingStrategyType = IndexingStrategyType.hnsw_cosine_similarity,
    ):
        logger.info(
            f"Building index: strategy_type={strategy_type}, type={type(strategy_type)}"
        )

        ctx = self._create_context(strategy_type)
        await ctx.execute_indexing_strategy()

    async def drop_indexes(self, strategy_type: IndexingStrategyType):
        ctx = self._create_context(strategy_type)
        has_indexes = await ctx.get_indexes()

        if has_indexes:
            await ctx.execute_drop_indexes()
        else:
            raise ValueError(
                "No indexes found to drop. Ensure that indexing has been built before attempting to drop."
            )

    async def configure_strategy(
        self, strategy_name: IndexingStrategyType
    ) -> IndexingStrategy:
        return await self.indexing_strategy_repository.configure_strategy(strategy_name)

    async def reset_strategy(self):
        await self.indexing_strategy_repository.reset_strategy()
