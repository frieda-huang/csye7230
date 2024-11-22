from typing import List

from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    Page,
    Query,
    IndexingStrategy,
)
from colpali_search.repository.base_repository import Repository
from colpali_search.types import VectorList
from colpali_search.utils import get_now
from loguru import logger
from sqlalchemy import delete, select
from colpali_search.types import IndexingStrategyType


class FileRepository(Repository[File]):
    async def get_by_filename(self, filename: str) -> File:
        file_stmt = select(File).filter_by(filename=filename)
        return await self.session.scalar(file_stmt)

    async def get_by_id(self, id: int) -> File:
        return await self.session.get(File, id)

    async def add(
        self,
        filename: str,
        total_pages: int,
    ) -> File:
        file = File(
            filename=filename,
            filetype="pdf",
            total_pages=total_pages,
            last_modified=get_now(),
            created_at=get_now(),
        )
        self.session.add(file)
        return file


class PageRepository(Repository[Page]):
    async def get_by_page_number_and_file(self, page_number: int, file: File) -> Page:
        file_stmt = select(Page).filter_by(id=page_number, file_id=file.id)
        return await self.session.scalar(file_stmt)

    async def add(self, page_number: int, file: File) -> Page:
        page = Page(
            page_number=page_number,
            last_modified=get_now(),
            created_at=get_now(),
            file=file,
        )
        self.session.add(page)
        return page


class EmbeddingRepository(Repository[Embedding]):
    async def get_by_page(self, page: Page) -> Embedding:
        embedding_stmt = select(Embedding).filter_by(page_id=page.id)
        return await self.session.scalar(embedding_stmt)

    async def add(self, vector_embedding: VectorList, page: Page) -> Embedding:
        embedding = Embedding(
            vector_embedding=vector_embedding,
            page=page,
            last_modified=get_now(),
            created_at=get_now(),
        )
        self.session.add(embedding)
        return embedding

    async def delete_by_page(self, page: Page):
        delete_stmt = delete(Embedding).where(Embedding.page_id == page.id)
        await self.session.execute(delete_stmt)

    async def add_or_replace(
        self, vector_embedding: VectorList, page: Page
    ) -> Embedding:
        logger.info(f"Adding or replacing embeddings for file: {page.file.filename}")
        existing_embedding = await self.get_by_page(page)

        if existing_embedding:
            await self.delete_by_page(page)

        return await self.add(vector_embedding, page)


class FlattenedEmbeddingRepository(Repository[FlattenedEmbedding]):
    async def add(
        self, vector_embedding: VectorList, embedding: Embedding
    ) -> List[FlattenedEmbedding]:
        flattened_embedding = [
            FlattenedEmbedding(
                vector_embedding=ve,
                last_modified=get_now(),
                created_at=get_now(),
                embedding=embedding,
            )
            for ve in vector_embedding
        ]
        self.session.add_all(flattened_embedding)
        return flattened_embedding

    async def delete_by_embedding(self, embedding: Embedding):
        delete_stmt = delete(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
        await self.session.execute(delete_stmt)

    async def add_or_replace(
        self, vector_embedding: VectorList, embedding: Embedding
    ) -> List[FlattenedEmbedding]:
        await self.delete_by_embedding(embedding)
        return await self.add(vector_embedding, embedding)


class QueryRepository(Repository[Query]):
    async def add(
        self, query: Query, query_embeddings: VectorList, user_id: int
    ) -> Query:
        async with self.session.begin():
            query = Query(
                text=query,
                vector_embedding=query_embeddings,
                created_at=get_now(),
                user_id=user_id,
            )
            self.session.add(query)
            return query


class IndexingStrategyRepository(Repository[IndexingStrategy]):
    async def add(self, strategy: IndexingStrategy) -> IndexingStrategy:
        return await self.session.add(strategy)

    async def get_current_strategy(self) -> IndexingStrategy:
        indexing_stmt = select(IndexingStrategy).filter_by(id=1)
        return await self.session.scalar(indexing_stmt)

    async def configure_strategy(
        self, strategy_name: IndexingStrategyType
    ) -> IndexingStrategy:
        name = strategy_name.alias
        strategy = await self.get_current_strategy()

        if strategy:
            strategy.strategy_name = name
            strategy.created_at = get_now()
            await self.session.commit()
        else:
            strategy = IndexingStrategy(
                id=1, strategy_name=name, current_time=get_now()
            )
            await self.add(strategy)

        return strategy

    async def reset_strategy(self):
        """Reset to default HNSWCosineSimilarity"""
        await self.configure_strategy(IndexingStrategyType.hnsw_cosine_similarity)
