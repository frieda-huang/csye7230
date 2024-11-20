from typing import List

import numpy as np
import torch
from colpali_search.database import execute_analyze
from colpali_search.models import Embedding, File, FlattenedEmbedding, Page, Query
from colpali_search.repository.repositories import (
    EmbeddingRepository,
    FileRepository,
    FlattenedEmbeddingRepository,
    PageRepository,
    QueryRepository,
)
from colpali_search.schemas.internal.pdf import ImageMetadata
from colpali_search.services.search_engine.context import IndexingContext
from colpali_search.services.search_engine.strategy_factory import (
    IndexingStrategyFactory,
)
from colpali_search.types import VectorList
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingSerivce:
    def __init__(
        self,
        session: AsyncSession,
    ):
        self.session = session

        self.file_repository = FileRepository(File, session=session)
        self.page_repository = PageRepository(Page, session=session)
        self.embedding_repository = EmbeddingRepository(Embedding, session=session)
        self.flattened_embedding_repository = FlattenedEmbeddingRepository(
            FlattenedEmbedding, session=session
        )
        self.query_repository = QueryRepository(Query, session=session)

    async def upsert_doc_embeddings(
        self, embeddings: List[torch.Tensor], metadata: List[ImageMetadata]
    ):
        logger.info("Initiating the upsert process for document embeddings...")
        chunk_size = 500
        embed_len = len(embeddings)

        async with self.session.begin():
            for i in range(0, embed_len, chunk_size):
                embeddings_chunk = embeddings[i : i + chunk_size]
                metadata_chunk = metadata[i : i + chunk_size]

                await self._process_chunk(embeddings_chunk, metadata_chunk)

        await self._finalize_operations()
        logger.info(
            "Successfully completed the upsert process for all document embeddings."
        )

    async def upsert_query_embeddings(
        self, user_id: int, query: str, query_embeddings: List[VectorList]
    ):
        await self.query_repository.add(query, query_embeddings, user_id)

    async def _get_or_add_file(self, metadata: ImageMetadata) -> File:
        filename = metadata.filename

        logger.info(f"Getting or adding file: {filename}")

        file = await self.file_repository.get_by_filename(filename)

        if not file:
            file = await self.file_repository.add(
                filename,
                metadata.total_pages,
            )
        return file

    async def _get_or_add_page(self, file: File, metadata: ImageMetadata) -> Page:
        page_number = metadata.page_number

        page = await self.page_repository.get_by_page_number_and_file(page_number, file)

        if not page:
            page = await self.page_repository.add(page_number, file)

        return page

    async def _process_chunk(
        self, embeddings_chunk: List[torch.Tensor], metadata_chunk: List[ImageMetadata]
    ):
        for embedding, metadata in zip(embeddings_chunk, metadata_chunk):
            await self._process_single_embedding(embedding, metadata)

    async def _process_single_embedding(
        self, embedding: torch.Tensor, metadata: ImageMetadata
    ):
        embeddings = embedding.tolist()
        vector_embedding = [np.array(e) for e in embeddings]

        file = await self._get_or_add_file(metadata)
        page = await self._get_or_add_page(file, metadata)

        embedding = await self.embedding_repository.add_or_replace(
            vector_embedding, page
        )

        await self.flattened_embedding_repository.add_or_replace(
            vector_embedding, embedding
        )

    async def _finalize_operations(self):
        """Update database statistics and execute indexing"""
        await execute_analyze()

        ctx = IndexingContext(
            IndexingStrategyFactory.create_strategy("HNSWCosineSimilarity")
        )
        await ctx.execute_indexing_strategy()
