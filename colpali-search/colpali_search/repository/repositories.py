from typing import List

from colpali_search.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    Page,
    Query,
)
from colpali_search.repository.base_repository import Repository
from colpali_search.utils import VectorList, get_now
from loguru import logger
from sqlalchemy import delete, select


class FileRepository(Repository[File]):
    async def get_by_filepath_filename(self, filepath: str, filename: str) -> File:
        file_stm = select(File).filter_by(filepath=filepath, filename=filename)
        return await self.session.scalar(file_stm)

    async def add(
        self,
        filepath: str,
        filename: str,
        total_pages: int,
    ) -> File:
        file = File(
            filename=filename,
            filepath=filepath,
            filetype="pdf",
            total_pages=total_pages,
            last_modified=get_now(),
            created_at=get_now(),
        )
        self.session.add(file)
        return file


class PageRepository(Repository[Page]):
    async def get_by_page_id_and_file(self, page_id: int, file: File) -> Page:
        file_stm = select(Page).filter_by(id=page_id, file_id=file.id)
        return await self.session.scalar(file_stm)

    async def add(self, page_id: int, file: File) -> Page:
        page = Page(
            page_number=page_id,
            last_modified=get_now(),
            created_at=get_now(),
            file=file,
        )
        self.session.add(page)
        return page


class EmbeddingRepository(Repository[Embedding]):
    async def get_by_page(self, page: Page) -> Embedding:
        embedding_stm = select(Embedding).filter_by(page_id=page.id)
        return await self.session.scalar(embedding_stm)

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
        delete_stm = delete(Embedding).where(Embedding.page_id == page.id)
        await self.session.execute(delete_stm)

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
        delete_stm = delete(FlattenedEmbedding).where(
            FlattenedEmbedding.embedding_id == embedding.id
        )
        await self.session.execute(delete_stm)

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
