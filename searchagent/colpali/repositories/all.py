import numpy.typing as npt
from searchagent.colpali.repositories.repository import Repository
from searchagent.db_connection import async_session
from searchagent.models import Embedding, File, FlattenedEmbedding, Folder, Page, Query
from searchagent.utils import VectorList, get_now
from sqlalchemy import select


class FolderRepository(Repository[Folder]):
    async def get_by_folder_path(self, folder_path: str) -> Folder:
        async with async_session() as session:
            folder_stm = select(Folder).filter_by(folder_path=folder_path)
            return await session.scalar(folder_stm)

    async def add(self, folder_name, folder_path, user_id: str) -> Folder:
        async with async_session() as session:
            folder = Folder(
                folder_name=folder_name,
                folder_path=folder_path,
                user_id=user_id,
                created_at=get_now(),
            )
            session.add(folder)
            return folder

    async def delete(self, folder_path: str):
        folder = self.get_by_folder_path(folder_path)
        async with async_session() as session:
            session.delete(folder)


class FileRepository(Repository[File]):
    async def get_by_filepath_filename(self, filepath: str, filename: str) -> File:
        async with async_session() as session:
            file_stm = select(File).filter_by(filepath=filepath, filename=filename)
            return await session.scalar(file_stm)

    async def add(
        self, filepath: str, filename: str, total_pages: int, folder: Folder
    ) -> File:
        async with async_session() as session:
            file = File(
                filename=filename,
                filepath=filepath,
                filetype="pdf",
                total_pages=total_pages,
                last_modified=get_now(),
                created_at=get_now(),
                folder=folder,
            )
            session.add(file)
            return file

    async def delete(self, id: int) -> None:
        pass


class PageRepository(Repository[Page]):
    async def add(self, page_id: int, file: File) -> Page:
        async with async_session() as session:
            page = Page(
                page_number=page_id,
                last_modified=get_now(),
                created_at=get_now(),
                file=file,
            )
            session.add(page)
            return page

    async def delete(self, id: int) -> None:
        pass


class EmbeddingRepository(Repository[Embedding]):
    async def add(self, vector_embedding: VectorList, page: Page) -> Embedding:
        async with async_session() as session:
            embedding = Embedding(
                vector_embedding=vector_embedding,
                page=page,
                last_modified=get_now(),
                created_at=get_now(),
            )
            session.add(embedding)
            return embedding

    async def delete(self, id: int) -> None:
        pass


class FlattenedEmbeddingRepository(Repository[FlattenedEmbedding]):
    async def add(
        self, vector_embedding: npt.NDArray, embedding: Embedding
    ) -> FlattenedEmbedding:
        async with async_session() as session:
            flattened_embedding = FlattenedEmbedding(
                vector_embedding=vector_embedding,
                last_modified=get_now(),
                created_at=get_now(),
                embedding=embedding,
            )
            session.add(flattened_embedding)
            return flattened_embedding

    async def delete(self, id: int) -> None:
        pass


class QueryRepository(Repository[Query]):
    async def add(
        self, query: Query, query_embeddings: VectorList, user_id: int
    ) -> Query:
        async with async_session.begin() as session:
            query = Query(
                text=query,
                vector_embedding=query_embeddings,
                created_at=get_now(),
                user_id=user_id,
            )
            session.add(query)
            return query

    async def delete(self, id: int) -> None:
        pass
