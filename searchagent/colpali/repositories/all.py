import numpy.typing as npt
from searchagent.colpali.repositories.repository import Repository
from searchagent.models import Embedding, File, FlattenedEmbedding, Folder, Page, Query
from searchagent.utils import VectorList, get_now
from sqlalchemy import select


class FolderRepository(Repository[Folder]):
    async def get_by_folder_path(self, folder_path: str) -> Folder:
        async with self.session.begin():
            folder_stm = select(Folder).filter_by(folder_path=folder_path)
            return await self.session.scalar(folder_stm)

    async def add(self, folder_name, folder_path, user_id: str) -> Folder:
        async with self.session.begin():
            folder = Folder(
                folder_name=folder_name,
                folder_path=folder_path,
                user_id=user_id,
                created_at=get_now(),
            )
            self.session.add(folder)
            return folder

    async def delete(self, folder_path: str):
        folder = self.get_by_folder_path(folder_path)
        async with self.session.begin():
            self.session.delete(folder)


class FileRepository(Repository[File]):
    async def get_by_filepath_filename(self, filepath: str, filename: str) -> File:
        async with self.session.begin():
            file_stm = select(File).filter_by(filepath=filepath, filename=filename)
            return await self.session.scalar(file_stm)

    async def add(
        self, filepath: str, filename: str, total_pages: int, folder: Folder
    ) -> File:
        async with self.session.begin():
            file = File(
                filename=filename,
                filepath=filepath,
                filetype="pdf",
                total_pages=total_pages,
                last_modified=get_now(),
                created_at=get_now(),
                folder=folder,
            )
            self.session.add(file)
            return file

    async def delete(self, id: int) -> None:
        pass


class PageRepository(Repository[Page]):
    async def add(self, page_id: int, file: File) -> Page:
        async with self.session.begin():
            page = Page(
                page_number=page_id,
                last_modified=get_now(),
                created_at=get_now(),
                file=file,
            )
            self.session.add(page)
            return page

    async def delete(self, id: int) -> None:
        pass


class EmbeddingRepository(Repository[Embedding]):
    async def add(self, vector_embedding: VectorList, page: Page) -> Embedding:
        async with self.session.begin():
            embedding = Embedding(
                vector_embedding=vector_embedding,
                page=page,
                last_modified=get_now(),
                created_at=get_now(),
            )
            self.session.add(embedding)
            return embedding

    async def delete(self, id: int) -> None:
        pass


class FlattenedEmbeddingRepository(Repository[FlattenedEmbedding]):
    async def add(
        self, vector_embedding: npt.NDArray, embedding: Embedding
    ) -> FlattenedEmbedding:
        async with self.session.begin():
            flattened_embedding = FlattenedEmbedding(
                vector_embedding=vector_embedding,
                last_modified=get_now(),
                created_at=get_now(),
                embedding=embedding,
            )
            self.session.add(flattened_embedding)
            return flattened_embedding

    async def delete(self, id: int) -> None:
        pass


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

    async def delete(self, id: int) -> None:
        pass
