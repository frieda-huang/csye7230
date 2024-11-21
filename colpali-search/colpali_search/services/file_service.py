from typing import List

from colpali_search.models import File
from colpali_search.repository.repositories import FileRepository
from sqlalchemy.ext.asyncio import AsyncSession


class FileService:
    def __init__(self, session: AsyncSession):
        self.file_repository = FileRepository(File, session=session)

    async def get_file_by_id(self, id: int) -> File:
        return await self.file_repository.get_by_id(id)

    async def get_all_files(self) -> List[File]:
        return await self.file_repository.get_all()

    async def delete_file_by_id(self, id: int):
        return await self.file_repository.delete(id)
