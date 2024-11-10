from abc import ABC, abstractmethod

from searchagent.db_connection import async_session
from sqlalchemy import select


class Repository[T](ABC):
    def __init__(self, model: T):
        self.model = model

    async def get_all(self) -> list[T]:
        async with async_session() as session:
            obj = await session.scalars(select(self.model))
            return obj.all()

    @abstractmethod
    async def add(self, **kwargs: object) -> T:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, id: int) -> None:
        raise NotImplementedError
