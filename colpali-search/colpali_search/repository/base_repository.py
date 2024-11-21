from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class Repository[T](ABC):
    def __init__(self, model: T, session: AsyncSession):
        self.model = model
        self.session = session

    async def get_all(self) -> list[T]:
        obj = await self.session.scalars(select(self.model))
        return obj.all()

    async def delete(self, id: int) -> None:
        instance = await self.session.get(self.model, id)
        if instance:
            await self.session.delete(instance)

    @abstractmethod
    async def add(self, **kwargs: object) -> T:
        raise NotImplementedError
