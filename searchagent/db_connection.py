import os

from sqlalchemy import text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DBNAME = "searchagent"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)
inspector = Inspector(bind=engine)
async_session = async_sessionmaker(bind=engine, expire_on_commit=False)


# Enable the pgvector extension
async def enable_pgvector_extension():
    async with async_session.begin() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
