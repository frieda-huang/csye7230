import os

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DBNAME = "searchagent"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(bind=engine, expire_on_commit=False)


# Enable the pgvector extension
async def enable_pgvector_extension():
    async with async_session.begin() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def get_table_names():
    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: inspect(sync_conn).get_table_names()
        )
        return table_names
