from abc import ABC, abstractmethod

import psycopg
from colpali_search.database import DBNAME
from colpali_search.models import VECT_DIM
from pgvector.psycopg import register_vector_async


async def execute_postgresql_indexing_command(command: str):
    conn = await psycopg.AsyncConnection.connect(dbname=DBNAME, autocommit=True)

    async with conn:
        await register_vector_async(conn)
        await conn.execute(command)


class IndexingStrategy(ABC):
    @abstractmethod
    async def build_index(self):
        pass

    async def drop_index(self):
        SQL_DROP_INDEX = """
        DO $$
        DECLARE
            index_name TEXT;
        BEGIN
            FOR index_name IN
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename = 'flattened_embedding'
                AND indexname != 'flattened_embedding_pkey'
            LOOP
                EXECUTE format('DROP INDEX IF EXISTS %I', index_name);
            END LOOP;
        END $$;
        """
        await execute_postgresql_indexing_command(SQL_DROP_INDEX)


class HNSWIndexingBinaryQuantizationHammingDistance(IndexingStrategy):
    async def build_index(self):
        """Use expression indexing for binary quantization"""

        SQL_INDEXING = f"""
            CREATE INDEX ON flattened_embedding
            USING hnsw ((binary_quantize(vector_embedding)::bit({VECT_DIM})) bit_hamming_ops);
            """
        await execute_postgresql_indexing_command(SQL_INDEXING)


class HNSWIndexingCosineSimilarity(IndexingStrategy):
    async def build_index(self):
        SQL_INDEXING = f"""CREATE INDEX ON flattened_embedding
        USING hnsw ((vector_embedding::halfvec({VECT_DIM})) halfvec_cosine_ops);
        """
        await execute_postgresql_indexing_command(SQL_INDEXING)
