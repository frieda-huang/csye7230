from abc import ABC, abstractmethod
from typing import List

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.cursor import Row
from psycopg.rows import dict_row
from searchagent.colpali.search_engine.distance_metrics import HammingDistance, MaxSim
from searchagent.db_connection import DBNAME
from searchagent.utils import VectorList


class SearchStrategy(ABC):
    @abstractmethod
    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        pass


class ExactMaxSimSearchStrategy(SearchStrategy):
    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        max_sim = MaxSim()
        max_sim.calculate()

        SQL_RETRIEVE_TOP_K_DOCS = f"""
        WITH top_pages AS (
            SELECT
                e.page_id, max_sim(e.vector_embedding, %s) AS max_sim
            FROM
                embedding e
            ORDER BY
                max_sim DESC
            LIMIT {top_k}
        )
        SELECT
            p.page_number,
            p.file_id,
            f.*
        FROM
            top_pages tp
        JOIN
            page p ON p.id = tp.page_id
        JOIN
            file f ON f.id = p.file_id
        ORDER BY
            tp.max_sim DESC;
        """
        async with psycopg.AsyncConnection.connect(
            dbname=DBNAME, autocommit=True
        ) as conn:
            await register_vector_async(conn)
            r = await conn.execute(SQL_RETRIEVE_TOP_K_DOCS, (query_embeddings,))
            result = await r.fetchall()

            for row in result:
                print(row)

            return result


class ANNHNSWHammingSearchStrategy(SearchStrategy):
    """Scale MaxSim using Binary Quantization and Hamming Distance

    Convert the floating point 128-dimensional vectors to 128-bit vectors
    """

    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        hamming = HammingDistance()
        hamming.calculate()

        # Rerank using cosine distance
        SQL_RERANK = f"""
        WITH hamming_results AS (
            SELECT *
            FROM (
                SELECT *
                FROM hamming(%s)
            )
            ORDER BY vector_embedding <=> query
            LIMIT {top_k}
        )
        SELECT
            p.id AS page_id,
            p.*,
            f.id AS file_id,
            f.*
        FROM
            hamming_results hr
        JOIN
            embedding e ON e.id = hr.embedding_id
        JOIN
            page p ON p.id = e.page_id
        JOIN
            file f ON f.id = p.file_id
        """
        conn = await psycopg.AsyncConnection.connect(
            dbname=DBNAME, autocommit=True, row_factory=dict_row
        )
        async with conn:
            await register_vector_async(conn)
            r = await conn.execute(SQL_RERANK, (query_embeddings,))
            result = await r.fetchall()

            for row in result:
                print(
                    f"Filename: {row['filename']}, Page number: {row['page_number']}, File ID: {row['file_id']}"
                )

        return result


class ANNIVFFlatEuclideanSearchStrategy(SearchStrategy):
    """Use IVFFlat with l2 distance"""

    async def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
        pass
