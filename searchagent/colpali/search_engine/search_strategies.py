from abc import ABC, abstractmethod
from typing import List

import psycopg
from pgvector.psycopg import register_vector
from psycopg.cursor import Row
from searchagent.colpali.search_engine.distance_metrics import MaxSim, HammingDistance
from searchagent.db_connection import DBNAME
from searchagent.utils import VectorList


class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query_embeddings: VectorList, top_k: int):
        pass


class ExactMaxSimSearchStrategy(SearchStrategy):
    def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
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
        with psycopg.connect(dbname=DBNAME, autocommit=True) as conn:
            register_vector(conn)
            result = conn.execute(
                SQL_RETRIEVE_TOP_K_DOCS, (query_embeddings,)
            ).fetchall()

            for row in result:
                print(row)

            return result


class ANNHNSWHammingSearchStrategy(SearchStrategy):
    """Scale MaxSim using Binary Quantization and Hamming Distance

    Convert the floating point 128-dimensional vectors to 128-bit vectors
    """

    def search(self, query_embeddings: VectorList, top_k: int) -> List[Row]:
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
            p.*,
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
        with psycopg.connect(dbname=DBNAME, autocommit=True) as conn:
            register_vector(conn)
            result = conn.execute(SQL_RERANK, (query_embeddings,)).fetchall()

            for row in result:
                print(row)

        return result


class ANNIVFFlatEuclideanSearchStrategy(SearchStrategy):
    """Use IVFFlat with l2 distance"""

    def search(self, query_embeddings: VectorList, top_k: int):
        pass
