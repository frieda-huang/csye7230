from abc import ABC, abstractmethod
from typing import List, TypeAlias

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

VectorList: TypeAlias = List[np.array]


class RankingStrategy(ABC):
    @abstractmethod
    def execute_ranking_function(self, query_embeddings: VectorList, top_k: int):
        pass


class MaxSimStrategy(RankingStrategy):
    def execute_ranking_function(self, query_embeddings: VectorList, top_k: int):
        """Perform exact search over the vectors
        It compares every query vector with every document vector
        and computes the maximum similarity.

        Based on https://github.com/pgvector/pgvector-python/blob/master/examples/colbert/exact.py
        """

        SQL_MAXSIM_FUNC = """
        CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[])
        RETURNS double precision AS $$
            WITH queries AS (
                SELECT row_number() OVER () AS query_number, *
                FROM (SELECT unnest(query) AS query)
            ),
            documents AS (
                SELECT unnest(document) AS document
            ),
            similarities AS (
                SELECT query_number, 1 - (document <=> query) AS similarity
                FROM queries CROSS JOIN documents
            ),
            max_similarities AS (
                SELECT MAX(similarity) AS max_similarity
                FROM similarities GROUP BY query_number
            )
            SELECT SUM(max_similarity)
            FROM max_similarities
        $$ LANGUAGE SQL
        """

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
            page p ON tp.page_id = p.id
        JOIN
            file f ON p.file_id = f.id
        ORDER BY
            tp.max_sim DESC;
        """

        with psycopg.connect(dbname="searchagent", autocommit=True) as conn:
            with conn.cursor() as cur:
                register_vector(conn)
                cur.execute(SQL_MAXSIM_FUNC)
                result = conn.execute(
                    SQL_RETRIEVE_TOP_K_DOCS, (query_embeddings,)
                ).fetchall()

                for row in result:
                    print(row)


class HammingDistanceStrategy(RankingStrategy):
    def execute_ranking_function(self, query_embeddings: VectorList, top_k: int):
        pass


class ColPaliStrategyContext:
    def __init__(self, strategy: RankingStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> RankingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: RankingStrategy):
        self._strategy = strategy

    def execute_ranking_function(self, query_embeddings: VectorList, top_k: int):
        return self.strategy.execute_ranking_function(query_embeddings, top_k)
