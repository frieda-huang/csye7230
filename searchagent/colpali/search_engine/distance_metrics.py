from abc import ABC, abstractmethod

import psycopg
from pgvector.psycopg import register_vector


class DistanceMetric(ABC):
    @abstractmethod
    def calculate():
        """Calculate similarity scores"""
        pass


class HammingDistance(DistanceMetric):
    def calculate():
        pass


class CosineSimilarity(DistanceMetric):
    def calculate():
        pass


class EuclideanDistance(DistanceMetric):
    def calculate():
        pass


class MaxSim(DistanceMetric):
    def calculate(self):
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
        with psycopg.connect(dbname="searchagent", autocommit=True) as conn:
            register_vector(conn)
            conn.execute(SQL_MAXSIM_FUNC)
