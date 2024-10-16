from abc import ABC, abstractmethod

import psycopg
from pgvector.psycopg import register_vector
from searchagent.db_connection import DBNAME
from searchagent.models import VECT_DIM


class IndexingStrategy(ABC):
    @abstractmethod
    def build_index(self):
        pass


class HNSWIndexing(IndexingStrategy):
    def build_index(self):
        """Use expression indexing for binary quantization"""

        SQL_INDEXING = f"""
            CREATE INDEX ON flattened_embedding
            USING hnsw ((binary_quantize(vector_embedding)::bit({VECT_DIM})) bit_hamming_ops);
            """
        with psycopg.connect(dbname=DBNAME, autocommit=True) as conn:
            register_vector(conn)
            conn.execute(SQL_INDEXING)


class IVFFlatIndexing(IndexingStrategy):
    def build_index(self):
        pass
