import numpy as np
from colpali_search.models import (
    Base,
    Embedding,
    File,
    FlattenedEmbedding,
    IndexingStrategy,
    Page,
    Query,
    User,
)
from colpali_search.utils import get_now
from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.orm import Session

engine = create_engine("postgresql+psycopg://localhost/colpalisearch_test")

with Session(engine) as session:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()

array_engine = create_engine("postgresql+psycopg://localhost/colpalisearch_test")


@event.listens_for(array_engine, "connect")
def connect(dbapi_connection, connection_record):
    register_vector(dbapi_connection)


Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)


def create_data():
    session = Session(engine)

    vectors = [np.array([1.0] * 128, dtype=np.float16)]

    user = User(
        id=1,
        email="test@gmail.com",
        password_hash="123",
        created_at=get_now(),
    )
    session.add(user)

    file = File(
        id=1,
        filename="test.pdf",
        filetype="pdf",
        total_pages=1,
        last_modified=get_now(),
        created_at=get_now(),
    )
    session.add(file)

    page = Page(
        id=1,
        page_number=1,
        last_modified=get_now(),
        created_at=get_now(),
        file=file,
    )
    session.add(page)

    embedding = Embedding(
        id=1,
        vector_embedding=vectors,
        last_modified=get_now(),
        created_at=get_now(),
        page=page,
    )
    session.add(embedding)

    flattened_embedding = FlattenedEmbedding(
        id=1,
        vector_embedding=vectors[0],
        last_modified=get_now(),
        created_at=get_now(),
        embedding=embedding,
    )
    session.add(flattened_embedding)

    query = Query(
        id=1,
        text="test query",
        vector_embedding=vectors,
        created_at=get_now(),
        user=user,
    )
    session.add(query)

    indexing_strategy = IndexingStrategy(
        id=1,
        strategy_name="ExactMaxSim",
        created_at=get_now(),
    )
    session.add(indexing_strategy)

    session.commit()


class TestSqlalchemy:
    def setup_method(self, test_method):
        with Session(engine) as session:
            models = [
                Query,
                FlattenedEmbedding,
                Embedding,
                Page,
                File,
                User,
                IndexingStrategy,
            ]
            for each in models:
                session.query(each).delete()
            session.commit()

    def test_file_orm(self):
        create_data()
        with Session(engine) as session:
            file_stmt = select(File)
            page_stmt = select(Page)
            file = session.scalar(file_stmt)
            page = session.scalar(page_stmt)

            assert file.filename == "test.pdf"
            assert file.pages[0].id == page.id

    def test_page_orm(self):
        create_data()
        with Session(engine) as session:
            page_stmt = select(Page)
            page = session.scalar(page_stmt)

            assert page.file_id == 1
            assert page.page_number == 1

    def test_embedding_orm(self):
        create_data()
        with Session(array_engine) as session:
            embedding_stmt = select(Embedding)
            page_stmt = select(Page)
            embedding = session.scalar(embedding_stmt)
            page = session.scalar(page_stmt)

            assert embedding.id == page.id

    def test_flattened_embedding_orm(self):
        create_data()
        with Session(array_engine) as session:
            stmt = select(FlattenedEmbedding)
            flattened_embedding = session.scalar(stmt)

            expected_vector = np.array([1.0] * 128, dtype=np.float16)

            assert np.array_equal(
                flattened_embedding.vector_embedding.to_list(), expected_vector
            )

    def test_query_orm(self):
        pass

    def test_user_orm(self):
        create_data()
        with Session(engine) as session:
            stmt = select(User)
            user = session.scalar(stmt)

            assert user.id
            assert user.email == "test@gmail.com"

    def test_indexing_strategy(self):
        create_data()
        with Session(engine) as session:
            stmt = select(IndexingStrategy)
            indexing_strategy = session.scalar(stmt)

            assert indexing_strategy.strategy_name == "ExactMaxSim"
