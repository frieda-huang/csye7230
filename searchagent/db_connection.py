import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import sessionmaker

DBNAME = "searchagent"
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
inspector = Inspector(bind=engine)
Session = sessionmaker(bind=engine)

# Enable the pgvector extension
with Session.begin() as session:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
