import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Enable the pgvector extension
with Session.begin() as session:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
