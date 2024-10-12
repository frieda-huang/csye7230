from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id = mapped_column(Integer, primary_key=True)
    filename = mapped_column(String, nullable=False)
    filepath = mapped_column(String, nullable=False)
    filesize = mapped_column(Float)
    filetype = mapped_column(String, nullable=False)
    last_modified = mapped_column(DateTime)
    created_at = mapped_column(DateTime, nullable=False)
    content = mapped_column(Text, nullable=True)
    pages: Mapped[List["Page"]] = relationship("Page", back_populates="files")

    def __repr__(self) -> str:
        return f"<File(filename={self.filename}, filetype={self.filetype})>"


class Page(Base):
    __tablename__ = "pages"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_number = mapped_column(Integer, nullable=False)
    text_content = mapped_column(Text, nullable=True)
    binary_content = mapped_column(LargeBinary, nullable=True)
    embedding = mapped_column(Vector(), nullable=True)
    embedding_dim = mapped_column(Integer, nullable=True)
    file_id = mapped_column(Integer, ForeignKey("files.id"))
    file: Mapped["File"] = relationship("File", back_populates="pages")
    last_modified = mapped_column(DateTime)
    created_at = mapped_column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"<Page(page_number={self.page_number}, file_id={self.file_id})>"


class Embedding(Base):
    __tablename__ = "embeddings"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id = mapped_column(Integer, ForeignKey("pages.id"), nullable=True)
    file_id = mapped_column(Integer, ForeignKey("files.id"), nullable=True)
    embedding = mapped_column(Vector(), nullable=True)

    def __repr__(self) -> str:
        return f"<Embedding(page_id={self.page_id}, file_id={self.file_id})>"


class Folder(Base):
    __tablename__ = "folders"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    folder_name = mapped_column(String, nullable=False)
    folder_path = mapped_column(String, nullable=False)
    created_at = mapped_column(DateTime, nullable=False)
    user_id = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="folders")
    file = relationship("File", back_populates="folders", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Folder(folder_name={self.folder_name}, user_id={self.user_id})>"


class User(Base):
    __tablename__ = "users"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    username = mapped_column(String, nullable=False, unique=True)
    email = mapped_column(String, nullable=False, unique=True)
    created_at = mapped_column(DateTime, nullable=False)
    folders = relationship(
        "Folder", back_populates="users", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(username={self.username}, email={self.email})>"
