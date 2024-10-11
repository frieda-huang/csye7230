from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped
from typing import List


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    filesize = Column(Float)
    filetype = Column(String, nullable=False)
    last_modified = Column(DateTime)
    created_at = Column(DateTime, nullable=False)
    content = Column(Text, nullable=True)
    pages: Mapped[List["Page"]] = relationship("Page", back_populates="files")

    def __repr__(self) -> str:
        return f"<File(filename={self.filename}, filetype={self.filetype})>"


class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    page_number = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=True)
    binary_content = Column(LargeBinary, nullable=True)
    embeddings = Column(Vector(dim=128), nullable=True)
    file_id = Column(Integer, ForeignKey("files.id"))
    file: Mapped["File"] = relationship("File", back_populates="pages")
    last_modified = Column(DateTime)
    created_at = Column(DateTime, nullable=False)

    def __repr__(self) -> str:
        return f"<Page(page_number={self.page_number}, file_id={self.file_id})>"


class Folder(Base):
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_name = Column(String, nullable=False)
    folder_path = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="folders")
    file = relationship("File", back_populates="folders", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Folder(folder_name={self.folder_name}, user_id={self.user_id})>"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False)
    folders = relationship(
        "Folder", back_populates="users", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(username={self.username}, email={self.email})>"
