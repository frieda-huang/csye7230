"""Initial migration

Revision ID: cc78dcc35341
Revises:
Create Date: 2024-11-27 20:43:06.830195

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector


# revision identifiers, used by Alembic.
revision: str = "cc78dcc35341"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "file",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("filetype", sa.String(), nullable=False),
        sa.Column("total_pages", sa.Integer(), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column("last_modified", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "indexing_strategy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("strategy_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "page",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text_content", sa.String(), nullable=True),
        sa.Column("binary_content", sa.LargeBinary(), nullable=True),
        sa.Column("last_modified", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["file_id"],
            ["file.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "query",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column(
            "vector_embedding",
            sa.ARRAY(pgvector.sqlalchemy.halfvec.HALFVEC(dim=128)),
            nullable=False,
        ),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "embedding",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "vector_embedding",
            sa.ARRAY(pgvector.sqlalchemy.halfvec.HALFVEC(dim=128)),
            nullable=False,
        ),
        sa.Column("last_modified", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["page_id"],
            ["page.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "flattened_embedding",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "vector_embedding",
            pgvector.sqlalchemy.halfvec.HALFVEC(dim=128),
            nullable=False,
        ),
        sa.Column("last_modified", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("embedding_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["embedding_id"], ["embedding.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("flattened_embedding")
    op.drop_table("embedding")
    op.drop_table("query")
    op.drop_table("page")
    op.drop_table("user")
    op.drop_table("indexing_strategy")
    op.drop_table("file")
    # ### end Alembic commands ###