"""Cascade delete flattened_embeddings if embedding is deleted

Revision ID: ed87483f811a
Revises: 91179eff2b40
Create Date: 2024-11-12 12:54:32.767998

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ed87483f811a"
down_revision: Union[str, None] = "91179eff2b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        "flattened_embedding_binary_quantize_idx",
        table_name="flattened_embedding",
        postgresql_using="hnsw",
    )
    op.drop_constraint(
        "flattened_embedding_embedding_id_fkey",
        "flattened_embedding",
        type_="foreignkey",
    )
    op.create_foreign_key(
        None,
        "flattened_embedding",
        "embedding",
        ["embedding_id"],
        ["id"],
        ondelete="CASCADE",
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, "flattened_embedding", type_="foreignkey")
    op.create_foreign_key(
        "flattened_embedding_embedding_id_fkey",
        "flattened_embedding",
        "embedding",
        ["embedding_id"],
        ["id"],
    )
    op.create_index(
        "flattened_embedding_binary_quantize_idx",
        "flattened_embedding",
        [sa.text("(binary_quantize(vector_embedding)::bit(128))")],
        unique=False,
        postgresql_using="hnsw",
    )
    # ### end Alembic commands ###
