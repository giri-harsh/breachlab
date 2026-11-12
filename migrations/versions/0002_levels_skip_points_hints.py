"""v1.1 game: levels.skipped_at, levels.points, levels.hints_used (CONTRACTS 7.5)

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("levels") as batch:
        batch.add_column(sa.Column("skipped_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("points", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("hints_used", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("levels") as batch:
        batch.drop_column("hints_used")
        batch.drop_column("points")
        batch.drop_column("skipped_at")
