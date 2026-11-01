"""Bind maintenance execution to a durable, non-expiring owner and active journal."""

from alembic import op
import sqlalchemy as sa

revision = "0012_context_execution_fence"
down_revision = "0011_context_release"
branch_labels = None
depends_on = None


def upgrade():
    # A pre-fence executor may still be dispatched to ES. This migration is
    # supported before first arm only; do not fabricate ownership for it.
    if (
        op.get_bind()
        .execute(sa.text("SELECT COUNT(*) FROM context_release_gate"))
        .scalar_one()
    ):
        raise RuntimeError(
            "0012 requires an unarmed release gate; armed 0011 needs trusted operator reconciliation before migration"
        )
    op.add_column("context_release_gate", sa.Column("active_journal_id", sa.String(64)))
    op.add_column("context_release_gate", sa.Column("execution_token", sa.String(32)))
    op.add_column("context_release_gate", sa.Column("execution_effect", sa.String(32)))
    op.add_column("context_release_journal", sa.Column("recovery_receipts", sa.JSON()))


def downgrade():
    op.drop_column("context_release_journal", "recovery_receipts")
    for name in ("execution_effect", "execution_token", "active_journal_id"):
        op.drop_column("context_release_gate", name)
