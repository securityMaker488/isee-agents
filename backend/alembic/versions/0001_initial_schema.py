"""Initial schema for I.See Agents Multi-Agent Platform with pgvector support

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-11-05 00:00:00
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="operator"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "model_configs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("base_model", sa.String(150), nullable=False),
        sa.Column("version", sa.String(50), nullable=False, server_default="1.0.0"),
        sa.Column("quantization", sa.String(50), nullable=True),
        sa.Column("context_window", sa.Integer(), nullable=False, server_default="8192"),
        sa.Column("default_temperature", sa.Float(), nullable=False, server_default="0.3"),
        sa.Column("default_top_p", sa.Float(), nullable=False, server_default="0.9"),
        sa.Column("default_max_tokens", sa.Integer(), nullable=False, server_default="1024"),
        sa.Column("adapter_type", sa.String(30), nullable=True),
        sa.Column("adapter_path", sa.String(500), nullable=True),
        sa.Column("adapter_rank", sa.Integer(), nullable=True),
        sa.Column("adapter_alpha", sa.Integer(), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "agents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("slug", sa.String(120), nullable=False, unique=True, index=True),
        sa.Column("slot_number", sa.Integer(), nullable=True, unique=True),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(150), nullable=False),
        sa.Column("base_model", sa.String(150), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("tools_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("knowledge_sources_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("rag_config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("lora_adapter_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("version", sa.String(40), nullable=False, server_default="1.0.0"),
        sa.Column("inference_config_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("test_cases_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("quality_criteria_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("security_constraints_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("status", sa.String(40), nullable=False, server_default="ready"),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_original_slot_integrated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "agent_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("changelog", sa.Text(), nullable=False),
        sa.Column("snapshot_json", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(120), nullable=False),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="600"),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("embedding_model", sa.String(120), nullable=False, server_default="nomic-embed-text"),
        sa.Column("is_isolated", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("mime_type", sa.String(120), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256_hash", sa.String(64), nullable=False, index=True),
        sa.Column("extracted_text", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(40), nullable=False, server_default="indexed"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("knowledge_base_id", sa.String(36), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("embedding_provider", sa.String(60), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    if is_postgres:
        op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vector vector(384);")
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw "
            "ON document_chunks USING hnsw (embedding_vector vector_cosine_ops);"
        )

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("role", sa.String(30), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("provider_used", sa.String(60), nullable=False),
        sa.Column("model_used", sa.String(120), nullable=False),
        sa.Column("is_demo_response", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("tokens_generated", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_per_second", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("citations_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("security_flags_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), nullable=True, index=True),
        sa.Column("event_type", sa.String(80), nullable=False, index=True),
        sa.Column("severity", sa.String(30), nullable=False, server_default="INFO"),
        sa.Column("resource", sa.String(150), nullable=False),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("document_chunks")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
    op.drop_table("agent_versions")
    op.drop_table("agents")
    op.drop_table("model_configs")
    op.drop_table("users")
