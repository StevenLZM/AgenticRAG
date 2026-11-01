"""Grounded source and search-representation construction contracts."""

from agentic_rag.ingestion.contextualization.models import (
    CanonicalSourceRef,
    ContextArtifact,
    ContextArtifactRef,
    ContextClaim,
    ContextScopeRegion,
    SearchRepresentation,
)
from agentic_rag.ingestion.contextualization.rendering import (
    ContextRenderingError,
    render_context,
)
from agentic_rag.ingestion.contextualization.sources import (
    SourceCatalog,
    SourceIntegrityError,
)

__all__ = [
    "CanonicalSourceRef",
    "ContextArtifact",
    "ContextArtifactRef",
    "ContextClaim",
    "ContextScopeRegion",
    "SearchRepresentation",
    "ContextRenderingError",
    "render_context",
    "SourceCatalog",
    "SourceIntegrityError",
]
