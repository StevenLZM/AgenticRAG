"""Build server-owned filters from agent-safe retrieval requests."""

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.models import RetrievalRequest, SearchFilter
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class FilterBuilder:
    """Inject immutable user and index scope into an agent retrieval request."""

    def build(
        self,
        request: RetrievalRequest,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> SearchFilter:
        """Return the only filter shape retrieval adapters may receive."""
        return SearchFilter(
            user_id=scope.user_id,
            index_generation=snapshot.index_generation,
            search_type=request.search_type,
            document_ids=request.document_ids,
            content_types=request.content_types,
            date_range=request.date_range,
        )
