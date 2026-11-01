"""Grounded context build orchestration, immutable artifacts and recovery."""

from __future__ import annotations
import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.ingestion.chunker import (
    ParentChunk,
    apply_search_representation,
    resolve_ast_locator,
)
from agentic_rag.ingestion.contextualization.assembly import ContextAssembler
from agentic_rag.ingestion.contextualization.gateway import (
    ContextModelPort,
    VerificationInput,
    prompt_text,
    ExtractionPayload,
    VerificationPayload,
)
from agentic_rag.ingestion.contextualization.map import DocumentContextMap
from agentic_rag.ingestion.contextualization.rendering import render_context
from agentic_rag.ingestion.contextualization.sources import (
    SourceCatalog,
    SourceIntegrityError,
)
from agentic_rag.ingestion.contextualization.store import (
    BuildKey,
    BudgetExhausted,
    ContextBuildStore,
)
from agentic_rag.ingestion.state import IngestionJobClaim
from agentic_rag.models.contextual import (
    ContextArtifact,
    ContextArtifactRef,
    ContextClaim,
    ContextScopeRegion,
    SearchRepresentation,
)
from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.runtime.model_gateway import (
    StructuredOutputValidationError,
    ModelOutputTruncatedError,
)
from openai import APIError

from agentic_rag.runtime.models import ContextBuildConfig
from agentic_rag.runtime.circuit import CircuitOpenError


@dataclass(frozen=True)
class ContextBuildResult:
    parents: tuple[ParentChunk, ...]
    context_map_ref: ContextArtifactRef
    context_artifact_ref: ContextArtifactRef
    enriched_chunks_ref: ContextArtifactRef
    build_fingerprint: str


def digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


class ContextualizationService:
    def __init__(
        self,
        *,
        artifacts: ArtifactStore,
        store: ContextBuildStore,
        model: ContextModelPort | None,
        count_tokens: Callable[[str], int],
        max_concurrency: int = 1,
        count_model_tokens: Callable[[str, str], int] | None = None,
    ):
        if max_concurrency <= 0:
            raise ValueError("context concurrency must be positive")
        self._artifacts = artifacts
        self._store = store
        self._model = model
        self._count = count_tokens
        self._count_model = count_model_tokens
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_concurrency = max_concurrency

    async def _write(self, base, kind, payload, before):
        await before()
        ref = await asyncio.to_thread(
            self._artifacts.put_json, f"{base}/{kind}/{digest(payload)}.json", payload
        )
        return ContextArtifactRef(
            uri=ref.uri, sha256=ref.sha256, size_bytes=ref.size_bytes
        )

    async def enrich(
        self,
        canonical: CanonicalAst,
        parents: Sequence[ParentChunk],
        *,
        canonical_ref: ContextArtifactRef,
        build: ContextBuildConfig,
        claim: IngestionJobClaim,
        before_side_effect: Callable[[], Awaitable[None]],
    ) -> ContextBuildResult:
        if self._max_concurrency > build.max_concurrency:
            raise ValueError("service concurrency exceeds frozen build")
        if build.extraction_enabled and self._count_model is None:
            raise ValueError(
                "configured extraction requires a separate model token counter"
            )
        if build.extraction_enabled and self._model is None:
            raise ValueError("configured extraction model port is missing")
        if not await asyncio.to_thread(self._artifacts.verify, canonical_ref):
            raise SourceIntegrityError("canonical artifact integrity failure")
        catalog = SourceCatalog.from_ast(canonical, canonical_ref.sha256)
        if (catalog.user_id, catalog.document_id, catalog.document_version_id) != (
            claim.user_id,
            claim.document_id,
            claim.document_version_id,
        ):
            raise SourceIntegrityError("canonical job identity mismatch")
        fingerprint = build.fingerprint(canonical_ref.sha256)
        base = f"documents/{claim.user_id}/{claim.document_id}/{claim.document_version_id}/context/{fingerprint}"
        mapping = DocumentContextMap.build(canonical, catalog)
        map_ref = await self._write(
            base, "maps", mapping.model_dump(mode="json"), before_side_effect
        )
        assembler = ContextAssembler(
            (
                lambda text: max(
                    self._count_model(text, build.extraction_model),
                    self._count_model(text, build.verification_model),
                )
            )
            if build.extraction_enabled
            else self._count
        )
        enriched = []
        accepted_refs = []
        for parent in parents:
            if (parent.user_id, parent.document_id, parent.document_version_id) != (
                claim.user_id,
                claim.document_id,
                claim.document_version_id,
            ):
                raise SourceIntegrityError("parent identity mismatch")
            children = []
            for child in parent.children:
                if (child.user_id, child.document_id, child.document_version_id) != (
                    claim.user_id,
                    claim.document_id,
                    claim.document_version_id,
                ) or child.parent_id != parent.id:
                    raise SourceIntegrityError("child identity mismatch")
                if self._count(child.contextualized_content) > build.base_token_budget:
                    raise ValueError("new context pipeline requires base budget <=320")
                if child.search_representation is not None:
                    raise SourceIntegrityError(
                        "context input must be original raw child"
                    )
                if resolve_ast_locator(canonical, child.ast_locator) != child.content:
                    raise SourceIntegrityError(
                        "child raw content differs from canonical source"
                    )
                key = BuildKey(
                    user_id=claim.user_id,
                    document_id=claim.document_id,
                    document_version_id=claim.document_version_id,
                    build_fingerprint=fingerprint,
                    source_child_id=child.id,
                )
                ready = await self._store.get_ready(key, claim)
                if ready is None:
                    input, regions = assembler.assemble(
                        child,
                        parent,
                        mapping,
                        max_tokens=build.max_input_tokens,
                        max_blocks=build.max_source_blocks,
                        max_hops=build.max_reference_hops,
                    )
                    claims = []
                    if (
                        build.extraction_enabled
                        and not input.context_budget_exhausted
                        and (
                            mapping.scope_candidates
                            or re.search(
                                r"本公司|上述人员|该制度|适用|标准|限额|定义|according|this company",
                                child.content,
                                re.I,
                            )
                        )
                    ):
                        response = await self._invoke(
                            "extract", input, key, claim, build, before_side_effect
                        )
                        if response is not None:
                            if response.target_source_child_id != child.id:
                                raise SourceIntegrityError(
                                    "model target identity mismatch"
                                )
                            for index, candidate in enumerate(response.claims):
                                allowed = {s.source_id for s in input.sources}
                                if any(
                                    q.source_id not in allowed
                                    for q in candidate.source_quotes
                                ):
                                    raise SourceIntegrityError(
                                        "model source outside input whitelist"
                                    )
                                refs = tuple(
                                    catalog.resolve_quote(q.source_id, q.quote)
                                    for q in candidate.source_quotes
                                )
                                scope = regions.get(candidate.scope_region_id)
                                verification = await self._invoke(
                                    "verify",
                                    VerificationInput(
                                        context=input, candidate=candidate
                                    ),
                                    key,
                                    claim,
                                    build,
                                    before_side_effect,
                                )
                                status = (
                                    verification.status
                                    if verification is not None
                                    else "insufficient"
                                )
                                reason = (
                                    verification.reason
                                    if verification is not None
                                    else "provider_or_budget_unavailable"
                                )
                                if verification is not None:
                                    if any(
                                        s not in allowed
                                        for s in verification.source_ids
                                    ):
                                        raise SourceIntegrityError(
                                            "verifier source outside input whitelist"
                                        )
                                    if not {r.source_id for r in refs} <= set(
                                        verification.source_ids
                                    ):
                                        status = "insufficient"
                                        reason = "verifier_missing_support"
                                    # The verifier schema reports source IDs, not
                                    # precise quotes. Persist the complete original
                                    # blocks it used, including dependencies beyond
                                    # the extractor's quotations. Qualification
                                    # checks below must inspect this same closure.
                                    dependency_ids = dict.fromkeys(
                                        (
                                            *[r.source_id for r in refs],
                                            *verification.source_ids,
                                        )
                                    )
                                    refs = tuple(
                                        catalog.resolve_quote(
                                            source_id, catalog.source_text(source_id)
                                        )
                                        for source_id in dependency_ids
                                    )
                                if scope is None or not catalog.covers_child(
                                    scope.source_refs, child
                                ):
                                    status = "insufficient"
                                    reason = "unsupported_scope"
                                quoted = "\n".join(
                                    catalog.verify_source(r) for r in refs
                                )
                                # No invented values/units/periods or stripped known
                                # exceptions, even if a verifier says supported.
                                if candidate.value not in quoted or any(
                                    e not in quoted
                                    for e in candidate.exclusion_conditions
                                ):
                                    status = "insufficient"
                                    reason = "unsupported_value"
                                support_blocks = "\n".join(
                                    catalog.source_text(r.source_id) for r in refs
                                )
                                exceptions = tuple(
                                    m.group(0).strip()
                                    for m in re.finditer(
                                        r"(?:不含|除外|仅限|不适用|except|excluding)[^。；;\n]*",
                                        support_blocks,
                                        re.I,
                                    )
                                )
                                if any(
                                    not any(
                                        e in x for x in candidate.exclusion_conditions
                                    )
                                    for e in exceptions
                                ):
                                    status = "insufficient"
                                    reason = "missing_exclusion"
                                accepted = status == "supported"
                                claims.append(
                                    ContextClaim(
                                        id=f"claim-{index}",
                                        field=candidate.field,
                                        value=candidate.value,
                                        display_value=candidate.value,
                                        scope_region=scope
                                        or ContextScopeRegion(kind="spans"),
                                        source_refs=refs,
                                        derivation="llm",
                                        status="accepted" if accepted else "rejected",
                                        verification_method="llm_supported",
                                        verification_status=status,
                                        allowed_uses=("search",) if accepted else (),
                                        exclusion_conditions=candidate.exclusion_conditions,
                                        rejection_reason=None
                                        if accepted
                                        else (reason or status),
                                    )
                                )
                    conflicts = {
                        c.field
                        for c in claims
                        if c.status == "accepted"
                        and len(
                            {
                                other.value
                                for other in claims
                                if other.field == c.field and other.status == "accepted"
                            }
                        )
                        > 1
                    }
                    claims = [
                        c.model_copy(
                            update={
                                "status": "ambiguous",
                                "allowed_uses": (),
                                "rejection_reason": "conflicting_whole_child_values",
                            }
                        )
                        if c.field in conflicts
                        else c
                        for c in claims
                    ]
                    for c in claims:
                        for r in (*c.source_refs, *c.scope_region.source_refs):
                            catalog.verify_source(r)
                    rep = render_context(
                        child,
                        claims,
                        count_tokens=self._count,
                        build_fingerprint=fingerprint,
                        context_ref=None,
                    )
                    artifact = ContextArtifact(
                        user_id=claim.user_id,
                        document_id=claim.document_id,
                        document_version_id=claim.document_version_id,
                        canonical_ast_sha256=canonical_ref.sha256,
                        source_child_id=child.id,
                        build_fingerprint=fingerprint,
                        claims=tuple(claims),
                        rendered_claim_ids=rep.context_claim_ids,
                    )
                    artifact_ref = await self._write(
                        base,
                        "claims",
                        artifact.model_dump(mode="json"),
                        before_side_effect,
                    )
                    rep = rep.with_context_ref(artifact_ref).require_persisted()
                    ready_ref = await self._write(
                        base,
                        "representations",
                        rep.model_dump(mode="json"),
                        before_side_effect,
                    )
                    ready = await self._store.accept(key, claim, ready_ref)
                rep = SearchRepresentation.model_validate(
                    await asyncio.to_thread(self._artifacts.read_json, ready)
                ).require_persisted()
                if (
                    rep.build_fingerprint != fingerprint
                    or rep.source_child_id != child.id
                ):
                    raise SourceIntegrityError(
                        "accepted representation identity mismatch"
                    )
                artifact = ContextArtifact.model_validate(
                    await asyncio.to_thread(self._artifacts.read_json, rep.context_ref)
                )
                if (
                    artifact.user_id,
                    artifact.document_id,
                    artifact.document_version_id,
                    artifact.source_child_id,
                    artifact.build_fingerprint,
                    artifact.canonical_ast_sha256,
                ) != (
                    claim.user_id,
                    claim.document_id,
                    claim.document_version_id,
                    child.id,
                    fingerprint,
                    canonical_ref.sha256,
                ):
                    raise SourceIntegrityError(
                        "accepted context artifact identity mismatch"
                    )
                for c in artifact.claims:
                    for r in (*c.source_refs, *c.scope_region.source_refs):
                        catalog.verify_source(r)
                if (
                    render_context(
                        child,
                        artifact.claims,
                        count_tokens=self._count,
                        build_fingerprint=fingerprint,
                        context_ref=rep.context_ref,
                    )
                    != rep
                ):
                    raise SourceIntegrityError(
                        "accepted context representation differs from artifact"
                    )
                children.append(apply_search_representation(child, rep))
                accepted_refs.append(rep.context_ref.model_dump(mode="json"))
            enriched.append(parent.model_copy(update={"children": tuple(children)}))
        root_ref = await self._write(
            base,
            "roots",
            {
                "canonical_ref": canonical_ref.model_dump(mode="json"),
                "build_fingerprint": fingerprint,
                "context_refs": accepted_refs,
            },
            before_side_effect,
        )
        chunks_ref = await self._write(
            base,
            "enriched",
            {
                "parents": [p.model_dump(mode="json") for p in enriched],
                "context_artifact_ref": root_ref.model_dump(mode="json"),
                "build_fingerprint": fingerprint,
            },
            before_side_effect,
        )
        return ContextBuildResult(
            tuple(enriched), map_ref, root_ref, chunks_ref, fingerprint
        )

    async def _invoke(self, kind, input, key, claim, build, before):
        prompt = prompt_text(kind)
        schema = ExtractionPayload if kind == "extract" else VerificationPayload
        input_tokens = (
            self._count_model(
                prompt
                + "\n"
                + input.model_dump_json()
                + "\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False),
                build.extraction_model
                if kind == "extract"
                else build.verification_model,
            )
            + 128
        )
        if input_tokens > build.max_input_tokens:
            return None
        tokens = build.max_input_tokens + build.max_output_tokens
        cost = (
            build.max_input_tokens * build.input_price_per_million
            + build.max_output_tokens * build.output_price_per_million
        ) / 1_000_000
        try:
            async with self._semaphore:
                await before()
                reservation = await self._store.reserve(
                    key, claim, tokens=tokens, cost=cost, build=build
                )
                await before()
                response = await asyncio.wait_for(
                    getattr(self._model, kind)(input), build.timeout_seconds
                )
                if response.usage.complete:
                    usage = response.usage
                    actual_cost = (
                        usage.input_tokens * build.input_price_per_million
                        + usage.output_tokens * build.output_price_per_million
                    ) / 1_000_000
                    await self._store.settle(
                        key,
                        claim,
                        reservation,
                        tokens=usage.input_tokens + usage.output_tokens,
                        cost=actual_cost,
                    )
                return response
        except (
            BudgetExhausted,
            TimeoutError,
            ConnectionError,
            CircuitOpenError,
            StructuredOutputValidationError,
            ModelOutputTruncatedError,
            APIError,
        ):
            # Unknown dispatch spend stays reserved across retries and crashes.
            return None
