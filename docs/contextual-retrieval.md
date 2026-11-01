# Contextual retrieval operations and verification boundary

The context-v1 implementation is opt-in. This branch provides code and offline verification; it has not run a live migration, model evaluation, rebuild, alias change, service restart or deployment. Synthetic supported responses verify plumbing, not LLM semantic accuracy.

## Configuration and production paths

`context_pipeline_enabled`, `query_context_enabled`, and `expected_representation_version` default off/absent. `context_release_enabled` is independent and defaults false. Unarmed legacy sessions issue no context-gate SQL and require no new schema. A release-controlled legacy deployment is the required bridge before adopting context-v1.

All roles use one server-built `RuntimeConfigSnapshot`. It freezes `index_generation`, optional `expected_representation_version='context-v1'`, `query_context_version`, and the build configuration fingerprint. Absent new fields are excluded from snapshot hashing. Context-v1 cannot be selected by an ES hit or stored source item declaring its own schema. Its missing provenance fails closed. Context-enabled source expansion rejects incompatible historical generation/policy snapshots; reprepare those answers on the selected generation. Historical legacy deployments retain their original route.

Production wiring:

- `bootstrap.build_container` installs session admission identity, the SQL multi-build authority, shared source hydrator, optional QueryCompiler, API CoordinationStore and ChatSourceService. Trusted profiles/entity directory/coverage are explicit injectable ports, absent by default. No user profile is inferred or ACL granted.
- `scripts/run_ingestion_worker.py` uses the 320-token child builder, actual embedding tokenizer, SQL build reservations, ContextualizationService, enabled graph node, context-v1 ES policy, and registry-aware publisher together. Publication registers the immutable build and missing rollback-generation requirement in the source publication transaction. Release-controlled publication never switches the global alias itself.
- Macro QueryGraph and Supervisor workers use the same compiler, frozen policy and source authority for retrieval, evidence preparation, final citations, final freezing, streaming/replay and public source expansion. Reranking binds the actual loaded encoder tokenizer and minimum finite encoder/tokenizer/model pair capacity. Maximum inference accounting reserves two pairs per admitted candidate; observed completed pairs remain separate.
- `RebuildExecutor` uses `retrieval_rebuild_jobs`, not completed upload jobs. Its `RebuildClaim` selects the rebuild table's live lease predicate in the **same** SQL transaction as context reservations/result acceptance. Upload workers/reconcilers cannot claim these jobs. Restart reclaims expired leases with a new generation; three failed/expired attempts exhaust the job. Unknown provider spend remains charged. Idempotent publication does not infer rebuild completion from an active source version.

Set identical generation/build/query settings on all deployment roles. Each process also has a unique `context_instance_id`. `context_deployment_roles` must cover `api`, `ingestion`, `rebuild`, and either `query`, or `coordinator` plus `supervisor/retrieval/analysis/evidence/answer` (the combined `all` role may replace the latter five).

An enabled structural-only build is `ContextBuildConfig(enabled=True)` with an explicit embedding tokenizer identity and no extraction/verification models. It needs SQL/artifacts/source validation but no model client, model tokenizer, model budget or global model slot. For LLM builds configure both model IDs, their independent tokenizer identities, provider fingerprint, finite per-document request/token/cost caps, conservative input/output prices covering both models, timeout and global concurrency. No expensive fallback model is selected.

`context_model_tokenizers` maps each generation model ID to a locally installed tokenizer path. `tokenizer_identity(loaded_tokenizer, model_id)` returns `model_id:sha256(serialized_tokenizer_backend)`. Freeze those exact identities in the build config; the embedding identity uses the configured embedding model ID. Model tokens and embedding tokens use separate counters. Enabled composition only loads local tokenizer assets, checks fingerprints, uses SDK retries 0 and gateway provider attempts 1 (no hidden schema repair), and wraps the gateway in SQL global capacity. SQL capacity is shared across worker instances, rejects inconsistent limits, and retains failed-call leases until timeout plus a 60-second grace period. Capacity contention can produce a recorded conservative fallback; it does not multiply provider concurrency by worker count.

The first model-slot acquisition fixes `max_concurrency` for the lifetime of the durable `context-model` limiter record. Releasing every lease and replacing all workers with the same new setting does not change that stored capacity; the new limit is still rejected. There is currently no supported resize operation. Worker startup must not reset or delete the record. Any future resize operation must be a separate maintenance-only procedure that preserves uncertain leases.

## Additive schema and source immutability

Apply `0010_context_builds`, `0011_context_release`, and `0012_context_execution_fence` through the existing Alembic maintenance workflow **before enabling release control**. Startup performs read-only schema checks; it never migrates. Migration declarations are frozen; 0010/0011 add tables and 0012 adds execution-fence fields. Startup explicitly requires the 0012 fields. 0012 refuses an already-armed 0011 gate before any DDL: earlier executor ownership cannot be inferred safely. This pre-release implementation supports migration before first arm; it provides no automatic reconciliation of an armed, unfenced deployment. Do not downgrade while any context-aware role is running.

`retrieval_builds` uniquely binds owner/version/generation to immutable manifest ref, representation policy, build fingerprint, original Parent projection hash and final child ID/search-hash digest. `document_versions` manifest/generation/count fields are the selected compatibility projection, not history. Registry membership still requires the document's current active source version, owner and no deletion/revocation fence.

`IndexWriter.stage_search_representation` checks existing active Parent bytes/coordinates, stages and embeds only children, reads back their exact IDs/hashes, then registers readiness under the live rebuild lease. It does not change Parent status/content or mark the active version building. It refuses the active generation. Rebuilds preserve Parent boundaries and the source embedding-model contract; changing Parent boundaries or the source embedding model is outside this child-only route.

Deletion marks every registered build revoked, cancels rebuild leases, and deletes child records across all registered generations through the existing reconciler. Document artifact-scope deletion includes all context artifacts. Registry hashes/usage metadata can remain as audit records; they cannot authorize deleted source text. New source publication cancels obsolete-version rebuilds and supersedes their registry rows.

## Maintenance authority and traffic gate

A database cannot prove an unknown old process has stopped. First arm and every switch/rollback therefore require a signed receipt from an explicitly trusted deployment manager. Configure only its Ed25519 raw-public-key base64 and issuer (`context_maintenance_public_key`, `context_maintenance_issuer`). The CLI does **not** generate or self-sign a stop certificate. Without this external authority, mutation readiness is blocked; offline plans still work.

The signed canonical JSON payload binds issuer, nonce, from/to generation, complete previous/new `{instance,role}` inventories, target snapshot hash, deployment ID, stopped-deployment proof reference, issue time and expiry (maximum one day). Signature validation does not observe physical process termination: completeness and stop-proof truth are external trust responsibilities. The issuer must establish that all old/unupgraded/unarmed instances are stopped and that the new inventory uses this release-controlled code. A receipt cannot omit a known old instance, add an unknown previous instance, omit a configured role, change config/generation, expire or reuse its durable nonce. The journal retains its receipt hash. Already accepted journals can be resumed without issuing another nonce; new compensating actions need a new signed receipt.

Each actual new instance validates dependencies, ES mapping/alias and schema before attesting using its server-owned session configuration. Reopening requires the exact signed instance inventory, not one arbitrary boolean or one receipt per role. Stale processes with a different selected configuration cannot admit or claim work. Starting an unmanaged process outside the signed inventory violates the external deployment precondition; the database gate is not an enterprise process supervisor or IAM system.

The state machine is `open -> draining -> frozen -> open`:

1. `draining` refuses new run/upload/delete admission with retryable maintenance errors. Existing queued tasks can still claim under the old configuration, heartbeat and finish. This is the progress path; the queue is not stranded behind a closed claim gate.
2. Drain checks queued/running uploads and rebuilds, unfinished source publication, Macro runs, Supervisor runs and queued/running/retry-wait work. Unfinished work must finish or be explicitly cancelled through its existing lifecycle. Failed rebuilds are not ready builds.
3. `frozen` rejects claims/publication. The journal binds before/after mappings for **every owner** in the current effective corpus. It independently verifies manifests, Parent sources, final ES child identity/hash collections and policy. A missing build blocks readiness.
4. Compatibility projections, exact active ES membership, alias switch and configuration are separate idempotent boundaries. The target index first deactivates all old document members, then activates only the current corpus mapping. A non-expiring durable execution token binds every step to the active journal, maintenance receipt and target configuration. Stage transitions are monotonic compare-and-swap operations; completed and aborted journals cannot replay. Only the owning executor may reopen. Known completed boundaries can resume without replaying earlier ES operations.
5. Processes restart with the selected configuration, attest, and the journal rechecks the target before opening. A crash after reopening can finish the journal without replaying an unsafe switch.

## CLI: plan first, then explicit execution

All commands default to offline plan mode and do not construct network clients. Files in examples are operator-owned private configuration; keep credentials outside shared reports. The same explicit inputs are required for execute. Do not run these examples against a live system without a separately authorized maintenance operation.

```sh
python scripts/rebuild_context_index.py enqueue \
  --source old-v1 --target context-v1 \
  --config /secure/current.json --target-config /secure/target.json \
  --manifest /secure/corpus-plan.json
```

The manifest binds `source_generation`, `target_generation`, `target_snapshot_id`, and an explicit `documents` list with `user_id`, `document_id`, `document_version_id`, `canonical_ast_sha256`. Snapshot hashes are computed by `build_query_snapshot(Settings(...))` from the exact deployed code/config. The plan output says readiness is **not checked offline**. `--execute` is never implied.

A controlled rollout sequence is:

1. Apply additive schema through 0012, before first arm, during authorized maintenance. Obtain a trusted receipt proving replacement of the complete unmanaged deployment with the release-controlled legacy deployment. `install` uses equal source/target generations and the legacy configuration; first arm cannot switch generations. Start every signed instance, then `open`. This is the external trust boundary that preserves default-off legacy startup without letting old unguarded processes participate in a release.
2. `import-source --execute` validates current raw/context artifacts and ES child identities and registers old builds. `enqueue --execute` creates durable target jobs for the explicit source manifest. `work --execute --max-jobs N` is a bounded restartable worker invocation for queued/expired jobs of that target generation; run it repeatedly under your worker supervisor. It is not a user-scoped query API. No generation is modified in place.
3. Finish offline and separately authorized real-service/model acceptance. Pause admission with `pause --execute`. Let already queued work drain. Reconcile candidate coverage for any newly active sources. Obtain the new deployment manager receipt and use `release --execute --journal RELEASE_ID --maintenance-receipt RECEIPT` with the same explicit config/manifest arguments. A `not_ready` result keeps user admission closed; it is not release success. Missing builds are checked before freezing, so shadow jobs can be configured and drained. If the corpus races the final pre-journal freeze, `continue-drain --execute` restores only maintenance worker progress; it refuses unfinished journals and never reopens user admission.
4. After the journal selects target configuration, restart the signed inventory with matching configuration and run `resume --execute --journal RELEASE_ID`. It requires the existing validated journal and exact target inputs. `open` is also available for the initial armed deployment once every instance has attested.

For rollback use `rollback --execute --journal ORIGINAL_RELEASE_ID --maintenance-receipt NEW_RECEIPT` with source=current generation and target=the original configuration, then resume `ORIGINAL_RELEASE_ID:rollback`. A partial switch uses a separate compensating journal and marks the original aborted. It selects current source versions only; it never restores an old active-version pointer or a deleted document. During the rollback window publication records missing old-generation builds as `blocked_config`. Supply the explicit compatible old target configuration through `enqueue` to make these durable jobs executable, and run `work`. The entire current-source rollback mapping must match both the old frozen representation and build-configuration fingerprint; resume revalidates that invariant. Until compatible builds are ready, full rollback is `not_ready`; no current material is silently omitted. Exhausted/failed jobs and missing config require operator correction, not free retries or source rollback.

## Uncertain ES writes and execution recovery

Before membership or alias writes, the executor commits an `execution_effect` latch under its exclusive token. No timeout-based takeover exists. A request timeout, cancellation, process crash or lost response may leave remote ES work running: resume, rollback, open and maintenance worker reopening remain blocked. Every subsequent remote write and stage transition rechecks the active journal/token/receipt. Normal exceptions at known completed boundaries release ownership and allow ordinary resume. A process crash between those boundaries also retains ownership and requires the recovery procedure below.

Inspect the current durable identity without mutating it (all common files/arguments still bind the selected source/target):

```sh
python scripts/rebuild_context_index.py execution-status --execute \
  --journal RELEASE_ID --source old-v1 --target context-v1 \
  --config /secure/current.json --target-config /secure/target.json \
  --manifest /secure/corpus-plan.json
```

The JSON output includes journal/stage/state, execution token, current effect, maintenance receipt hash, exact from/to generation, target config hash and instance inventories. Send this binding to the trusted deployment manager. It must **stop the old release executor and prove all of its related ES writes have completed or been cancelled**. Request a fresh signed maintenance envelope using the existing issuer/nonce/time/deployment/inventory fields and this additional payload object:

```json
{
  "execution_recovery": {
    "journal_id": "RELEASE_ID",
    "execution_token": "EXACT_32_HEX_TOKEN_FROM_STATUS",
    "maintenance_receipt_hash": "EXACT_64_HEX_HASH_FROM_STATUS",
    "effect": "membership",
    "stopped_executor_ref": "deployment-audit://stopped/release-executor",
    "es_quiescence_ref": "es-audit://all-related-writes-completed-or-cancelled"
  }
}
```

Use the exact status `effect` (which can be null), generations, config hash and unchanged inventories. This is an extension to the complete maintenance payload, not a standalone signature. The proof references' truth is the external authority's responsibility; SQL and signature validation do not observe remote quiescence. A CLI-generated signature, operator boolean, elapsed timeout or local PID observation cannot replace it. Ordinary installation/release receipts cannot clear an executor; recovery receipts cannot authorize a deployment switch.

Run `recover-execution --execute --journal RELEASE_ID --maintenance-receipt /secure/recovery.json` with the same common arguments. It consumes the one-time nonce, records the complete signed recovery envelope, and clears only the matching current execution token/effect; it does not open traffic, change the selected journal/receipt/configuration, or perform ES writes. Then run ordinary `resume --execute --journal RELEASE_ID`, or request a separately signed rollback. Wrong journal/token/receipt/target/configuration, expired proof, inventory changes and replay are rejected. If the old coroutine returns late after accepted recovery, its token cannot authorize another write or stage transition. Safe recovery still relies on the trusted manager settling already-dispatched requests before issuing the proof; this code does not implement an ES-wide remote cancellation/fencing service.

## Offline evaluation and observations

`evals/context_ablation.py` defines A (legacy384), B (320/raw rerank), C (same320/structural), D (same boundaries/verified LLM), E (D's index/query scope). `validate_ablation_configs` enforces equal corpus/ACL/models/candidate/task budgets and fixed B/C/D/E source boundaries. Gold matches document/version/block/range coverage, independent of final child IDs. `source_metrics` and `summarize_recorded` report source recall, rank metrics, supporting-source/attribution failures, run/config/build provenance, tokens/cost and P95 recorded latency. These are recorded-run analyses, not fresh model-quality evidence, confidence intervals or an automatic go-live verdict. Keep tuning/held-out splits and multi-company strata separate in the input harness.

Durable context reservations retain attempted requests, known consumed and unknown reserved token/cost usage. `context_build_observations` records document/model trigger/completion, field verification outcomes, budget/provider fallback reason codes, source-integrity failures, stage latency and before/after search hashes. Logs contain IDs/hashes/enums, never profile/source text or untrusted verifier explanations. Context source permissions and answer support closure remain independent of a successful tool or arithmetic result.

Offline SQLite/local-artifact/double tests establish contract and recovery behavior. Real MySQL lock behavior, real ES mapping/alias/readback behavior, provider billing/latency, tokenizer deployment identities, external maintenance issuer integration, LLM false acceptance and business retrieval gains require future authorized acceptance. No mock outcome is evidence of production semantic quality.
