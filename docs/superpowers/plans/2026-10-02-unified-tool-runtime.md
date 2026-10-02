# Unified Tool Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Chat、Fast RAG、Research 使用共享工具发现与执行层，接入只读高德 MCP 并展示多轮地图卡片。

**Architecture:** 在 Worker 内创建 ToolRuntime，原生和 MCP 适配器保持独立。三种执行策略共享目录、权限、预算和调用记录，输出经来源校验后进入现有公开回答与会话链路。

**Tech Stack:** Python 3.11/3.12、Pydantic、LangGraph、SQLite/aiosqlite、MCP Python SDK、原生 JavaScript。

**Spec:** docs/superpowers/specs/2026-10-02-unified-tool-runtime-design.md

## Global Constraints

- 用户明确指定 main，直接在 main 开发；保留 tmp/，不提交临时文件。
- 不把真实密钥写入版本控制、测试、日志、Prompt 或 Checkpoint。
- 不增加 MCP 路由；Chat、Fast RAG、Research 均可访问统一层。
- 所有远端工具首期只读；OAuth/写操作未实现时明确拒绝配置或调用。
- 默认每 Run 12 次物理调用、4 次发现、单调用 20 秒；策略升级不重置总预算。
- 保持现有文档权限、引用、审核、多轮聊天和异步恢复行为。
- 用户已授权设计文档落地后开始开发，无须再次等待文档审批。先完成测试和审查，代码留在工作区供走查，不自动推送。

## Review Focus

- 远端目录或返回值包含注入内容时，不能取得凭据、扩大权限或生成危险链接。
- 同一调用恢复、重复提交或参数变化时，不重复计费重放已成功结果，也不串用户。
- Fast RAG 升级 Research 后继续使用已取得结果并累计预算。
- 公开 Chat 工具结果不能因为 route=chat 跳过事实来源校验。
- MCP 故障和缺失配置不能破坏纯本地 Chat/检索；老回答仍正常显示。

## Task 1: Shared runtime, native adapters and durable journal

**Files:** Create `src/agentic_rag/tool_runtime/{models,runtime,store,native,__init__}.py`; tests under `tests/unit/tool_runtime/`.

**Interfaces:** Produce ToolDefinition, ToolContext, ToolResult, ToolError, ToolAdapter and ToolRuntime exactly as the spec. Native adapter returns EvidenceBatch JSON under data.batch. Store uses SQLite path provided at composition time.

- [x] Write tests for discovery filtering, schema validation, scopes, repeated call IDs, persistent budgets, cancellation and changed arguments.
- [x] Run them and record expected missing-feature failures.
- [x] Implement models, runtime and journal; raw exceptions/secrets never become observations.
- [x] Add native retrieval/calculator adapter tests and implementation using existing services.
- [x] Run `python -m pytest tests/unit/tool_runtime -q` and record results.

## Task 2: MCP SDK transport and credentials

**Files:** Create `src/agentic_rag/tool_runtime/{mcp,credentials}.py`, protocol tests; update SDK dependency in pyproject.toml.

**Interfaces:** Consume Task 1 contracts. Produce McpServerConfig, McpAdapter(config, credentials); none/bearer/header provider; list_tools(context), call_tool(definition, arguments, context), aclose(). MCP data is `{structured: object|null, text: str}`. SDK sessions stay process-owned and are never serialized.

- [x] Test approved tool filtering, secrets, scope boundaries, endpoint validation and SDK response normalization against controlled transport fixtures.
- [x] Verify failures before implementation.
- [x] Implement SSE and Streamable HTTP clients with bounded reads and structured errors.
- [x] Verify the installed SDK API and pin a compatible stable dependency.
- [x] Run MCP transport and credential tests, record evidence.

## Task 3: Public tool results and cards

**Files:** Create `query/tool_answers.py`; modify `query/public_answer.py`, `api/static/chat-view.js`, relevant CSS, projection and browser tests.

**Interfaces:** Consume ToolResult; produce build_tool_answer(results, *, route, document_answer=None). Public fields tool_audited/external_sources/cards as spec. Preserve old answer projection behavior without new fields.

- [x] Test 高德 place/route normalization, malformed/oversized results, unsafe links, mixed answers and Chat audit enforcement.
- [x] Verify failures before implementation.
- [x] Implement deterministic result projection, typed cards and safe DOM rendering with optional details.
- [x] Test legacy messages and card persistence/rendering using browser fixtures.
- [x] Run public-answer and frontend integration tests.

## Task 4: Shared discovery and execution in all strategies

**Files:** query/{chat,fast_rag,research_loop,tools,graph,state,routing_policy,routing_context,router}.py; models/schemas.py; prompts; tests/unit/query.

**Interfaces:** Inject optional ToolRuntime into graph dependencies; create ToolContext from server state. Stable call IDs and tool observations in JSON state; no adapter in state. Route enum remains unchanged.

- [x] Write routing/discovery/call tests across chat/fast_rag/research, including progressive loading and unknown capabilities.
- [x] Observe missing-feature failures, implement bounded actions and escalation without budget reset.
- [x] Route existing native knowledge search and calculator through ToolRuntime when injected; preserve legacy standalone test/dependency compatibility.
- [x] Integrate tool results into source-specific publication, preserving document audits and research evidence packing.
- [x] Add multi-step mixed-source and checkpoint recovery tests; run query suite.

## Task 5: Production composition, configuration and multi-turn integration

**Files:** config.py; runtime/{query_composition,models,evaluation_identity,query_worker}.py; persistence/conversations.py; tests for composition, worker and conversations.

**Interfaces:** Configure disabled-by-default MCP servers and server-side DASHSCOPE_API_KEY alias; include sanitized tool config fingerprint in snapshots. Shared runtime is closed with dependencies. Completed cards become bounded session references.

- [x] Test missing/invalid configuration, partial service availability, resource cleanup and credential-free fingerprints.
- [x] Implement Settings, runtime wiring, public persistence and bounded map context.
- [x] Add three-strategy end-to-end fixtures and same-session follow-up tests.
- [x] Configure local user-authorized key without exposing it and perform minimal authenticated discovery/read-only 高德 probe if network permits.

## Task 6: Regression, review and validation report

- [x] Run default full pytest; run Ruff, Mypy for changed boundaries, JS checks, git diff --check.
- [x] Verify cards in browser, plus existing task switching and two-tab recovery regressions.
- [x] Review whole diff with fresh context; fix concrete findings and run covering tests.
- [x] Write `docs/validation/2026-10-02-unified-tool-runtime.md` with exact commands, results and remaining limits.
- [x] Deliver code and design links; report live integration honestly. Do not push without a user request.

## Completion evidence

Completed on main on 2026-10-02. See `docs/validation/2026-10-02-unified-tool-runtime.md` for full regression (1418 passed, 89 skipped), full Mypy (120 files), three live model + AMap turns, browser checks, and five independently rechecked review fixes. One additional new-Run recovery test passed after the full run. Existing API/Worker was not restarted. Changes remain uncommitted for user walkthrough.
