(() => {
  "use strict";

  const PUBLIC_EVENT_TYPES = new Set([
    "RUN_STARTED", "RUN_COMPLETED", "RUN_FAILED",
    "MEMORY_LOADED", "QUERY_ROUTED", "FAST_RAG_COMPLETED",
    "RESEARCH_LOOP_COMPLETED", "RETRIEVAL_COMPLETED", "EVIDENCE_GRADED",
    "FAITHFULNESS_AUDITED", "CITATION_VALIDATED", "ANSWER_GENERATED",
    "ANSWER_FINALIZED", "TODO_UPDATED", "TOOL_STARTED", "TOOL_COMPLETED",
    "COMPONENT_DEGRADED", "COMPONENT_REFUSED", "RETRIEVAL_DEGRADED",
    "CIRCUIT_OPEN", "OUTBOX_RETRY", "WORKER_DLQ", "QUERY_REFUSED",
    "QUERY_CANCELLED", "QUERY_TIMEOUT", "LEASE_LOST", "MODEL_RETRY",
    "MODEL_RETRY_EXHAUSTED", "MODEL_REPAIR_EXHAUSTED", "AUDIT_REFUSED",
    "RUN_CANCEL_REQUESTED", "RUN_CANCELLED", "USER_FEEDBACK"
  ]);
  const NOTICES = {
    RETRIEVAL_DEGRADED: "检索能力降级：结果可能不完整，请稍后重试或检查索引。",
    CIRCUIT_OPEN: "熔断器已打开：对应依赖暂时被隔离。",
    MODEL_REPAIR_EXHAUSTED: "模型结构化响应修复已耗尽，系统已安全拒绝本次结果。",
    WORKER_DLQ: "任务已进入死信队列，需要运维处理后重试。",
    LEASE_LOST: "任务租约已丢失，执行已停止以避免重复处理。",
    COMPONENT_DEGRADED: "部分组件已降级，回答可能受影响。",
    research_action_invalid: "研究动作不符合安全约束，系统未继续执行。",
    audit_failed: "回答未通过审计，系统不会展示未审计结果。",
    research_round_limit: "已达到全局研究轮次上限，系统停止继续研究。",
    cannot_answer: "现有证据不足以安全回答，系统未展示草稿。",
    refuse: "该请求已被安全拒绝，系统未继续生成回答。",
    clarify: "需要补充问题或上下文后才能继续检索。",
    subagent_unavailable: "子 Agent 当前不可用，系统将显示已降级的处理状态。",
    todo_creation_empty: "未生成可执行的 Todo，任务拆分没有继续。"
  };
  const EVENT_NOTICE_CODES = {
    AUDIT_REFUSED: "audit_failed",
    QUERY_REFUSED: "refuse"
  };
  const SAFE_DEGRADATION_COMPONENTS = new Set([
    "dense", "bm25", "reranker", "memory", "mem0", "llm", "elasticsearch",
    "redis", "artifact_store", "retrieval", "router", "generation", "audit",
    "citation", "outbox", "query_worker", "worker", "mysql", "checkpoint", "unknown"
  ]);
  const SAFE_DEGRADATION_REASONS = new Set([
    "lane_failure", "lane_timeout", "retrieval_unavailable", "reranker_unavailable",
    "memory_unavailable", "router_unavailable", "router_schema_invalid",
    "model_unavailable", "model_schema_invalid", "generation_unavailable", "audit_failed",
    "subagent_unavailable", "research_action_invalid",
    "authorization_unavailable", "provider_outage", "circuit_open", "outbox_retry",
    "lease_lost", "cancelled", "worker_timeout", "worker_dlq", "invalid_input", "unknown"
  ]);
  const SAFE_DEGRADATION_OUTCOMES = new Set(["degraded", "refused", "dlq"]);
  const SAFE_MODEL_TEXT = /^[A-Za-z0-9_.:-]{1,128}$/;
  const TERMINAL_NOTICE_CODES = new Set([
    "research_action_invalid", "research_round_limit", "audit_failed",
    "cannot_answer", "refuse", "clarify"
  ]);

  const elements = {};
  let activeRunId = null;
  let lastEventId = 0;
  let streamCancelled = false;
  let activeRoute = null;
  let answerSettled = false;
  const ROUTE_LABELS = {chat: "聊天", fast_rag: "快速检索", research: "深入研究"};
  const ROUTE_WAITING = {
    chat: "正在生成聊天回复…", fast_rag: "正在检索资料…", research: "正在深入研究…"
  };

  function safeRoute(route) {
    return typeof route === "string" && Object.hasOwn(ROUTE_LABELS, route) ? route : null;
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function initializeElements() {
    [
      "snapshot-id", "health-grid", "document-upload", "document-file",
      "ingestion-status", "memory-list", "reload-memories", "query-form",
      "query-input", "run-status", "cancel-run", "timeline", "answer",
      "evidence", "audit", "provenance", "degradation-banner"
    ].forEach((id) => { elements[id] = byId(id); });
  }

  function setText(element, value) {
    if (element) element.textContent = value;
  }

  function showNotice(code) {
    const message = NOTICES[code];
    if (!message || !elements["degradation-banner"]) return;
    elements["degradation-banner"].hidden = false;
    elements["degradation-banner"].textContent = message;
  }

  function clearNotice() {
    if (elements["degradation-banner"]) {
      elements["degradation-banner"].hidden = true;
      elements["degradation-banner"].textContent = "";
    }
  }

  function buildQueryPayload(question) {
    const query = String(question || "").trim();
    return query ? { query, wait_seconds: 0 } : null;
  }

  function buildSseHeaders(cursor) {
    const headers = { Accept: "text/event-stream" };
    if (Number.isInteger(cursor) && cursor > 0) {
      headers["Last-Event-ID"] = String(cursor);
    }
    return headers;
  }

  function noticeCodeForEvent(eventType) {
    if (EVENT_NOTICE_CODES[eventType]) return EVENT_NOTICE_CODES[eventType];
    return Object.prototype.hasOwnProperty.call(NOTICES, eventType) ? eventType : null;
  }

  function degradationAttributes(event) {
    const raw = event && typeof event.attributes === "object" && event.attributes
      ? event.attributes : {};
    const attributes = {};
    ["initial_route", "route"].forEach((key) => {
      if (safeRoute(raw[key])) attributes[key] = raw[key];
    });
    if (["conversation", "capability_unavailable", "clarify", "technical_error"].includes(raw.response_mode)) {
      attributes.response_mode = raw.response_mode;
    }
    if (["none", "missing_facts", "multi_step_required", "query_ambiguous", "external_realtime_required", "external_lookup_required", "irrelevant_results", "unknown"].includes(raw.gap_type)) {
      attributes.gap_type = raw.gap_type;
    }
    const nodes = new Set(["memory_loader", "route", "chat", "fast_rag", "record_fast_grade", "research_agent_loop", "evidence_builder", "evidence_grader", "generate", "faithfulness", "citation", "finalize"]);
    if (Array.isArray(raw.executed_path) && raw.executed_path.length <= 64 && raw.executed_path.every((node) => nodes.has(node))) {
      attributes.executed_path = raw.executed_path;
    }
    if (SAFE_DEGRADATION_COMPONENTS.has(raw.component)) attributes.component = raw.component;
    if (SAFE_DEGRADATION_REASONS.has(raw.reason)) attributes.reason = raw.reason;
    if (SAFE_DEGRADATION_OUTCOMES.has(raw.outcome)) attributes.outcome = raw.outcome;
    if (typeof raw.retryable === "boolean") attributes.retryable = raw.retryable;
    if (Number.isInteger(raw.attempt) && raw.attempt >= 0) attributes.attempt = raw.attempt;
    ["operation", "requested_model", "protocol", "error_class", "provider_request_id"].forEach((key) => {
      if (typeof raw[key] === "string" && SAFE_MODEL_TEXT.test(raw[key])) {
        attributes[key] = raw[key];
      }
    });
    if (Number.isInteger(raw.http_status) && raw.http_status >= 100 && raw.http_status <= 599) {
      attributes.http_status = raw.http_status;
    }
    if (typeof raw.client_timeout_seconds === "number"
      && Number.isFinite(raw.client_timeout_seconds)
      && raw.client_timeout_seconds >= 0) {
      attributes.client_timeout_seconds = raw.client_timeout_seconds;
    }
    return attributes;
  }

  function eventPresentation(event) {
    const known = PUBLIC_EVENT_TYPES.has(event.event_type);
    if (!known) return null;
    const attributes = known ? degradationAttributes(event) : {};
    const presentation = {
      label: known ? event.event_type : "进度更新",
      summary: known ? (event.summary || "执行中") : "进度更新",
      noticeCode: noticeCodeForEvent(event.event_type)
    };
    if (Object.keys(attributes).length) presentation.attributes = attributes;
    if (event.event_type === "QUERY_ROUTED" && safeRoute(event.route)) {
      presentation.label = "路由选择";
      presentation.summary = ROUTE_LABELS[event.route];
    }
    return presentation;
  }

  function terminalNoticeCode(run) {
    const answer = run && run.answer;
    const details = answer && typeof answer === "object" ? answer : {};
    const candidates = [details.status, run && run.error_code];
    return candidates.find((candidate) => (
      typeof candidate === "string" && TERMINAL_NOTICE_CODES.has(candidate)
    )) || null;
  }

  function memoryErrorPresentation(detail) {
    return {
      className: "error-card",
      message: `Mem0 不可用：${detail || "请检查 provider 状态。"}`
    };
  }

  function provenanceFor(details, run) {
    const provenance = {};
    if (Array.isArray(details.evidence_parent_ids)) {
      provenance.evidence_parent_ids = details.evidence_parent_ids.filter((value) => typeof value === "string");
    }
    if (["chat", "fast_rag", "research"].includes(details.route)) provenance.route = details.route;
    if (typeof details.runtime_config_snapshot_id === "string") {
      provenance.runtime_config_snapshot_id = details.runtime_config_snapshot_id;
    } else if (typeof run.runtime_config_snapshot_id === "string") {
      provenance.runtime_config_snapshot_id = run.runtime_config_snapshot_id;
    }
    if (["api", "fixture", "real_query_api", "real_query_graph"].includes(details.client_provenance)) {
      provenance.client_provenance = details.client_provenance;
    }
    return provenance;
  }

  function answerPresentation(run) {
    const details = run && run.answer && typeof run.answer === "object" ? run.answer : null;
    if (!details || !Array.isArray(details.segments)) return null;
    const isChat = details.route === "chat";
    if (isChat) {
      if (details.audited != null || details.citation_coverage != null
        || (details.evidence_parent_ids || []).length
        || details.segments.some(s => !s || s.kind !== "content" || (s.evidence_ids || []).length)) return null;
    } else if (details.audited !== true) return null;
    const segments = details.segments.filter((segment) => (
      segment && typeof segment === "object"
      && ["content", "heading", "separator", "references"].includes(segment.kind)
      && typeof segment.text === "string"
    ));
    if (!segments.length) return null;
    const evidenceIds = [];
    segments.forEach((segment) => {
      if (!Array.isArray(segment.evidence_ids)) return;
      segment.evidence_ids.forEach((value) => {
        if (typeof value === "string" && !evidenceIds.includes(value)) evidenceIds.push(value);
      });
    });
    const parentIds = Array.isArray(details.evidence_parent_ids)
      ? details.evidence_parent_ids.filter((value) => typeof value === "string") : [];
    const audit = isChat ? {} : { audited: true };
    if (typeof details.citation_coverage === "number" && details.citation_coverage >= 0 && details.citation_coverage <= 1) {
      audit.citation_coverage = details.citation_coverage;
    }
    return {
      text: segments.map((segment) => segment.text).join("\n"),
      evidence: { evidence_ids: evidenceIds, evidence_parent_ids: parentIds },
      audit,
      provenance: provenanceFor(details, run)
    };
  }

  function renderSafeTerminalNotice(code) {
    answerSettled = true;
    showNotice(code);
    setText(elements.answer, NOTICES[code]);
  }

  async function responseJson(response) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload.message || "服务请求失败，请稍后重试。");
      error.payload = payload;
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  function appendTimeline(event) {
    const presentation = eventPresentation(event);
    if (!presentation) return;
    if (safeRoute(event.route)) {
      activeRoute = event.route;
      if (!answerSettled) setText(elements.answer, ROUTE_WAITING[activeRoute]);
    }
    const item = document.createElement("li");
    const detail = presentation.attributes
      ? `（${Object.entries(presentation.attributes).map(([key, value]) => `${key}=${value}`).join(", ")}）`
      : "";
    item.textContent = `${presentation.label}：${presentation.summary}${detail}`;
    const time = document.createElement("time");
    time.dateTime = event.created_at || "";
    time.textContent = event.created_at || "";
    item.appendChild(time);
    elements.timeline.appendChild(item);
    if (presentation.noticeCode) {
      if (TERMINAL_NOTICE_CODES.has(presentation.noticeCode)) {
        renderSafeTerminalNotice(presentation.noticeCode);
      } else {
        showNotice(presentation.noticeCode);
      }
    }
  }

  function renderObject(target, value, fallback) {
    if (value === undefined || value === null || value === "") {
      setText(target, fallback);
      return;
    }
    setText(target, typeof value === "string" ? value : JSON.stringify(value, null, 2));
  }

  function displayAnswer(run) {
    const answer = run.answer;
    const termination = terminalNoticeCode(run);
    const presentation = answerPresentation(run);
    activeRoute = safeRoute(answer && answer.route) || activeRoute;
    answerSettled = !!presentation || !!termination || ["completed", "failed", "cancelled"].includes(run.status);
    if (termination) {
      renderSafeTerminalNotice(termination);
    } else if (presentation) {
      renderObject(elements.answer, presentation.text, "未返回可展示的回答。");
    } else if (run.error_code) {
      setText(elements.answer, "任务未完成，请查看执行状态。");
    } else if (answerSettled) {
      setText(elements.answer, run.status === "cancelled" ? "任务已取消。" : "任务已结束，暂无可展示的回答。");
    } else {
      setText(elements.answer, ROUTE_WAITING[activeRoute] || "正在处理消息…");
    }
    const isChat = presentation && presentation.provenance.route === "chat";
    renderObject(elements.evidence, isChat ? "不适用：聊天回复不引用文档证据。" : presentation && presentation.evidence, "服务端响应中暂无可展示的证据。");
    renderObject(elements.audit, isChat ? "不适用：聊天回复不进行文档证据审计。" : presentation && presentation.audit, "服务端响应中暂无审计信息。");
    renderObject(elements.provenance, presentation && presentation.provenance, "服务端响应中暂无溯源信息。");
  }

  async function loadRuntimeSummary() {
    try {
      const summary = await responseJson(await fetch("/v1/runtime/summary"));
      setText(elements["snapshot-id"], summary.runtime_config_snapshot_id);
      elements["health-grid"].replaceChildren();
      Object.entries(summary.dependencies).forEach(([name, state]) => {
        const chip = document.createElement("div");
        chip.className = `health-chip ${state}`;
        chip.textContent = `${name}: ${state === "available" ? "可用" : "不可用"}`;
        elements["health-grid"].appendChild(chip);
      });
      if (!summary.memory_available) showNotice("COMPONENT_DEGRADED");
      return summary;
    } catch (error) {
      setText(elements["snapshot-id"], "运行时摘要不可用");
      renderError(elements["health-grid"], error);
      return null;
    }
  }

  async function loadHealth() {
    try {
      const [live, ready] = await Promise.all([
        responseJson(await fetch("/health/live")),
        responseJson(await fetch("/health/ready"))
      ]);
      setText(elements["run-status"], ready.status === "ready" ? "服务就绪" : "服务尚未就绪");
      return { live, ready };
    } catch (error) {
      setText(elements["run-status"], "健康检查不可用");
      return null;
    }
  }

  async function submitQuery(question) {
    const payload = buildQueryPayload(question);
    if (!payload) {
      setText(elements["run-status"], "请输入问题后再提交。");
      return null;
    }
    clearNotice();
    activeRunId = null;
    activeRoute = null;
    answerSettled = false;
    elements.timeline.replaceChildren();
    displayAnswer({status: "queued"});
    lastEventId = 0;
    streamCancelled = false;
    try {
      const run = await responseJson(await fetch("/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }));
      activeRunId = run.run_id;
      setText(elements["run-status"], `任务 ${run.status}`);
      elements["cancel-run"].disabled = false;
      displayAnswer(run);
      if (run.status === "completed" || run.status === "failed" || run.status === "cancelled") {
        elements["cancel-run"].disabled = true;
      }
      // Replaying the scoped event stream after a synchronous completion is
      // required to surface safe audit-refusal/degradation notices.
      void streamRun(run.run_id);
      return run;
    } catch (error) {
      setText(elements["run-status"], "创建任务失败");
      renderError(elements.answer, error);
      return null;
    }
  }

  function parseSseBlock(block) {
    const values = { id: null, data: "" };
    block.split("\n").forEach((line) => {
      if (line.startsWith("id:")) values.id = line.slice(3).trim();
      if (line.startsWith("data:")) values.data += line.slice(5).trim();
    });
    if (!values.data) return null;
    try { return { id: values.id, payload: JSON.parse(values.data) }; } catch { return null; }
  }

  async function streamRun(runId) {
    let retries = 0;
    while (!streamCancelled && activeRunId === runId && retries < 3) {
      try {
        const headers = buildSseHeaders(lastEventId);
        const response = await fetch(`/v1/query-runs/${encodeURIComponent(runId)}/events`, { headers });
        if (!response.ok || !response.body) throw new Error("无法连接任务事件流。");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (streamCancelled || activeRunId !== runId) {
            await reader.cancel();
            return null;
          }
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";
          blocks.forEach((block) => {
            const event = parseSseBlock(block);
            if (!event) return;
            const numericId = Number(event.id);
            if (Number.isInteger(numericId) && numericId > 0) lastEventId = numericId;
            appendTimeline(event.payload);
          });
        }
        const run = await loadRun(runId);
        if (run && ["completed", "failed", "cancelled"].includes(run.status)) return run;
        retries += 1;
      } catch (error) {
        retries += 1;
        setText(elements["run-status"], `事件流重连中（${retries}/3）`);
        await new Promise((resolve) => window.setTimeout(resolve, 500 * retries));
      }
    }
    return loadRun(runId);
  }

  async function loadRun(runId) {
    try {
      const run = await responseJson(await fetch(`/v1/query-runs/${encodeURIComponent(runId)}`));
      setText(elements["run-status"], `任务 ${run.status}`);
      displayAnswer(run);
      if (["completed", "failed", "cancelled"].includes(run.status)) {
        elements["cancel-run"].disabled = true;
      }
      return run;
    } catch (error) {
      setText(elements["run-status"], "读取任务状态失败");
      renderError(elements.answer, error);
      return null;
    }
  }

  async function cancelActiveRun() {
    if (!activeRunId) return;
    streamCancelled = true;
    try {
      const run = await responseJson(await fetch(`/v1/query-runs/${encodeURIComponent(activeRunId)}/cancel`, { method: "POST" }));
      displayAnswer(run);
      setText(elements["run-status"], `任务 ${run.status}`);
    } catch (error) {
      renderError(elements.answer, error);
    }
  }

  async function uploadDocument(file) {
    if (!file) return null;
    const body = new FormData();
    body.append("file", file);
    setText(elements["ingestion-status"], "正在上传文档…");
    try {
      const job = await responseJson(await fetch("/v1/documents", { method: "POST", body }));
      setText(elements["ingestion-status"], `入库任务 ${job.job_id}：${job.status}`);
      if (!["completed", "failed", "cancelled"].includes(job.status)) void pollIngestionJob(job.job_id);
      return job;
    } catch (error) {
      renderError(elements["ingestion-status"], error);
      return null;
    }
  }

  async function pollIngestionJob(jobId) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
      try {
        const job = await responseJson(await fetch(`/v1/ingestion-jobs/${encodeURIComponent(jobId)}`));
        setText(elements["ingestion-status"], `入库任务 ${job.job_id}：${job.status}`);
        if (["completed", "failed", "cancelled"].includes(job.status)) return job;
      } catch (error) {
        renderError(elements["ingestion-status"], error);
        return null;
      }
    }
    setText(elements["ingestion-status"], "入库任务仍在执行，请稍后刷新状态。");
    return null;
  }

  function renderError(target, error) {
    const message = error && error.message ? error.message : "服务暂时不可用。";
    setText(target, message);
    if (target && target.classList) target.classList.add("error-card");
    if (error && error.payload && error.payload.error_code) showNotice(error.payload.error_code);
  }

  async function loadMemories() {
    elements["memory-list"].replaceChildren();
    try {
      const payload = await responseJson(await fetch("/v1/memories"));
      if (!payload.memories.length) {
        const item = document.createElement("li");
        item.textContent = "暂无已保存的 Mem0 记忆。";
        elements["memory-list"].appendChild(item);
        return payload.memories;
      }
      payload.memories.forEach((memory) => {
        const item = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = memory.text || memory.id;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary";
        remove.textContent = "删除";
        remove.addEventListener("click", () => { void deleteMemory(memory.id); });
        item.append(label, remove);
        elements["memory-list"].appendChild(item);
      });
      return payload.memories;
    } catch (error) {
      const item = document.createElement("li");
      const presentation = memoryErrorPresentation(error.message);
      item.className = presentation.className;
      item.textContent = presentation.message;
      elements["memory-list"].appendChild(item);
      showNotice("COMPONENT_DEGRADED");
      return null;
    }
  }

  async function deleteMemory(memoryId) {
    try {
      const response = await fetch(`/v1/memories/${encodeURIComponent(memoryId)}`, { method: "DELETE" });
      if (!response.ok) await responseJson(response);
      return loadMemories();
    } catch (error) {
      const item = document.createElement("li");
      item.className = "error-card";
      item.textContent = `删除记忆失败：${error.message || "服务暂时不可用。"}`;
      elements["memory-list"].prepend(item);
      return null;
    }
  }

  function bindEvents() {
    elements["query-form"].addEventListener("submit", (event) => {
      event.preventDefault();
      void submitQuery(elements["query-input"].value);
    });
    elements["document-upload"].addEventListener("submit", (event) => {
      event.preventDefault();
      void uploadDocument(elements["document-file"].files[0]);
    });
    elements["reload-memories"].addEventListener("click", () => { void loadMemories(); });
    elements["cancel-run"].addEventListener("click", () => { void cancelActiveRun(); });
  }

  function initialize() {
    initializeElements();
    bindEvents();
    void loadRuntimeSummary();
    void loadHealth();
    void loadMemories();
  }

  Object.assign(window, {
    loadRuntimeSummary, submitQuery, streamRun, loadRun, uploadDocument, loadMemories, deleteMemory,
    AgenticRagConsole: {
      contract: {
        buildQueryPayload, buildSseHeaders, eventPresentation, noticeCodeForEvent,
        terminalNoticeCode, memoryErrorPresentation, provenanceFor, degradationAttributes,
        answerPresentation
      }
    }
  });
  document.addEventListener("DOMContentLoaded", initialize);
})();
