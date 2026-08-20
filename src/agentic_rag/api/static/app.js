(() => {
  "use strict";

  const PUBLIC_EVENT_TYPES = new Set([
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
    "subagents unavailable": "子 Agent 当前不可用，系统将显示已降级的处理状态。",
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
    "authorization_unavailable", "provider_outage", "circuit_open", "outbox_retry",
    "lease_lost", "cancelled", "worker_timeout", "worker_dlq", "invalid_input", "unknown"
  ]);
  const SAFE_DEGRADATION_OUTCOMES = new Set(["degraded", "refused", "dlq"]);
  const TERMINAL_NOTICE_CODES = new Set([
    "research_action_invalid", "research_round_limit", "audit_failed",
    "cannot_answer", "refuse", "clarify"
  ]);

  const elements = {};
  let activeRunId = null;
  let lastEventId = 0;
  let streamCancelled = false;

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
    return query ? { query, wait_seconds: 30 } : null;
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
    if (SAFE_DEGRADATION_COMPONENTS.has(raw.component)) attributes.component = raw.component;
    if (SAFE_DEGRADATION_REASONS.has(raw.reason)) attributes.reason = raw.reason;
    if (SAFE_DEGRADATION_OUTCOMES.has(raw.outcome)) attributes.outcome = raw.outcome;
    if (typeof raw.retryable === "boolean") attributes.retryable = raw.retryable;
    if (Number.isInteger(raw.attempt) && raw.attempt >= 0) attributes.attempt = raw.attempt;
    return attributes;
  }

  function eventPresentation(event) {
    const known = PUBLIC_EVENT_TYPES.has(event.event_type);
    const attributes = known ? degradationAttributes(event) : {};
    const presentation = {
      label: known ? event.event_type : "进度更新",
      summary: known ? (event.summary || "执行中") : "进度更新",
      noticeCode: noticeCodeForEvent(event.event_type)
    };
    if (Object.keys(attributes).length) presentation.attributes = attributes;
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
    if (details.evidence_parent_ids !== undefined) {
      provenance.evidence_parent_ids = details.evidence_parent_ids;
    }
    if (details.parent_ids !== undefined) provenance.parent_ids = details.parent_ids;
    if (details.parentIds !== undefined) provenance.parent_ids = details.parentIds;
    if (details.route !== undefined) provenance.route = details.route;
    if (details.runtime_config_snapshot_id !== undefined) {
      provenance.runtime_config_snapshot_id = details.runtime_config_snapshot_id;
    } else if (run.runtime_config_snapshot_id !== undefined) {
      provenance.runtime_config_snapshot_id = run.runtime_config_snapshot_id;
    }
    if (details.client_provenance !== undefined) {
      provenance.client_provenance = details.client_provenance;
    }
    return provenance;
  }

  function renderSafeTerminalNotice(code) {
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
    const details = answer && typeof answer === "object" ? answer : run;
    const termination = terminalNoticeCode(run);
    if (termination) {
      renderSafeTerminalNotice(termination);
    } else if (answer !== null && answer !== undefined) {
      renderObject(elements.answer, answer, "未返回可展示的回答。");
    } else if (run.error_code) {
      setText(elements.answer, "任务未完成，请查看执行状态。");
    }
    renderObject(
      elements.evidence,
      details.evidence || details.citations || details.evidence_ids,
      "服务端响应中暂无可展示的证据。"
    );
    const audit = {};
    if (typeof details.audited === "boolean") audit.audited = details.audited;
    if (details.citation_coverage !== undefined) audit.citation_coverage = details.citation_coverage;
    if (details.audit !== undefined) audit.audit = details.audit;
    renderObject(elements.audit, Object.keys(audit).length ? audit : null, "服务端响应中暂无审计信息。");
    const provenance = provenanceFor(details, run);
    renderObject(elements.provenance, Object.keys(provenance).length ? provenance : null, "服务端响应中暂无溯源信息。");
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
    elements.timeline.replaceChildren();
    setText(elements.answer, "正在创建检索任务…");
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
        terminalNoticeCode, memoryErrorPresentation, provenanceFor, degradationAttributes
      }
    }
  });
  document.addEventListener("DOMContentLoaded", initialize);
})();
