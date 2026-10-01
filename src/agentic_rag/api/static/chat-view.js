(function (root) {
  "use strict";
  const notices = {
    research_action_invalid: "研究动作不符合安全约束，系统未继续执行。",
    audit_failed: "回答未通过审核，系统不会展示未审核结果。",
    research_round_limit: "已达到研究轮次上限，系统停止继续研究。",
    cannot_answer: "现有证据不足以安全回答，系统未展示草稿。",
    refuse: "该请求已被安全拒绝，系统未继续生成回答。",
    clarify: "需要补充问题或上下文后才能继续回答。",
    failed: "本次处理未完成，请稍后重试。",
    cancelled: "已停止本次回答。",
    completed: "本次处理已结束，暂无可展示的回答。",
  };
  const phaseLabels = {
    processing: "处理中",
    retrieving: "检索中",
    researching: "研究中",
    auditing: "审核中",
  };
  function answerPresentation(run) {
    if (!run || run.status !== "completed" || run.error_code) return null;
    const answer = run.answer;
    if (!answer || !Array.isArray(answer.segments) || !answer.segments.length)
      return null;
    const chat = answer.route === "chat";
    if (
      answer.status &&
      !(chat && ["clarify", "cannot_answer"].includes(answer.status))
    )
      return null;
    if (chat) {
      if (
        answer.audited != null ||
        answer.citation_coverage != null ||
        (answer.evidence_parent_ids || []).length ||
        answer.segments.some(
          (s) => !s || s.kind !== "content" || (s.evidence_ids || []).length,
        )
      )
        return null;
    } else if (answer.audited !== true) return null;
    if (
      answer.segments.some(
        (s) =>
          !s ||
          !["content", "heading", "separator", "references"].includes(s.kind) ||
          typeof s.text !== "string",
      )
    )
      return null;
    return {
      text: answer.segments.map((s) => s.text).join("\n\n"),
      hasSources:
        !chat && answer.segments.some((s) => (s.evidence_ids || []).length > 0),
    };
  }
  function turnText(run) {
    if (run.status === "queued") return "等待中";
    if (run.status === "cancel_requested") return "正在停止";
    if (run.status === "running") return phaseLabels[run.phase] || "处理中";
    if (run.status === "submitting") return "提交中";
    if (run.status === "unknown") return "提交结果尚未确认，请检查提交状态。";
    const presentation = answerPresentation(run);
    if (presentation) return presentation.text;
    const code =
      run.error_code || run.terminal_code || run.answer?.status || run.status;
    return notices[code] || notices[run.status] || notices.completed;
  }
  function createChatView(document) {
    const byId = (id) => document.getElementById(id),
      list = byId("chat-messages"),
      scroll = byId("chat-scroll");
    const records = new Map();
    function node(tag, className, text) {
      const e = document.createElement(tag);
      e.className = className || "";
      if (text !== undefined) e.textContent = text;
      return e;
    }
    const nearBottom = () =>
      scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight <= 80;
    const scrollToLatest = () => {
      scroll.scrollTop = scroll.scrollHeight;
      byId("scroll-latest").hidden = true;
    };
    scroll.addEventListener("scroll", () => {
      byId("scroll-latest").hidden = nearBottom();
    });
    byId("scroll-latest").addEventListener("click", scrollToLatest);
    function makeTurn(turn) {
      const user = node("article", "message message-user");
      user.setAttribute("data-run-id", turn.run_id);
      user.setAttribute("data-role", "user");
      const question = node("div", "message-body", turn.question || "");
      user.append(node("div", "message-role", "你"), question);
      const assistant = node("article", "message message-assistant");
      assistant.setAttribute("data-run-id", turn.run_id);
      assistant.setAttribute("data-role", "assistant");
      const body = node("div", "message-body"),
        button = node("button", "source-button", "查看来源"),
        sources = node("div", "sources");
      button.type = "button";
      button.setAttribute("data-action", "sources");
      button.setAttribute("data-run-id", turn.run_id);
      button.setAttribute("aria-expanded", "false");
      sources.id = `sources-${turn.run_id}`;
      sources.hidden = true;
      button.setAttribute("aria-controls", sources.id);
      assistant.append(
        node("div", "message-role", "Agentic RAG"),
        body,
        button,
        sources,
      );
      const record = { user, assistant, body, question, button, sources };
      records.set(turn.run_id, record);
      return record;
    }
    function updateTurn(turn, follow = true) {
      const shouldFollow = follow && nearBottom();
      let record = records.get(turn.run_id);
      if (!record) {
        record = makeTurn(turn);
        list.append(record.user, record.assistant);
      }
      record.question.textContent =
        turn.question || record.question.textContent;
      record.body.textContent = turnText(turn);
      record.assistant.classList.toggle(
        "message-pending",
        [
          "queued",
          "running",
          "cancel_requested",
          "submitting",
          "unknown",
        ].includes(turn.status),
      );
      record.button.hidden = !answerPresentation(turn)?.hasSources;
      byId("chat-empty").hidden = list.children.length > 0;
      if (shouldFollow) scrollToLatest();
    }
    function clearPending() {
      const pending = records.get("pending");
      if (pending) {
        pending.user.remove();
        pending.assistant.remove();
        records.delete("pending");
      }
    }
    function renderPending(pending) {
      clearPending();
      if (pending?.question)
        updateTurn({
          run_id: "pending",
          question: pending.question,
          status: pending.status,
        });
    }
    function renderSession(session) {
      records.clear();
      list.replaceChildren();
      for (const id of session.orderedRunIds)
        updateTurn(session.turns.get(id), false);
      renderPending(session.pendingSubmission);
      byId("chat-empty").hidden = list.children.length > 0;
      byId("older-turns").hidden = !session.historyCursor;
    }
    function prependTurns(turns) {
      const before = scroll.scrollHeight,
        top = scroll.scrollTop,
        nodes = [];
      for (const turn of turns) {
        if (records.has(turn.run_id)) continue;
        const r = makeTurn(turn);
        updateTurn(turn, false);
        nodes.push(r.user, r.assistant);
      }
      list.prepend(...nodes);
      scroll.scrollTop = top + (scroll.scrollHeight - before);
    }
    function renderSessionList(items, selectedId) {
      const target = byId("session-list"),
        nodes = [];
      for (const item of items) {
        const button = node("button", "session-item");
        button.setAttribute("data-session-id", item.session_id);
        button.setAttribute(
          "aria-current",
          String(item.session_id === selectedId),
        );
        button.append(node("span", "session-name", item.title));
        if (item.active_run_id) {
          const dot = node("span", "activity-dot");
          dot.setAttribute("aria-label", "进行中");
          button.append(dot);
        }
        nodes.push(button);
      }
      target.replaceChildren(...nodes);
    }
    function setSourcesOpen(id, open) {
      const r = records.get(id);
      if (!r) return;
      r.sources.hidden = !open;
      r.button.setAttribute("aria-expanded", String(open));
      r.button.textContent = open ? "收起来源" : "查看来源";
      if (!open) r.sources.replaceChildren();
    }
    function renderSources(id, sourceView) {
      const r = records.get(id);
      if (!r) return;
      r.sources.replaceChildren();
      if (sourceView.status === "loading") {
        r.sources.append(node("p", "muted", "正在核验来源…"));
        return;
      }
      if (!sourceView.items?.length) {
        r.sources.append(
          node(
            "p",
            "muted",
            sourceView.status === "none"
              ? "本轮无需引用文档。"
              : "来源暂不可用，文档可能已删除或无权访问。",
          ),
        );
        return;
      }
      for (const source of sourceView.items) {
        const card = node("section", "source-card");
        card.append(node("h3", "source-title", source.filename || "文档"));
        const meta = [...(source.heading_path || [])];
        if (source.page_from)
          meta.push(
            `第 ${source.page_from}${source.page_to !== source.page_from ? `–${source.page_to}` : ""} 页`,
          );
        if (source.version_status === "historical") meta.push("历史版本");
        card.append(
          node("div", "source-meta", meta.join(" · ")),
          node("p", "source-excerpt", source.excerpt || "片段因长度限制省略。"),
        );
        if (source.truncated || source.heading_truncated)
          card.append(node("span", "source-note", "内容已截取"));
        r.sources.append(card);
      }
      if (sourceView.omitted_source_count)
        r.sources.append(
          node(
            "p",
            "source-note",
            `另有 ${sourceView.omitted_source_count} 条来源省略或不可访问。`,
          ),
        );
    }
    function setComposer({ draft, canSend, canStop }) {
      const input = byId("query-input");
      if (input.value !== draft) input.value = draft;
      const length = Array.from(draft.trim()).length;
      byId("send-button").disabled = !canSend || !length || length > 32000;
      byId("cancel-button").hidden = !canStop;
      byId("composer-status").textContent =
        length > 32000
          ? "消息不能超过 32,000 个字符"
          : "Enter 发送 · Shift + Enter 换行";
    }
    return {
      renderSessionList,
      renderSession,
      updateTurn,
      prependTurns,
      renderSources,
      setSourcesOpen,
      setComposer,
      scrollToLatest,
      renderPending,
      clearPending,
      notice(text) {
        byId("chat-notice").textContent = text || "";
        byId("chat-notice").hidden = !text;
      },
    };
  }
  const api = { createChatView, answerPresentation, turnText };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AgenticRagChatView = api;
})(typeof window !== "undefined" ? window : globalThis);
