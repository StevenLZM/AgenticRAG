/* Persistent chat controller: every asynchronous view write checks its generation. */
(function (root) {
  "use strict";
  const common = typeof module !== "undefined" && module.exports;
  const S = common ? require("./chat-state.js") : root.AgenticRagChatState;
  const A = common ? require("./chat-api.js") : root.AgenticRagChatApi;
  const V = common ? require("./chat-view.js") : root.AgenticRagChatView;
  const T = common
    ? require("./console-tools.js")
    : root.AgenticRagConsoleTools;
  function createChatController({
    document,
    window = root,
    api = A.createChatApi(),
    storage,
    tools,
    uuid = () => globalThis.crypto.randomUUID(),
  } = {}) {
    const state = S.createChatState(),
      view = V.createChatView(document),
      byId = (id) => document.getElementById(id);
    if (storage === undefined) {
      try {
        storage = window.localStorage;
      } catch (_) {
        storage = null;
      }
    }
    let token = null,
      viewAbort = null,
      observer = null,
      listGeneration = 0,
      sessionCursor = null,
      creating = false,
      composing = false,
      disposed = false,
      sourceGeneration = 0,
      sourceCounter = 0;
    let summaries = new Map();
    const openSources = new Map();
    const current = () =>
      state.selectedSessionId
        ? S.sessionState(state, state.selectedSessionId)
        : null;
    const valid = (t) => !disposed && S.isCurrent(state, t);
    function save() {
      S.saveResumeMetadata(storage, state);
      if (state.storageWarning)
        view.notice(
          "浏览器无法保存恢复标识。已受理的对话仍保存在服务器，可从会话列表找回。",
        );
    }
    function updateComposer() {
      const session = current();
      byId("query-input").disabled = !session;
      view.setComposer({
        draft: session?.draft || "",
        canSend:
          !!session &&
          session.loaded &&
          !session.activeRunId &&
          !session.pendingSubmission,
        canStop: !!session?.activeRunId,
      });
      const pending = session?.pendingSubmission;
      byId("pending-actions").hidden = pending?.status !== "unknown";
      byId("retry-submission").hidden = !pending?.question;
      byId("rename-chat").disabled = !session;
      byId("delete-chat").disabled = !session;
    }
    function renderList() {
      view.renderSessionList([...summaries.values()], state.selectedSessionId);
      byId("more-sessions").hidden = !sessionCursor;
      byId("session-title").textContent =
        summaries.get(state.selectedSessionId)?.title ||
        current()?.metadata?.title ||
        "新对话";
    }
    async function refreshSessions(append = false) {
      const stamp = ++listGeneration;
      try {
        const page = await api.listSessions({
          cursor: append ? sessionCursor : null,
        });
        if (disposed || stamp !== listGeneration) return;
        if (!append) summaries = new Map();
        for (const item of page.items) summaries.set(item.session_id, item);
        sessionCursor = page.next_cursor;
        renderList();
      } catch (_) {
        if (!disposed && stamp === listGeneration)
          view.notice("会话列表暂不可用，请稍后重试。");
      }
    }
    function stopObservation() {
      observer?.abort();
      observer = null;
    }
    function clearSources() {
      sourceGeneration++;
      for (const id of openSources.keys()) view.setSourcesOpen(id, false);
      openSources.clear();
    }
    async function authorizeSource(runId, t, stamp) {
      const generation = sourceGeneration;
      view.renderSources(runId, { status: "loading" });
      try {
        const result = await api.getSources(
          t.sessionId,
          runId,
          viewAbort?.signal,
        );
        if (
          valid(t) &&
          !document.hidden &&
          generation === sourceGeneration &&
          openSources.get(runId) === stamp
        )
          view.renderSources(runId, result);
      } catch (_) {
        if (
          valid(t) &&
          generation === sourceGeneration &&
          openSources.get(runId) === stamp
        )
          view.renderSources(runId, { status: "unavailable" });
      }
    }
    function toggleSource(runId) {
      if (!current()) return;
      if (openSources.has(runId)) {
        openSources.delete(runId);
        view.setSourcesOpen(runId, false);
        return;
      }
      const stamp = ++sourceCounter;
      openSources.set(runId, stamp);
      view.setSourcesOpen(runId, true);
      void authorizeSource(runId, token, stamp);
    }
    function documentsChanged() {
      sourceGeneration++;
      for (const id of openSources.keys()) {
        const stamp = ++sourceCounter;
        openSources.set(id, stamp);
        view.renderSources(id, { status: "loading" });
        if (!document.hidden) void authorizeSource(id, token, stamp);
      }
    }
    const toolView =
      tools ||
      T.createConsoleTools({ document, onDocumentsChanged: documentsChanged });
    async function refreshLatest(t) {
      try {
        const page = await api.listTurns(t.sessionId, {
          signal: viewAbort?.signal,
        });
        if (!valid(t)) return;
        for (const run of page.items) {
          if (S.applyRun(state, t, run))
            view.updateTurn(current().turns.get(run.run_id));
        }
        updateComposer();
        save();
      } catch (_) {
        if (valid(t)) view.notice("最新回答暂未同步，可重新打开会话继续查看。");
      }
    }
    function observe(runId, t) {
      stopObservation();
      if (!runId || document.hidden || !valid(t)) return;
      const control = new AbortController();
      observer = control;
      const session = current();
      void api
        .watchRun({
          runId,
          cursor: session.turns.get(runId)?.eventCursor || session.eventCursor,
          signal: control.signal,
          onPhase(event) {
            if (
              valid(t) &&
              S.applyPhase(state, t, event.run_id, event.id, event.phase)
            ) {
              view.updateTurn(current().turns.get(event.run_id));
              save();
            }
          },
          async onRun(run) {
            if (control.signal.aborted || !valid(t)) return;
            if (S.applyRun(state, t, run)) {
              view.updateTurn(current().turns.get(run.run_id));
              updateComposer();
              save();
            }
            if (S.isTerminal(run)) {
              await refreshLatest(t);
              void refreshSessions();
            }
          },
          onConnection(status) {
            if (valid(t) && status === "polling")
              view.notice("连接暂时不稳定，正在定时检查任务状态。");
          },
        })
        .catch((error) => {
          if (!valid(t) || control.signal.aborted) return;
          if ([404, 410].includes(error.status)) {
            view.notice("当前会话已不可用，请从列表选择其他对话。");
            current().activeRunId = null;
            void refreshSessions();
          } else view.notice("暂时无法获取任务状态，可重新打开会话继续查看。");
          updateComposer();
        });
    }
    async function checkSubmission(t = token) {
      if (!valid(t)) return;
      const session = current(),
        pending = session.pendingSubmission;
      if (!pending) return;
      try {
        const turn = await api.findSubmission(
          t.sessionId,
          pending.requestId,
          viewAbort?.signal,
        );
        S.acknowledgeSubmission(state, t.sessionId, pending.requestId);
        if (!valid(t)) return;
        view.clearPending();
        S.applyRun(state, t, turn);
        view.updateTurn(current().turns.get(turn.run_id));
        view.notice("");
        updateComposer();
        save();
        if (!S.isTerminal(turn)) observe(turn.run_id, t);
      } catch (error) {
        if (!valid(t)) return;
        S.markSubmissionUnknown(state, t.sessionId, pending.requestId);
        view.notice(
          error.status === 404
            ? "尚未确认这次提交是否受理，请继续检查状态；系统不会重复发问。"
            : "暂时无法核验提交状态，请稍后再检查。",
        );
        updateComposer();
        save();
      }
    }
    function closeSidebar() {
      byId("session-sidebar").classList.remove("is-open");
      byId("sidebar-backdrop").hidden = true;
      byId("session-sidebar").removeAttribute("aria-modal");
      byId("session-sidebar").removeAttribute("role");
    }
    async function selectSession(id) {
      stopObservation();
      viewAbort?.abort();
      viewAbort = new AbortController();
      clearSources();
      token = S.activateSession(state, id);
      const t = token;
      current().loaded = false;
      closeSidebar();
      view.notice("");
      view.renderSession(current());
      renderList();
      updateComposer();
      save();
      try {
        const [summary, page] = await Promise.all([
          api.getSession(id, viewAbort.signal),
          api.listTurns(id, { signal: viewAbort.signal }),
        ]);
        if (!valid(t)) return;
        summaries.set(id, summary);
        const session = current();
        session.metadata = summary;
        session.activeRunId = summary.active_run_id;
        for (const run of page.items) S.applyRun(state, t, run);
        session.historyCursor = page.next_cursor;
        session.loaded = true;
        if (session.pendingSubmission) {
          const accepted = page.items.find(
            (run) =>
              run.client_request_id === session.pendingSubmission.requestId,
          );
          if (accepted)
            S.acknowledgeSubmission(
              state,
              id,
              session.pendingSubmission.requestId,
            );
        }
        view.renderSession(session);
        view.scrollToLatest();
        renderList();
        updateComposer();
        save();
        if (session.pendingSubmission) await checkSubmission(t);
        if (valid(t) && current().activeRunId)
          observe(current().activeRunId, t);
      } catch (error) {
        if (!valid(t)) return;
        view.notice(
          [404, 410].includes(error.status)
            ? "此对话已删除或不可访问，请选择其他对话。"
            : "对话加载失败，请重新选择以重试。",
        );
        if ([404, 410].includes(error.status)) {
          state.sessions.delete(id);
          summaries.delete(id);
          token = S.activateSession(state, null);
          view.renderSession({
            turns: new Map(),
            orderedRunIds: [],
            historyCursor: null,
          });
          renderList();
        }
        updateComposer();
      }
    }
    async function newChat() {
      if (creating) return;
      creating = true;
      byId("new-chat").disabled = true;
      stopObservation();
      viewAbort?.abort();
      clearSources();
      token = S.activateSession(state, null);
      const generation = token.generation;
      state.creationRequestId = state.creationRequestId || uuid();
      const key = state.creationRequestId;
      save();
      updateComposer();
      try {
        const summary = await api.createSession(key);
        if (state.creationRequestId === key) state.creationRequestId = null;
        summaries.set(summary.session_id, summary);
        save();
        if (!disposed && state.viewGeneration === generation)
          await selectSession(summary.session_id);
        void refreshSessions();
      } catch (_) {
        if (!disposed)
          view.notice(
            "新建对话尚未确认。再次点击新建会使用同一个请求标识继续确认。",
          );
      } finally {
        creating = false;
        byId("new-chat").disabled = false;
      }
    }
    async function send(retry = false) {
      const session = current(),
        t = token;
      if (!session || !session.loaded || session.activeRunId) return;
      const existing = session.pendingSubmission;
      if (existing && (!retry || !existing.question)) return;
      const question = (existing?.question || session.draft).trim();
      if (!question || Array.from(question).length > 32000) return;
      const pending = S.beginSubmission(state, t.sessionId, question, uuid());
      pending.status = "submitting";
      view.notice("");
      view.renderPending(pending);
      updateComposer();
      save();
      try {
        const turn = await api.submitTurn(
          t.sessionId,
          pending.question,
          pending.requestId,
        );
        S.acknowledgeSubmission(state, t.sessionId, pending.requestId);
        save();
        if (!valid(t)) return;
        view.clearPending();
        S.applyRun(state, t, turn);
        view.updateTurn(current().turns.get(turn.run_id));
        updateComposer();
        void refreshSessions();
        if (!S.isTerminal(turn)) observe(turn.run_id, t);
      } catch (error) {
        if ([400, 401, 403, 404, 409, 410, 422].includes(error.status)) {
          S.rejectSubmission(state, t.sessionId, pending.requestId);
          if (valid(t)) {
            view.clearPending();
            view.notice(
              error.errorCode === "SESSION_BUSY"
                ? "这个对话正在处理另一条问题，你的草稿已保留。"
                : error.errorCode === "IDEMPOTENCY_CONFLICT"
                  ? "提交标识冲突，原问题未被覆盖。"
                  : "本次提交未受理，请检查输入或重新打开对话。",
            );
            if (error.errorCode === "SESSION_BUSY") {
              const match = error.location?.match(
                /^\/v1\/query-runs\/([A-Za-z0-9_-]+)$/,
              );
              if (match) {
                current().activeRunId = match[1];
                observe(match[1], t);
              }
            }
          }
        } else {
          S.markSubmissionUnknown(state, t.sessionId, pending.requestId);
          if (valid(t)) {
            view.renderPending(current().pendingSubmission);
            view.notice("提交结果尚未确认，请检查状态，或使用原请求重试。");
          }
        }
        if (valid(t)) updateComposer();
        save();
      }
    }
    async function stop() {
      const t = token,
        id = current()?.activeRunId;
      if (!id) return;
      byId("cancel-button").disabled = true;
      try {
        const run = await api.cancelRun(id);
        if (valid(t)) {
          S.applyRun(state, t, run);
          view.updateTurn(current().turns.get(id));
          updateComposer();
          if (S.isTerminal(run)) {
            stopObservation();
            await refreshLatest(t);
            void refreshSessions();
          }
        }
      } catch (_) {
        if (valid(t)) view.notice("停止请求暂未确认，任务状态仍在同步。");
      } finally {
        if (valid(t)) byId("cancel-button").disabled = false;
      }
    }
    async function older() {
      const session = current(),
        t = token;
      if (!session?.historyCursor) return;
      const button = byId("older-turns");
      button.disabled = true;
      try {
        const page = await api.listTurns(t.sessionId, {
          cursor: session.historyCursor,
          signal: viewAbort?.signal,
        });
        if (!valid(t)) return;
        const added = page.items.filter(
          (run) => !session.turns.has(run.run_id),
        );
        for (const run of page.items) S.applyRun(state, t, run);
        session.historyCursor = page.next_cursor;
        view.prependTurns(added);
        button.hidden = !page.next_cursor;
      } catch (_) {
        if (valid(t)) view.notice("更早消息加载失败，请重试。");
      } finally {
        if (valid(t)) button.disabled = false;
      }
    }
    async function rename() {
      const id = state.selectedSessionId;
      if (!id) return;
      const title = window.prompt("对话名称", summaries.get(id)?.title || "");
      if (title === null) return;
      if (!title.trim() || Array.from(title.trim()).length > 100) {
        view.notice("名称应为 1–100 个字符。");
        return;
      }
      try {
        const summary = await api.renameSession(id, title.trim());
        summaries.set(id, summary);
        if (state.selectedSessionId === id) renderList();
      } catch (_) {
        view.notice("重命名失败，请稍后重试。");
      }
    }
    async function remove() {
      const id = state.selectedSessionId;
      if (
        !id ||
        !window.confirm("删除这个对话？对话将从列表隐藏，用户长期记忆会保留。")
      )
        return;
      try {
        await api.deleteSession(id);
        state.sessions.delete(id);
        summaries.delete(id);
        if (state.selectedSessionId === id) {
          stopObservation();
          clearSources();
          token = S.activateSession(state, null);
          await refreshSessions();
          const first = summaries.keys().next().value;
          if (first) await selectSession(first);
          else await newChat();
        }
        save();
      } catch (error) {
        view.notice(
          error.errorCode === "SESSION_BUSY"
            ? "请先停止正在进行的回答，再删除对话。"
            : "删除失败，请稍后重试。",
        );
      }
    }
    byId("query-input").addEventListener("input", () => {
      if (current()) {
        current().draft = byId("query-input").value;
        updateComposer();
      }
    });
    byId("query-input").addEventListener("compositionstart", () => {
      composing = true;
    });
    byId("query-input").addEventListener("compositionend", () => {
      composing = false;
    });
    byId("query-input").addEventListener("keydown", (event) => {
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing &&
        !composing &&
        event.keyCode !== 229
      ) {
        event.preventDefault();
        void send();
      }
    });
    byId("query-form").addEventListener("submit", (event) => {
      event.preventDefault();
      if (!composing) void send();
    });
    byId("new-chat").addEventListener("click", () => void newChat());
    byId("cancel-button").addEventListener("click", () => void stop());
    byId("rename-chat").addEventListener("click", () => void rename());
    byId("delete-chat").addEventListener("click", () => void remove());
    byId("older-turns").addEventListener("click", () => void older());
    byId("more-sessions").addEventListener(
      "click",
      () => void refreshSessions(true),
    );
    byId("session-list").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-session-id]");
      if (button) void selectSession(button.dataset.sessionId);
    });
    byId("chat-messages").addEventListener("click", (event) => {
      const button = event.target.closest('button[data-action="sources"]');
      if (button) toggleSource(button.dataset.runId);
    });
    byId("check-submission").addEventListener(
      "click",
      () => void checkSubmission(),
    );
    byId("retry-submission").addEventListener("click", () => void send(true));
    for (const kind of ["documents", "memory", "system"])
      byId(`open-${kind}`).addEventListener("click", () => {
        closeSidebar();
        toolView.open(kind);
      });
    byId("open-sidebar").addEventListener("click", () => {
      byId("session-sidebar").classList.add("is-open");
      byId("session-sidebar").setAttribute("role", "dialog");
      byId("session-sidebar").setAttribute("aria-modal", "true");
      byId("sidebar-backdrop").hidden = false;
      byId("new-chat").focus();
    });
    for (const id of ["close-sidebar", "sidebar-backdrop"])
      byId(id).addEventListener("click", () => {
        closeSidebar();
        byId("open-sidebar").focus();
      });
    document.addEventListener("keydown", (event) => {
      if (!byId("session-sidebar").classList.contains("is-open")) return;
      if (event.key === "Escape") {
        closeSidebar();
        byId("open-sidebar").focus();
      }
      if (event.key === "Tab") {
        const focusable = [
          ...byId("session-sidebar").querySelectorAll("button"),
        ].filter((b) => !b.hidden && !b.disabled);
        const first = focusable[0],
          last = focusable.at(-1);
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    });
    let foregroundGeneration = 0;
    async function refreshVisible() {
      const t = token,
        session = current(),
        stamp = ++foregroundGeneration;
      if (!session) return;
      const wasLoaded = session.loaded;
      session.loaded = false;
      stopObservation();
      updateComposer();
      try {
        const [summary, page] = await Promise.all([
          api.getSession(t.sessionId, viewAbort?.signal),
          api.listTurns(t.sessionId, { signal: viewAbort?.signal }),
        ]);
        if (!valid(t) || stamp !== foregroundGeneration) return;
        session.metadata = summary;
        session.activeRunId = summary.active_run_id;
        summaries.set(t.sessionId, summary);
        for (const run of page.items) {
          if (S.applyRun(state, t, run))
            view.updateTurn(session.turns.get(run.run_id));
        }
        session.loaded = true;
        renderList();
        updateComposer();
        save();
        if (session.activeRunId) observe(session.activeRunId, t);
      } catch (error) {
        if (!valid(t) || stamp !== foregroundGeneration) return;
        if ([404, 410].includes(error.status)) {
          clearSources();
          state.sessions.delete(t.sessionId);
          summaries.delete(t.sessionId);
          token = S.activateSession(state, null);
          view.renderSession({ turns: new Map(), orderedRunIds: [] });
          view.notice("此对话已删除或不可访问，请选择其他对话。");
          renderList();
        } else {
          session.loaded = wasLoaded;
          view.notice("会话状态暂未同步，可重新选择此对话以重试。");
          if (session.activeRunId) observe(session.activeRunId, t);
        }
        updateComposer();
      }
    }
    const onFocus = () => {
      if (disposed || document.hidden) return;
      documentsChanged();
      void refreshSessions();
      void refreshVisible();
    };
    window.addEventListener("focus", onFocus);
    const onVisibility = () => {
      if (document.hidden) {
        stopObservation();
        sourceGeneration++;
        for (const id of openSources.keys())
          view.renderSources(id, { status: "loading" });
      } else onFocus();
    };
    document.addEventListener("visibilitychange", onVisibility);
    async function start() {
      updateComposer();
      const resume = S.loadResumeMetadata(storage);
      state.creationRequestId = resume.creationRequestId;
      for (const item of resume.sessions) {
        const session = S.sessionState(state, item.sessionId);
        session.activeRunId = item.activeRunId;
        session.eventCursor = item.eventCursor;
        if (item.pendingRequestId)
          session.pendingSubmission = {
            requestId: item.pendingRequestId,
            question: null,
            status: "unknown",
          };
      }
      await refreshSessions();
      if (state.creationRequestId) {
        await newChat();
        return;
      }
      const selected =
        resume.selectedSessionId || summaries.keys().next().value;
      if (selected) await selectSession(selected);
      else await newChat();
    }
    function dispose() {
      disposed = true;
      stopObservation();
      viewAbort?.abort();
      clearSources();
      toolView.dispose?.();
      window.removeEventListener?.("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    }
    return {
      start,
      selectSession,
      newChat,
      send,
      checkSubmission,
      documentsChanged,
      dispose,
      state,
    };
  }
  const exported = { createChatController };
  if (common) module.exports = exported;
  else {
    root.AgenticRagChat = exported;
    document.addEventListener("DOMContentLoaded", () => {
      const controller = createChatController({ document, window: root });
      root.AgenticRagChat.controller = controller;
      void controller.start();
    });
  }
})(typeof window !== "undefined" ? window : globalThis);
