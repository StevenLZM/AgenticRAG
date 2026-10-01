/* Session-local drafts and monotonic Run state. No answer text enters storage. */
(function (root) {
  "use strict";
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);
  const PHASES = new Set([
    "processing",
    "retrieving",
    "researching",
    "auditing",
  ]);
  const RESUME_KEY = "agenticrag.chat.resume.v1";
  const safeId = (value) =>
    typeof value === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(value);
  const safeCursor = (value) =>
    Number.isSafeInteger(value) && value >= 0 ? value : 0;
  const isTerminal = (run) => !!run && TERMINAL.has(run.status);
  const isCurrent = (state, token) =>
    !!token &&
    token.sessionId === state.selectedSessionId &&
    token.generation === state.viewGeneration;

  function createChatState() {
    return {
      selectedSessionId: null,
      viewGeneration: 0,
      sessions: new Map(),
      creationRequestId: null,
      storageWarning: false,
    };
  }
  function sessionState(state, id) {
    if (!state.sessions.has(id))
      state.sessions.set(id, {
        sessionId: id,
        turns: new Map(),
        orderedRunIds: [],
        draft: "",
        pendingSubmission: null,
        historyCursor: null,
        activeRunId: null,
        eventCursor: 0,
        loaded: false,
      });
    return state.sessions.get(id);
  }
  function activateSession(state, sessionId) {
    state.selectedSessionId = sessionId;
    state.viewGeneration += 1;
    if (sessionId) sessionState(state, sessionId);
    return { sessionId, generation: state.viewGeneration };
  }
  function compareTurns(a, b) {
    // Server RFC3339 has six fractional digits. Date would discard precision.
    const at = a.created_at || "",
      bt = b.created_at || "";
    if (at !== bt) return at < bt ? -1 : 1;
    return a.run_id === b.run_id ? 0 : a.run_id < b.run_id ? -1 : 1;
  }
  function applyRun(state, token, turn) {
    if (!isCurrent(state, token) || !turn || !safeId(turn.run_id)) return false;
    if (turn.thread_id && turn.thread_id !== token.sessionId) return false;
    const session = sessionState(state, token.sessionId);
    const previous = session.turns.get(turn.run_id);
    if (isTerminal(previous) && turn.status !== previous.status) return false;
    if (
      previous?.status === "cancel_requested" &&
      ["queued", "running"].includes(turn.status)
    )
      return false;
    if (previous?.status === "running" && turn.status === "queued")
      return false;
    const merged = { ...previous, ...turn };
    if (isTerminal(merged)) merged.phase = null;
    session.turns.set(turn.run_id, merged);
    session.orderedRunIds = [...session.turns.values()]
      .sort(compareTurns)
      .map((item) => item.run_id);
    if (isTerminal(merged)) {
      if (session.activeRunId === merged.run_id) session.activeRunId = null;
    } else if (
      ["queued", "running", "cancel_requested"].includes(merged.status)
    ) {
      session.activeRunId = merged.run_id;
    }
    return true;
  }
  function applyPhase(state, token, runId, eventId, phase) {
    if (
      !isCurrent(state, token) ||
      !PHASES.has(phase) ||
      !Number.isSafeInteger(eventId)
    )
      return false;
    const session = sessionState(state, token.sessionId);
    const turn = session.turns.get(runId);
    if (!turn || isTerminal(turn) || eventId <= (turn.eventCursor || 0))
      return false;
    turn.eventCursor = eventId;
    turn.phase = phase;
    if (session.activeRunId === runId) session.eventCursor = eventId;
    return true;
  }
  function beginSubmission(state, id, question, requestId) {
    const session = sessionState(state, id);
    if (session.pendingSubmission) return session.pendingSubmission;
    session.pendingSubmission = {
      requestId,
      question: question.trim(),
      status: "submitting",
    };
    return session.pendingSubmission;
  }
  function markSubmissionUnknown(state, id, requestId) {
    const pending = sessionState(state, id).pendingSubmission;
    if (pending?.requestId === requestId) pending.status = "unknown";
  }
  function acknowledgeSubmission(state, id, requestId) {
    const session = sessionState(state, id),
      pending = session.pendingSubmission;
    if (pending?.requestId !== requestId) return false;
    if (pending.question !== null && session.draft.trim() === pending.question)
      session.draft = "";
    session.pendingSubmission = null;
    return true;
  }
  function rejectSubmission(state, id, requestId) {
    const session = sessionState(state, id);
    if (session.pendingSubmission?.requestId === requestId)
      session.pendingSubmission = null;
  }
  function saveResumeMetadata(storage, state) {
    try {
      const sessions = [...state.sessions.values()]
        .filter((s) => s.activeRunId || s.pendingSubmission)
        .slice(-100)
        .map((s) => ({
          sessionId: s.sessionId,
          activeRunId: s.activeRunId,
          eventCursor: safeCursor(s.eventCursor),
          pendingRequestId: s.pendingSubmission?.requestId || null,
        }));
      storage.setItem(
        RESUME_KEY,
        JSON.stringify({
          version: 1,
          selectedSessionId: state.selectedSessionId,
          creationRequestId: state.creationRequestId,
          sessions,
        }),
      );
      state.storageWarning = false;
    } catch (_) {
      state.storageWarning = true;
    }
  }
  function loadResumeMetadata(storage) {
    const empty = {
      version: 1,
      selectedSessionId: null,
      creationRequestId: null,
      sessions: [],
    };
    try {
      const raw = storage.getItem(RESUME_KEY);
      if (!raw || raw.length > 65536) return empty;
      const value = JSON.parse(raw);
      if (value?.version !== 1 || !Array.isArray(value.sessions)) return empty;
      return {
        version: 1,
        selectedSessionId: safeId(value.selectedSessionId)
          ? value.selectedSessionId
          : null,
        creationRequestId: safeId(value.creationRequestId)
          ? value.creationRequestId
          : null,
        sessions: value.sessions
          .slice(0, 100)
          .filter((s) => s && safeId(s.sessionId))
          .map((s) => ({
            sessionId: s.sessionId,
            activeRunId: safeId(s.activeRunId) ? s.activeRunId : null,
            pendingRequestId: safeId(s.pendingRequestId)
              ? s.pendingRequestId
              : null,
            eventCursor: safeCursor(s.eventCursor),
          })),
      };
    } catch (_) {
      return empty;
    }
  }
  const api = {
    createChatState,
    sessionState,
    activateSession,
    isCurrent,
    isTerminal,
    applyRun,
    applyPhase,
    beginSubmission,
    markSubmissionUnknown,
    acknowledgeSubmission,
    rejectSubmission,
    saveResumeMetadata,
    loadResumeMetadata,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AgenticRagChatState = api;
})(typeof window !== "undefined" ? window : globalThis);
