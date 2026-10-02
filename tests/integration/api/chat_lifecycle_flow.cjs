"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const { fromFile, flush } = require("./chat_dom.cjs");
const dir = process.argv[2];
const { createChatController } = require(path.join(dir, "app.js"));
const { createChatApi, HttpError } = require(path.join(dir, "chat-api.js"));
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};
const answer = {
  route: "chat",
  audited: null,
  segments: [{ kind: "content", text: "第二轮完整答案", evidence_ids: [] }],
};
const turn = (id, status = "running") => ({
  run_id: id,
  question: id,
  status,
  created_at: `2026-10-01T00:00:0${id.endsWith("2") ? 2 : 1}.000000Z`,
});
function setup({ resume = null } = {}) {
  const document = fromFile(path.join(dir, "index.html"));
  let saved = resume,
    sequence = 0;
  const storage = {
    getItem: () => saved,
    setItem: (_, value) => {
      saved = value;
    },
  };
  const sessions = [
    { session_id: "s1", title: "A", active_run_id: "r1" },
    { session_id: "s2", title: "B", active_run_id: "b1" },
  ];
  const histories = { s1: [turn("r1")], s2: [turn("b1")] };
  const listeners = {};
  const watches = [],
    creates = [];
  const api = {
    listSessions: async () => ({ items: sessions, next_cursor: null }),
    getSession: async (id) => sessions.find((s) => s.session_id === id),
    listTurns: async (id) => ({ items: histories[id], next_cursor: "older" }),
    watchRun: async (opts) => {
      watches.push(opts);
    },
    submitTurn: async (id, question, key) => {
      const run = { ...turn("r2", "queued"), question, client_request_id: key };
      histories[id].push(run);
      sessions.find((s) => s.session_id === id).active_run_id = run.run_id;
      return run;
    },
    createSession: async (key) => {
      creates.push(key);
      return sessions[0];
    },
  };
  const controller = createChatController({
    document,
    api,
    storage,
    window: {
      addEventListener(name, fn) {
        listeners[name] = fn;
      },
      removeEventListener() {},
      matchMedia: () => ({ matches: false }),
      confirm: () => true,
    },
    tools: { open() {}, dispose() {} },
    uuid: () => `fresh-${++sequence}`,
  });
  return {
    document,
    api,
    listeners,
    controller,
    histories,
    sessions,
    watches,
    creates,
    saved: () => JSON.parse(saved),
  };
}
async function lateCancel() {
  const h = setup(),
    response = deferred();
  h.api.cancelRun = () => response.promise;
  await h.controller.start();
  h.document.getElementById("cancel-button").click();
  await flush();
  const terminal = { ...turn("r1", "cancelled") };
  h.histories.s1[0] = terminal;
  h.sessions[0].active_run_id = null;
  await h.watches[0].onRun(terminal);
  const input = h.document.getElementById("query-input");
  input.value = "第二轮";
  input.dispatchEvent({ type: "input" });
  await h.controller.send();
  const second = h.watches.at(-1);
  assert.equal(second.runId, "r2");
  assert.equal(second.signal.aborted, false);
  response.resolve(terminal);
  await flush();
  assert.equal(
    second.signal.aborted,
    false,
    "R1 cancellation must not abort R2 observation",
  );
  h.histories.s1[1] = { ...turn("r2", "completed"), answer };
  h.sessions[0].active_run_id = null;
  await second.onRun(h.histories.s1[1]);
  assert.match(
    h.document.getElementById("chat-messages").textContent,
    /第二轮完整答案/,
  );
  assert.equal(h.controller.state.sessions.get("s1").activeRunId, null);
  input.value = "下一问";
  input.dispatchEvent({ type: "input" });
  assert.equal(h.document.getElementById("send-button").disabled, false);
  h.controller.dispose();
}
async function crossSessionCancel() {
  const h = setup(),
    first = deferred(),
    second = deferred();
  let calls = 0;
  h.api.cancelRun = () => {
    calls++;
    return calls === 1 ? first.promise : second.promise;
  };
  await h.controller.start();
  h.document.getElementById("cancel-button").click();
  await flush();
  await h.controller.selectSession("s2");
  assert.equal(
    h.document.getElementById("cancel-button").disabled,
    false,
    "A's pending cancel must not disable B's stop button",
  );
  h.document.getElementById("cancel-button").click();
  await flush();
  assert.equal(calls, 2);
  first.resolve(turn("r1", "cancelled"));
  await flush();
  assert.equal(
    h.document.getElementById("cancel-button").disabled,
    true,
    "A's response must not unlock B's pending stop",
  );
  second.resolve(turn("b1", "cancel_requested"));
  await flush();
  assert.equal(h.document.getElementById("cancel-button").disabled, false);
  h.controller.dispose();
}
async function historySwitch() {
  const h = setup(),
    first = deferred(),
    second = deferred();
  let calls = 0;
  const list = h.api.listTurns;
  h.api.listTurns = (id, options) =>
    options?.cursor
      ? (calls++, id === "s1" ? first.promise : second.promise)
      : list(id);
  await h.controller.start();
  h.document.getElementById("older-turns").click();
  await flush();
  await h.controller.selectSession("s2");
  const button = h.document.getElementById("older-turns");
  assert.equal(button.hidden, false);
  assert.equal(
    button.disabled,
    false,
    "B pagination must not inherit A loading state",
  );
  button.click();
  await flush();
  assert.equal(calls, 2);
  first.resolve({ items: [], next_cursor: null });
  await flush();
  assert.equal(
    button.disabled,
    true,
    "late A cleanup must not unlock B's pending pagination",
  );
  second.resolve({
    items: [{ ...turn("b0", "completed"), answer }],
    next_cursor: null,
  });
  await flush();
  assert.equal(button.disabled, false);
  assert.equal(button.hidden, true);
  assert(h.controller.state.sessions.get("s2").turns.has("b0"));
  h.controller.dispose();
}
async function deletedCreation() {
  const h = setup({
    resume: JSON.stringify({
      version: 1,
      creationRequestId: "deleted-key",
      sessions: [],
    }),
  });
  h.api.createSession = createChatApi({
    fetchImpl: async (_, options) => {
      const key = JSON.parse(options.body).creation_request_id;
      h.creates.push(key);
      return new Response(
        JSON.stringify(
          key === "deleted-key"
            ? { error_code: "SESSION_GONE" }
            : h.sessions[0],
        ),
        { status: key === "deleted-key" ? 410 : 201 },
      );
    },
  }).createSession;
  await h.controller.start();
  assert.deepEqual(
    h.creates,
    ["deleted-key"],
    "410 must not auto-create a replacement session",
  );
  assert.equal(
    h.saved().creationRequestId,
    null,
    "clear the definitively deleted request from recovery storage",
  );
  assert.match(h.document.getElementById("chat-notice").textContent, /已删除/);
  await h.controller.newChat();
  assert.equal(h.creates[1], "fresh-1");
  assert.equal(h.controller.state.selectedSessionId, "s1");
  h.controller.dispose();
}
function draft(h, text) {
  const input = h.document.getElementById("query-input");
  input.value = text;
  input.dispatchEvent({ type: "input" });
}
async function lateSubmissionCheck() {
  const h = setup(),
    first = deferred(),
    second = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  h.api.submitTurn = async () => {
    throw new TypeError("connection lost");
  };
  let calls = 0;
  h.api.findSubmission = () => (++calls === 1 ? first.promise : second.promise);
  await h.controller.start();
  draft(h, "第一轮");
  await h.controller.send();
  const p1 = h.controller.checkSubmission(),
    p2 = h.controller.checkSubmission();
  first.resolve({ ...turn("r1", "completed"), answer });
  await p1;
  h.api.submitTurn = async (_, question, key) => ({
    ...turn("r2"),
    question,
    client_request_id: key,
  });
  draft(h, "第二轮");
  await h.controller.send();
  const secondWatch = h.watches.at(-1);
  second.resolve(turn("r1"));
  await p2;
  assert.equal(
    secondWatch.signal.aborted,
    false,
    "late R1 check must not stop R2",
  );
  assert.equal(h.watches.at(-1).runId, "r2");
  h.controller.dispose();
}
async function lateSubmissionError() {
  const h = setup(),
    first = deferred(),
    second = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  h.api.submitTurn = async () => {
    throw new TypeError("connection lost");
  };
  let calls = 0;
  h.api.findSubmission = () => (++calls === 1 ? first.promise : second.promise);
  await h.controller.start();
  draft(h, "第一轮");
  await h.controller.send();
  const p1 = h.controller.checkSubmission(),
    p2 = h.controller.checkSubmission();
  first.resolve({ ...turn("r1", "completed"), answer });
  await p1;
  draft(h, "第二轮");
  await h.controller.send();
  const notice = h.document.getElementById("chat-notice").textContent;
  second.reject(new HttpError(404, "NOT_FOUND"));
  await p2;
  assert.equal(
    h.document.getElementById("chat-notice").textContent,
    notice,
    "late R1 error must not replace R2 submission notice",
  );
  assert.match(
    h.document.getElementById("chat-messages").textContent,
    /第二轮/,
  );
  h.controller.dispose();
}
async function deleteNavigation() {
  const h = setup(),
    refresh = deferred();
  await h.controller.start();
  h.api.deleteSession = async () => null;
  h.api.listSessions = () => refresh.promise;
  h.document.getElementById("delete-chat").click();
  await flush();
  await h.controller.selectSession("s2");
  refresh.resolve({
    items: [
      { session_id: "s3", title: "C", active_run_id: null },
      h.sessions[1],
    ],
    next_cursor: null,
  });
  await flush();
  assert.equal(
    h.controller.state.selectedSessionId,
    "s2",
    "delete completion must not override later navigation",
  );
  h.controller.dispose();
}
async function historyGap() {
  const h = setup();
  const completed = (n) => ({
    ...turn(`r${n}`, "completed"),
    answer,
    created_at: `2026-10-01T00:00:0${n}.000000Z`,
  });
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [completed(1), completed(2)];
  await h.controller.start();
  await h.controller.selectSession("s2");
  h.histories.s1 = [completed(5), completed(6)];
  await h.controller.selectSession("s1");
  h.api.listTurns = async () => ({
    items: [completed(3), completed(4)],
    next_cursor: "older",
  });
  h.document.getElementById("older-turns").click();
  await flush();
  const ids = h.document
    .getElementById("chat-messages")
    .querySelectorAll('[data-role="user"]')
    .map((e) => e.dataset.runId);
  assert.deepEqual(
    ids,
    ["r1", "r2", "r3", "r4", "r5", "r6"],
    "paginated history must remain chronological across cached page gaps",
  );
  h.controller.dispose();
}
async function revokedObservation() {
  const h = setup(),
    watch = deferred();
  h.api.watchRun = () => watch.promise;
  await h.controller.start();
  draft(h, "下一问");
  watch.reject(new HttpError(410, "SESSION_GONE"));
  await flush();
  assert.equal(
    h.document.getElementById("chat-messages").children.length,
    0,
    "known revoked session must clear the transcript",
  );
  assert.equal(h.document.getElementById("send-button").disabled, true);
  assert.equal(h.saved().selectedSessionId, null);
  h.controller.dispose();
}
async function foregroundDuringSubmit() {
  const h = setup(),
    response = deferred(),
    snapshot = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  await h.controller.start();
  h.api.submitTurn = () => response.promise;
  draft(h, "新问题");
  const sending = h.controller.send();
  h.api.getSession = () => snapshot.promise;
  h.listeners.focus();
  await flush();
  response.resolve({ ...turn("r2"), question: "新问题" });
  await sending;
  snapshot.resolve({ session_id: "s1", title: "A", active_run_id: null });
  await flush();
  draft(h, "下一问");
  assert.equal(
    h.controller.state.sessions.get("s1").activeRunId,
    "r2",
    "foreground snapshot predating POST acknowledgement must not clear the new run",
  );
  assert.equal(h.document.getElementById("send-button").disabled, true);
  assert.equal(h.watches.at(-1).signal.aborted, false);
  h.controller.dispose();
}
async function overlappingForegroundRefresh() {
  const h = setup(),
    response = deferred(),
    snapshot = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  await h.controller.start();
  h.api.submitTurn = () => response.promise;
  draft(h, "新问题");
  const sending = h.controller.send();
  h.api.getSession = () => snapshot.promise;
  h.listeners.focus();
  h.listeners.focus();
  await flush();
  response.resolve({ ...turn("r2", "completed"), answer });
  await sending;
  snapshot.resolve({ session_id: "s1", title: "A", active_run_id: null });
  await flush();
  draft(h, "下一问");
  assert.equal(
    h.document.getElementById("send-button").disabled,
    false,
    "overlapping foreground refreshes must not leave a finished session disabled",
  );
  h.controller.dispose();
}
async function acknowledgedAfterReopen() {
  const h = setup(),
    response = deferred(),
    lookup = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  await h.controller.start();
  h.api.submitTurn = () => response.promise;
  h.api.findSubmission = () => lookup.promise;
  draft(h, "新问题");
  const sending = h.controller.send();
  await h.controller.selectSession("s2");
  const opening = h.controller.selectSession("s1");
  await flush();
  response.resolve({ ...turn("r2"), question: "新问题" });
  await sending;
  lookup.resolve({ ...turn("r2"), question: "新问题" });
  await opening;
  const session = h.controller.state.sessions.get("s1");
  assert.equal(
    session.activeRunId,
    "r2",
    "accepted run must reach the reopened view",
  );
  assert.equal(session.turns.has("r2"), true);
  assert.equal(h.watches.at(-1).runId, "r2");
  assert.equal(
    h.document
      .getElementById("chat-messages")
      .querySelectorAll('[data-run-id="pending"]').length,
    0,
  );
  h.controller.dispose();
}
async function missedTurnCatchup() {
  const h = setup();
  const complete = (n) => ({
    ...turn(`r${n}`, "completed"),
    answer,
    created_at: `2026-10-01T00:00:0${n}.000000Z`,
  });
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [complete(1)];
  await h.controller.start();
  h.api.submitTurn = async () => {
    throw new HttpError(409, "SESSION_BUSY", "/v1/query-runs/r3");
  };
  draft(h, "新问题");
  await h.controller.send();
  const watch = h.watches.at(-1);
  await watch.onRun({ ...complete(3), status: "running", answer: null });
  h.histories.s1 = [complete(1), complete(2), complete(3)];
  await watch.onRun(complete(3));
  assert.deepEqual(
    h.document
      .getElementById("chat-messages")
      .querySelectorAll('[data-role="user"]')
      .map((e) => e.dataset.runId),
    ["r1", "r2", "r3"],
  );
  h.controller.dispose();
}
async function revokedSubmission() {
  const h = setup();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [{ ...turn("r1", "completed"), answer }];
  await h.controller.start();
  h.api.submitTurn = async () => {
    throw new HttpError(410, "SESSION_GONE");
  };
  draft(h, "下一问");
  await h.controller.send();
  assert.equal(h.controller.state.selectedSessionId, null);
  assert.equal(h.document.getElementById("chat-messages").children.length, 0);
  assert.equal(h.document.getElementById("send-button").disabled, true);
  h.controller.dispose();
}
async function queuedProgress(eventType = "QUERY_PHASE_CHANGED") {
  const h = setup();
  h.histories.s1 = [turn("r1", "queued")];
  let stream;
  h.api.watchRun = createChatApi({
    fetchImpl: async (url, options) => {
      if (!url.endsWith("/events")) return Response.json(turn("r1", "queued"));
      return new Response(
        new ReadableStream({
          start(control) {
            stream = control;
            options.signal.addEventListener("abort", () => control.close(), {
              once: true,
            });
          },
        }),
      );
    },
  }).watchRun;
  try {
    await h.controller.start();
    await flush();
    stream.enqueue(
      new TextEncoder().encode(
        `id: 1\nevent: ${eventType}\ndata: {"run_id":"r1","phase":"retrieving"}\n\n`,
      ),
    );
    await flush();
    assert.equal(
      h.controller.state.sessions.get("s1").turns.get("r1").status,
      "running",
    );
    assert.match(
      h.document.getElementById("chat-messages").textContent,
      eventType === "RUN_STARTED" ? /处理中/ : /检索中/,
    );
  } finally {
    h.controller.dispose();
  }
}
async function startedProgress() {
  await queuedProgress("RUN_STARTED");
}
async function staleBusyAfterReopen() {
  const h = setup(),
    post = deferred(),
    lookup = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  h.api.submitTurn = () => post.promise;
  h.api.findSubmission = () => lookup.promise;
  await h.controller.start();
  draft(h, "保留草稿");
  const sending = h.controller.send();
  await h.controller.selectSession("s2");
  h.histories.s1 = [{ ...turn("r2", "completed"), answer }];
  const reopening = h.controller.selectSession("s1");
  await flush();
  post.reject(new HttpError(409, "SESSION_BUSY", "/v1/query-runs/r2"));
  await sending;
  lookup.reject(new HttpError(404, "CHAT_NOT_FOUND"));
  await reopening;
  assert.equal(
    h.controller.state.sessions.get("s1").activeRunId,
    null,
    "an old busy Location must not reactivate a known completed task",
  );
  assert.equal(h.document.getElementById("send-button").disabled, false);
  h.controller.dispose();
}
async function nextRunObservation() {
  const h = setup();
  await h.controller.start();
  const first = h.watches[0];
  await first.onPhase({ run_id: "r1", id: 80, phase: "auditing" });
  const done = { ...turn("r1", "completed"), answer };
  h.histories.s1 = [done, turn("r2")];
  h.sessions[0].active_run_id = "r2";
  await first.onRun(done);
  const second = h.watches.at(-1);
  assert.equal(second.runId, "r2", "history-discovered R2 must be observed");
  assert.equal(second.cursor, 0, "R2 must not inherit R1's event cursor");
  await first.onRun(done);
  assert.equal(
    second.signal.aborted,
    false,
    "late R1 callbacks must not stop R2",
  );
  assert.equal(h.watches.filter((w) => w.runId === "r2").length, 1);
  h.histories.s1[1] = { ...turn("r2", "completed"), answer };
  h.sessions[0].active_run_id = null;
  await second.onRun(h.histories.s1[1]);
  assert.equal(h.controller.state.sessions.get("s1").activeRunId, null);
  assert.match(
    h.document.getElementById("chat-messages").textContent,
    /第二轮完整答案/,
  );
  draft(h, "第三问");
  assert.equal(h.document.getElementById("send-button").disabled, false);
  h.controller.dispose();
}
async function completedBeforeObservation() {
  const h = setup();
  await h.controller.start();
  const first = h.watches[0],
    done = { ...turn("r1", "completed"), answer };
  h.histories.s1 = [done, turn("r2")];
  h.sessions[0].active_run_id = "r2";
  const transport = createChatApi({
    fetchImpl: async (url) => {
      assert.equal(
        url,
        "/v1/query-runs/r2",
        "completed R2 needs no SSE connection",
      );
      const completed = { ...turn("r2", "completed"), answer };
      h.histories.s1[1] = completed;
      h.sessions[0].active_run_id = null;
      return Response.json(completed);
    },
  });
  h.api.watchRun = transport.watchRun;
  await first.onRun(done);
  await flush();
  assert.equal(
    h.controller.state.sessions.get("s1").turns.get("r2").status,
    "completed",
  );
  draft(h, "第三问");
  assert.equal(h.document.getElementById("send-button").disabled, false);
  h.controller.dispose();
}
async function switchDuringTerminalRefresh() {
  const h = setup(),
    refresh = deferred();
  await h.controller.start();
  const first = h.watches[0],
    list = h.api.listTurns;
  h.api.listTurns = (id, options) =>
    id === "s1" ? refresh.promise : list(id, options);
  const catchingUp = first.onRun({ ...turn("r1", "completed"), answer });
  await h.controller.selectSession("s2");
  const currentWatch = h.watches.at(-1);
  refresh.resolve({ items: [turn("r2")], next_cursor: null });
  await catchingUp;
  assert.equal(h.controller.state.selectedSessionId, "s2");
  assert.equal(h.watches.at(-1).runId, "b1");
  assert.equal(currentWatch.signal.aborted, false);
  assert.doesNotMatch(
    h.document.getElementById("chat-messages").textContent,
    /r2/,
  );
  h.controller.dispose();
}
async function rejectedAfterReopen(status = 422, code = "INVALID_INPUT") {
  const h = setup(),
    post = deferred(),
    lookup = deferred();
  h.sessions[0].active_run_id = null;
  h.histories.s1 = [];
  h.api.submitTurn = () => post.promise;
  h.api.findSubmission = () => lookup.promise;
  await h.controller.start();
  draft(h, "保留草稿");
  const sending = h.controller.send();
  await h.controller.selectSession("s2");
  const reopening = h.controller.selectSession("s1");
  await flush();
  if (code === "SESSION_BUSY") {
    h.sessions[0].active_run_id = "r2";
    h.histories.s1 = [turn("r2")];
  }
  post.reject(new HttpError(status, code, "/v1/query-runs/r2"));
  await sending;
  lookup.reject(new HttpError(404, "CHAT_NOT_FOUND"));
  await reopening;
  assert.doesNotMatch(
    h.document.getElementById("chat-messages").textContent,
    /提交中/,
  );
  if (status === 410) {
    assert.equal(h.controller.state.selectedSessionId, null);
  } else {
    const session = h.controller.state.sessions.get("s1");
    assert.equal(session.draft, "保留草稿");
    if (status === 503) {
      assert.equal(session.pendingSubmission.status, "unknown");
      assert.equal(h.document.getElementById("pending-actions").hidden, false);
    } else {
      assert.equal(session.pendingSubmission, null);
      assert.equal(
        h.document.getElementById("send-button").disabled,
        code === "SESSION_BUSY",
      );
      if (code === "SESSION_BUSY") {
        assert.equal(h.watches.at(-1).runId, "r2");
        h.histories.s1 = [{ ...turn("r2", "completed"), answer }];
        h.sessions[0].active_run_id = null;
        await h.watches.at(-1).onRun(h.histories.s1[0]);
        assert.equal(h.document.getElementById("send-button").disabled, false);
      }
    }
  }
  h.controller.dispose();
}
const cases = {
  queuedProgress,
  startedProgress,
  staleBusyAfterReopen,
  nextRunObservation,
  completedBeforeObservation,
  switchDuringTerminalRefresh,
  rejectedAfterReopen,
  busyAfterReopen: () => rejectedAfterReopen(409, "SESSION_BUSY"),
  goneAfterReopen: () => rejectedAfterReopen(410, "SESSION_GONE"),
  unknownAfterReopen: () => rejectedAfterReopen(503, "UNAVAILABLE"),
  revokedSubmission,
  missedTurnCatchup,
  acknowledgedAfterReopen,
  overlappingForegroundRefresh,
  foregroundDuringSubmit,
  lateSubmissionCheck,
  lateSubmissionError,
  deleteNavigation,
  historyGap,
  revokedObservation,
  lateCancel,
  crossSessionCancel,
  historySwitch,
  deletedCreation,
};
cases[process.argv[3]]()
  .then(() => console.log(`${process.argv[3]} passed`))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
