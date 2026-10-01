"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const { fromFile, flush } = require("./chat_dom.cjs");
const dir = process.argv[2];
const { createChatController } = require(path.join(dir, "app.js"));
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
      addEventListener() {},
      removeEventListener() {},
      matchMedia: () => ({ matches: false }),
    },
    tools: { open() {}, dispose() {} },
    uuid: () => `fresh-${++sequence}`,
  });
  return {
    document,
    api,
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
  h.api.createSession = async (key) => {
    h.creates.push(key);
    if (key === "deleted-key")
      throw Object.assign(new Error("deleted"), {
        status: 410,
        code: "SESSION_GONE",
      });
    return h.sessions[0];
  };
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
const cases = {
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
