"use strict";
const assert = require("node:assert/strict"),
  path = require("node:path");
const { fromFile, flush } = require("./chat_dom.cjs");
const dir = path.dirname(process.argv[2]);
const { createChatController } = require(path.join(dir, "app.js"));
async function main() {
  const document = fromFile(path.join(dir, "index.html"));
  const listeners = {};
  const window = {
    addEventListener: (name, fn) => {
      listeners[name] = fn;
    },
    matchMedia: () => ({ matches: false }),
    confirm: () => true,
    prompt: () => "新标题",
  };
  const sessions = [
    { session_id: "s1", title: "第一段对话", active_run_id: null },
    { session_id: "s2", title: "第二段对话", active_run_id: null },
  ];
  let holdSession = null,
    sends = 0,
    sources = 0,
    cancels = 0,
    watchCallbacks = null,
    lateSource;
  const answer = {
    route: "research",
    audited: true,
    segments: [{ kind: "content", text: "安全答案", evidence_ids: ["e1"] }],
  };
  const histories = {
    s1: [
      {
        run_id: "r1",
        client_request_id: "key1",
        question: "原问题",
        status: "completed",
        answer,
        created_at: "2026-10-01T00:00:00.000001Z",
        source_status: "available",
      },
    ],
    s2: [],
  };
  const api = {
    listSessions: async () => ({ items: sessions, next_cursor: null }),
    getSession: async (id) =>
      holdSession
        ? new Promise((resolve) => {
            holdSession = () =>
              resolve(sessions.find((s) => s.session_id === id));
          })
        : sessions.find((s) => s.session_id === id),
    listTurns: async (id) => ({ items: histories[id], next_cursor: null }),
    submitTurn: async (id, q, key) => {
      sends++;
      const run = {
        run_id: "r-new",
        client_request_id: key,
        question: q,
        status: "queued",
        created_at: "2026-10-01T00:00:01.000001Z",
      };
      histories[id].push(run);
      return run;
    },
    watchRun: async (opts) => {
      watchCallbacks = opts;
    },
    getSources: async () => {
      sources++;
      if (lateSource)
        return new Promise((r) => {
          lateSource = r;
        });
      return {
        status: "available",
        items: [
          {
            filename: "资料",
            excerpt: "引用片段",
            heading_path: [],
            version_status: "current",
          },
        ],
      };
    },
    cancelRun: async () => {
      cancels++;
      return { run_id: "r-new", status: "cancel_requested" };
    },
    renameSession: async (id, title) => ({
      ...sessions.find((s) => s.session_id === id),
      title,
    }),
    deleteSession: async () => null,
  };
  const tools = { open() {}, close() {}, dispose() {} };
  const storage = { getItem: () => null, setItem() {} };
  const controller = createChatController({
    document,
    window,
    api,
    storage,
    tools,
    uuid: () => "new-key",
  });
  await controller.start();
  const input = document.getElementById("query-input");
  input.value = "中文问题";
  input.dispatchEvent({ type: "input" });
  input.dispatchEvent({ type: "compositionstart" });
  input.dispatchEvent({ type: "keydown", key: "Enter", isComposing: true });
  await flush();
  assert.equal(sends, 0);
  input.dispatchEvent({ type: "compositionend" });
  input.dispatchEvent({ type: "keydown", key: "Enter", shiftKey: true });
  await flush();
  assert.equal(sends, 0);
  input.dispatchEvent({ type: "keydown", key: "Enter" });
  await flush();
  assert.equal(sends, 1);
  assert.equal(document.getElementById("chat-messages").children.length, 4);
  input.value = "下一条草稿";
  input.dispatchEvent({ type: "input" });
  const oldWatch = watchCallbacks;
  await controller.selectSession("s2");
  await oldWatch.onRun({
    run_id: "r-new",
    thread_id: "s1",
    status: "completed",
    answer,
  });
  assert.equal(document.getElementById("chat-messages").children.length, 0);
  assert.equal(cancels, 0);
  await controller.selectSession("s1");
  assert.equal(input.value, "下一条草稿");
  const button = document
    .getElementById("chat-messages")
    .querySelector('[data-action="sources"]');
  button.click();
  await flush();
  assert.equal(sources, 1);
  button.click();
  button.click();
  await flush();
  assert.equal(sources, 2);
  listeners.focus();
  await flush();
  assert.equal(sources, 3);
  controller.documentsChanged();
  await flush();
  assert.equal(sources, 4);
  // Collapse while a request is pending: a late source cannot reappear.
  button.click();
  lateSource = true;
  button.click();
  await flush();
  button.click();
  lateSource({
    status: "available",
    items: [{ filename: "private", excerpt: "late secret", heading_path: [] }],
  });
  await flush();
  assert(
    !document
      .getElementById("chat-messages")
      .textContent.includes("late secret"),
  );
  document.hidden = true;
  document.dispatchEvent({ type: "visibilitychange" });
  await flush();
  assert.equal(cancels, 0);
  document.hidden = false;
  document.dispatchEvent({ type: "visibilitychange" });
  await flush();
  document.getElementById("cancel-button").click();
  await flush();
  assert.equal(cancels, 1);
  holdSession = true;
  const loading = controller.selectSession("s2");
  await flush();
  input.value = "加载时输入";
  input.dispatchEvent({ type: "input" });
  input.dispatchEvent({ type: "keydown", key: "Enter" });
  await flush();
  assert.equal(sends, 1);
  holdSession();
  await loading;
  holdSession = null;
  histories.s2.push({
    run_id: "background",
    question: "另一标签页的问题",
    status: "running",
    created_at: "2026-10-01T00:00:02.000000Z",
  });
  sessions[1].active_run_id = "background";
  listeners.focus();
  await flush();
  assert(
    document
      .getElementById("chat-messages")
      .textContent.includes("另一标签页的问题"),
  );
  assert.equal(document.getElementById("send-button").disabled, true);
  controller.dispose();
  console.log("chat controller flow passed");
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
