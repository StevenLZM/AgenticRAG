"use strict";
const assert = require("node:assert/strict"),
  path = require("node:path");
const { fromFile, flush } = require("./chat_dom.cjs");
const dir = process.argv[2],
  { createChatController } = require(path.join(dir, "app.js"));
const { HttpError } = require(path.join(dir, "chat-api.js"));
const summary = { session_id: "s", title: "会话", active_run_id: null };
const window = {
  addEventListener() {},
  removeEventListener() {},
  confirm: () => true,
  prompt: () => null,
};
const tools = { open() {}, dispose() {} };
async function main() {
  const doc = fromFile(path.join(dir, "index.html"));
  let saved = "",
    posts = [];
  const api = {
    listSessions: async () => ({ items: [summary] }),
    getSession: async () => summary,
    listTurns: async () => ({ items: [], next_cursor: null }),
    watchRun: async () => {},
    submitTurn: async (id, q, key) => {
      posts.push({ id, q, key });
      if (posts.length === 1) throw new TypeError("lost response");
      return {
        run_id: "r",
        client_request_id: key,
        question: q,
        status: "queued",
        created_at: "2026-10-01T00:00:00.000000Z",
      };
    },
  };
  const c = createChatController({
    document: doc,
    window,
    tools,
    api,
    storage: {
      getItem: () => null,
      setItem: (k, v) => {
        saved = v;
      },
    },
    uuid: () => "stable-key",
  });
  await c.start();
  const input = doc.getElementById("query-input");
  input.value = "原问题秘密";
  input.dispatchEvent({ type: "input" });
  doc.getElementById("query-form").requestSubmit();
  await flush();
  assert.equal(c.state.sessions.get("s").pendingSubmission.status, "unknown");
  assert(!saved.includes("秘密"));
  input.value = "下一条草稿";
  input.dispatchEvent({ type: "input" });
  doc.getElementById("retry-submission").click();
  await flush();
  assert.deepEqual(posts[0], posts[1]);
  assert.equal(input.value, "下一条草稿");
  c.dispose();
  const resumedDoc = fromFile(path.join(dir, "index.html"));
  let keys = [],
    accepted = false,
    duplicatePosts = 0;
  const resumeApi = {
    ...api,
    submitTurn: async () => {
      duplicatePosts++;
    },
    findSubmission: async (id, key) => {
      keys.push(key);
      if (!accepted) throw new HttpError(404, "CHAT_NOT_FOUND");
      return {
        run_id: "old-r",
        question: "旧问题",
        status: "completed",
        created_at: "2026-10-01T00:00:00.000000Z",
      };
    },
  };
  const resumed = createChatController({
    document: resumedDoc,
    window,
    tools,
    api: resumeApi,
    storage: {
      getItem: () =>
        JSON.stringify({
          version: 1,
          selectedSessionId: "s",
          sessions: [{ sessionId: "s", pendingRequestId: "original-key" }],
        }),
      setItem() {},
    },
  });
  await resumed.start();
  assert.equal(duplicatePosts, 0);
  assert.equal(
    resumed.state.sessions.get("s").pendingSubmission.requestId,
    "original-key",
  );
  resumedDoc.getElementById("check-submission").click();
  await flush();
  assert.deepEqual(keys, ["original-key", "original-key"]);
  assert.equal(resumedDoc.getElementById("retry-submission").hidden, true);
  const draft = resumedDoc.getElementById("query-input");
  draft.value = "恢复后新输入";
  draft.dispatchEvent({ type: "input" });
  accepted = true;
  resumedDoc.getElementById("check-submission").click();
  await flush();
  assert.equal(draft.value, "恢复后新输入");
  assert.equal(resumed.state.sessions.get("s").pendingSubmission, null);
  assert.equal(duplicatePosts, 0);
  resumed.dispose();
  console.log("unknown submission recovery passed");
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
