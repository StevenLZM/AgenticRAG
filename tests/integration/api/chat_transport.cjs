"use strict";
const assert = require("node:assert/strict");
const { createChatApi, HttpError } = require(
  require("node:path").join(process.argv[2], "chat-api.js"),
);
const json = (value, status = 200, headers = {}) =>
  new Response(JSON.stringify(value), { status, headers });
const run = (status) => ({ run_id: "r", thread_id: "s", status });
const encoder = new TextEncoder();
function stream(chunks) {
  return new Response(
    new ReadableStream({
      start(c) {
        for (const x of chunks) c.enqueue(x);
        c.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
}
async function main() {
  // Replaying a lost response and a 404 lookup never invent another key.
  const bodies = [];
  const postApi = createChatApi({
    fetchImpl: async (url, options) => {
      if (url.includes("submissions"))
        return json({ error_code: "CHAT_NOT_FOUND" }, 404);
      bodies.push(JSON.parse(options.body));
      if (bodies.length === 1) throw new TypeError("network lost after commit");
      return json({ run_id: "r" });
    },
  });
  await assert.rejects(postApi.submitTurn("s", "问题秘密", "request"));
  await assert.rejects(
    postApi.findSubmission("s", "request"),
    (e) => e instanceof HttpError && e.status === 404,
  );
  await postApi.submitTurn("s", "问题秘密", "request");
  assert.deepEqual(bodies[0], bodies[1]);
  const calls = [],
    phases = [],
    snapshots = [];
  let gets = 0;
  const bytes = encoder.encode(
    'id: 7\r\nevent: QUERY_PHASE_CHANGED\r\ndata: {"run_id":"r","phase":"auditing","summary":"审核中"}\r\n\r\n',
  );
  const api = createChatApi({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith("events"))
        return stream(Array.from(bytes, (b) => new Uint8Array([b])));
      return json(run(++gets === 1 ? "running" : "completed"));
    },
  });
  await api.watchRun({
    runId: "r",
    cursor: 3,
    signal: new AbortController().signal,
    onPhase: (p) => phases.push(p),
    onRun: (r) => snapshots.push(r),
  });
  assert.equal(
    calls.find((c) => c.url.endsWith("events")).options.headers[
      "Last-Event-ID"
    ],
    "3",
  );
  assert.equal(phases.length, 1);
  assert.equal(phases[0].phase, "auditing");
  assert.equal(phases[0].id, 7);
  assert.equal(snapshots.at(-1).status, "completed"); // EOF verified via GET
  const delays = [],
    pollingCalls = [];
  let pollGets = 0;
  const timers = {
    setTimeout(fn, ms) {
      delays.push(ms);
      queueMicrotask(fn);
      return delays.length;
    },
    clearTimeout() {},
  };
  const fallback = createChatApi({
    timers,
    fetchImpl: async (url) => {
      pollingCalls.push(url);
      if (url.endsWith("events")) throw new TypeError("offline");
      return json(run(++pollGets >= 6 ? "completed" : "running"));
    },
  });
  await fallback.watchRun({
    runId: "r",
    signal: new AbortController().signal,
    onRun() {},
    onPhase() {},
  });
  assert.deepEqual(delays.slice(0, 3), [500, 1000, 2000]);
  assert.equal(delays.at(-1), 2000);
  assert.equal(pollingCalls.filter((u) => u.endsWith("events")).length, 4);
  const controller = new AbortController();
  const abortCalls = [];
  const observing = createChatApi({
    fetchImpl: async (url, opts) => {
      abortCalls.push(url);
      if (!url.endsWith("events")) return json(run("running"));
      controller.abort();
      throw new DOMException("Aborted", "AbortError");
    },
  });
  await observing.watchRun({
    runId: "r",
    signal: controller.signal,
    onRun() {},
    onPhase() {},
  });
  assert(!abortCalls.some((u) => u.endsWith("cancel")));
  let reads = 0;
  const resumed = createChatApi({
    fetchImpl: async (url) => {
      reads++;
      return json(run("completed"));
    },
  });
  await resumed.watchRun({
    runId: "r",
    cursor: 7,
    signal: new AbortController().signal,
    onRun() {},
    onPhase() {},
  });
  assert.equal(reads, 1); // hide/restore starts from authoritative GET
  const deleted = createChatApi({
    fetchImpl: async () => json({ error_code: "CHAT_NOT_FOUND" }, 404),
  });
  await assert.rejects(
    deleted.watchRun({
      runId: "r",
      signal: new AbortController().signal,
      onRun() {},
    }),
    (e) => e.status === 404,
  );
  console.log("chat transport contracts passed");
}
main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
