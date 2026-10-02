/* Real Chrome + production assets/HTTP/SSE, with an isolated in-memory server.
 * No application database, model provider, or running application is touched.
 * NODE_PATH must contain playwright-core; CHAT_BROWSER_OUTPUT is optional.
 */
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const http = require("node:http");
const path = require("node:path");
const { chromium } = require("playwright-core");
const root = path.resolve(__dirname, "../../src/agentic_rag/api/static");
const sessions = [
  { session_id: "s1", title: "会话 A", active_run_id: null },
  { session_id: "s2", title: "会话 B", active_run_id: null },
];
const runs = new Map(),
  streams = new Map(),
  events = new Map();
let heldHistory = null,
  holdHistory = false,
  heldPost = null,
  posts = 0;
const json = (response, value, status = 200, headers = {}) => {
  response.writeHead(status, {
    "Content-Type": "application/json",
    ...headers,
  });
  response.end(JSON.stringify(value));
};
const history = (id) => ({
  items: [...runs.values()].filter((r) => r.thread_id === id),
  next_cursor: null,
});
function emit(id, type, data = {}) {
  const frames = events.get(id) || [];
  const frame = { id: frames.length + 1, text: "" };
  frame.text = `id: ${frame.id}\nevent: ${type}\ndata: ${JSON.stringify({ run_id: id, ...data })}\n\n`;
  frames.push(frame);
  events.set(id, frames);
  for (const response of streams.get(id) || []) response.write(frame.text);
}
function phase(id, value) {
  Object.assign(runs.get(id), { status: "running", phase: value });
  emit(id, "QUERY_PHASE_CHANGED", { phase: value });
}
function complete(id) {
  Object.assign(runs.get(id), {
    status: "completed",
    phase: null,
    answer: {
      route: "chat",
      audited: null,
      segments: [
        { kind: "content", text: `${id} 的完整回答`, evidence_ids: [] },
      ],
    },
  });
  sessions[0].active_run_id = null;
  emit(id, "RUN_COMPLETED");
}
const server = http.createServer(async (request, response) => {
  const url = new URL(request.url, "http://localhost");
  const segments = url.pathname.split("/").filter(Boolean);
  try {
    if (segments[0] !== "v1") {
      const name =
        url.pathname === "/" ? "index.html" : path.basename(url.pathname);
      const contents = await fs.readFile(path.join(root, name));
      response.writeHead(200, {
        "Content-Type": name.endsWith(".js")
          ? "application/javascript"
          : name.endsWith(".css")
            ? "text/css"
            : "text/html",
      });
      response.end(contents);
    } else if (segments[1] === "chat-sessions") {
      const id = segments[2];
      if (!id) return json(response, { items: sessions, next_cursor: null });
      if (!segments[3])
        return json(
          response,
          sessions.find((s) => s.session_id === id),
        );
      if (segments[3] === "submissions")
        return json(response, { error_code: "CHAT_NOT_FOUND" }, 404);
      if (segments[3] === "turns" && request.method === "POST") {
        let body = "";
        for await (const chunk of request) body += chunk;
        const input = JSON.parse(body);
        posts++;
        if (input.query === "迟到拒绝") {
          heldPost = response;
          return;
        }
        const runId = `r${runs.size + 1}`;
        const run = {
          run_id: runId,
          thread_id: id,
          question: input.query,
          client_request_id: input.client_request_id,
          status: "queued",
          phase: null,
          created_at: `2026-10-02T00:00:0${runs.size + 1}.000000Z`,
        };
        runs.set(runId, run);
        sessions.find((s) => s.session_id === id).active_run_id = runId;
        return json(response, run, 202);
      }
      if (segments[3] === "turns") {
        if (
          holdHistory &&
          request.headers["x-test-tab"] === "A" &&
          id === "s1"
        ) {
          holdHistory = false;
          heldHistory = response;
          return;
        }
        return json(response, history(id));
      }
      json(response, { error_code: "NOT_FOUND" }, 404);
    } else if (segments[1] === "query-runs") {
      const id = segments[2];
      if (segments[3] !== "events") return json(response, runs.get(id));
      response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
      });
      response.write(": connected\n\n");
      const open = streams.get(id) || new Set();
      open.add(response);
      streams.set(id, open);
      response.on("close", () => open.delete(response));
      const cursor = Number(request.headers["last-event-id"] || 0);
      for (const frame of events.get(id) || [])
        if (frame.id > cursor) response.write(frame.text);
    } else json(response, { error_code: "NOT_FOUND" }, 404);
  } catch (error) {
    json(
      response,
      { error_code: "FIXTURE_FAILURE", detail: error.message },
      500,
    );
  }
});
async function until(predicate) {
  const end = Date.now() + 10000;
  while (!predicate()) {
    assert(Date.now() < end, "timed out waiting for controlled HTTP boundary");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
async function loaded(page) {
  await page.waitForFunction(
    () => window.AgenticRagChat?.controller.state.sessions.get("s1")?.loaded,
  );
}
async function send(page, text) {
  await page.locator("#query-input").fill(text);
  await page.locator("#send-button").click();
}
async function bodyContains(page, id, text) {
  await page.waitForFunction(
    ({ id, text }) =>
      document
        .querySelector(`[data-role="assistant"][data-run-id="${id}"]`)
        ?.textContent.includes(text),
    { id, text },
  );
}
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let browser;
  const checks = [],
    errors = [];
  const mark = (name) => {
    checks.push(name);
    console.log("PASS", name);
  };
  try {
    browser = await chromium.launch({
      headless: true,
      executablePath:
        process.env.CHAT_CHROME_PATH ||
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    });
    const context = await browser.newContext({
      viewport: { width: 1440, height: 960 },
    });
    const a = await context.newPage(),
      b = await context.newPage();
    for (const [page, name] of [
      [a, "A"],
      [b, "B"],
    ]) {
      page.on("pageerror", (error) => errors.push(error.message));
      await page.setExtraHTTPHeaders({ "X-Test-Tab": name });
      await page.goto(`http://127.0.0.1:${server.address().port}`);
      await loaded(page);
    }
    await send(a, "第一问");
    await bodyContains(a, "r1", "等待中");
    await until(() => streams.get("r1")?.size);
    emit("r1", "RUN_STARTED");
    await bodyContains(a, "r1", "处理中");
    phase("r1", "retrieving");
    await bodyContains(a, "r1", "检索中");
    mark(
      "queued run advances on RUN_STARTED and phase SSE without page reload",
    );

    // B reloads to observe the same first run, then submits R2 while A's
    // terminal history refresh is in flight. Both pages use the real transport.
    await b.reload();
    await loaded(b);
    await bodyContains(b, "r1", "检索中");
    holdHistory = true;
    complete("r1");
    await until(() => heldHistory);
    await bodyContains(b, "r1", "完整回答");
    await send(b, "第二问");
    await until(() => runs.has("r2"));
    phase("r2", "researching");
    json(heldHistory, history("s1"));
    heldHistory = null;
    await bodyContains(a, "r2", "研究中");
    phase("r2", "auditing");
    await bodyContains(a, "r2", "审核中");
    complete("r2");
    await bodyContains(a, "r2", "完整回答");
    await a.locator("#query-input").fill("下一问");
    assert.equal(await a.locator("#send-button").isEnabled(), true);
    mark(
      "A subscribes to B's R2 after terminal refresh and unlocks on completion",
    );

    await send(a, "迟到拒绝");
    await until(() => heldPost);
    await a.locator('[data-session-id="s2"]').click();
    await a.locator('[data-session-id="s1"]').click();
    await a.waitForFunction(
      () => !document.querySelector("#pending-actions").hidden,
    );
    await bodyContains(b, "r2", "完整回答");
    await send(b, "第三问");
    await until(() => runs.has("r3"));
    json(heldPost, { error_code: "SESSION_BUSY" }, 409, {
      Location: "/v1/query-runs/r3",
    });
    heldPost = null;
    phase("r3", "retrieving");
    await bodyContains(a, "r3", "检索中");
    assert.equal(await a.locator('[data-run-id="pending"]').count(), 0);
    complete("r3");
    await bodyContains(a, "r3", "完整回答");
    assert.equal(await a.locator("#query-input").inputValue(), "迟到拒绝");
    assert.equal(await a.locator("#send-button").isEnabled(), true);
    assert.equal(posts, 4, "recovery must not resubmit the rejected question");
    mark(
      "late SESSION_BUSY after A-B-A removes pending UI, observes busy run and preserves draft",
    );
    assert.deepEqual(errors, []);
    if (process.env.CHAT_BROWSER_OUTPUT) {
      await fs.mkdir(process.env.CHAT_BROWSER_OUTPUT, { recursive: true });
      await a.screenshot({
        path: path.join(process.env.CHAT_BROWSER_OUTPUT, "recovered-chat.png"),
        fullPage: true,
      });
      await fs.writeFile(
        path.join(process.env.CHAT_BROWSER_OUTPUT, "results.json"),
        JSON.stringify({ checks, errors }, null, 2),
      );
    }
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
