/* Focused regression checks against the disposable chat preview; no real providers.
 * Uses production transport/controller/view, HTTP interception only for explicit faults.
 */
"use strict";
const assert = require("node:assert/strict");
const { chromium } = require("playwright-core");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.env.CHAT_PREVIEW_URL;
assert(base && /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base));
const output = process.env.CHAT_BROWSER_OUTPUT;
assert(output);
(async () => {
  const browser = await chromium.launch({
    executablePath:
      process.env.CHAT_CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
  });
  const page = await context.newPage(),
    errors = [],
    checks = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const mark = (name) => {
    checks.push(name);
    console.log("PASS", name);
  };
  try {
    await page.goto(base);
    await page.waitForFunction(
      () =>
        window.AgenticRagChat?.controller.state.sessions.get(
          window.AgenticRagChat.controller.state.selectedSessionId,
        )?.loaded && !document.querySelector("#new-chat").disabled,
    );
    const positions = await page.evaluate(() => {
      window.AgenticRagChat.controller.dispose();
      const view = window.AgenticRagChatView.createChatView(document);
      const turn = (n) => ({
        run_id: `r${n}`,
        question: `第${n}问`,
        status: "completed",
        source_status: "available",
        answer: {
          route: "research",
          audited: true,
          segments: [
            {
              kind: "content",
              text: "详细回答\n".repeat(20),
              evidence_ids: ["e1"],
            },
          ],
        },
      });
      const initial = [1, 2, 5, 6].map(turn);
      view.renderSession({
        turns: new Map(initial.map((t) => [t.run_id, t])),
        orderedRunIds: initial.map((t) => t.run_id),
        historyCursor: "older",
      });
      const scroll = document.querySelector("#chat-scroll"),
        anchor = document.querySelector('[data-role="user"][data-run-id="r5"]');
      scroll.scrollTop = anchor.offsetTop - 80;
      view.setSourcesOpen("r5", true);
      view.renderSources("r5", {
        status: "available",
        items: [
          {
            filename: "资料",
            excerpt: "已展开来源",
            heading_path: [],
            version_status: "current",
          },
        ],
      });
      const before = anchor.getBoundingClientRect().top;
      view.prependTurns(
        [turn(3), turn(4)],
        [1, 2, 3, 4, 5, 6].map((n) => `r${n}`),
      );
      return {
        before,
        after: anchor.getBoundingClientRect().top,
        ids: [...document.querySelectorAll('[data-role="user"]')].map(
          (e) => e.dataset.runId,
        ),
        source: document.querySelector("#sources-r5").textContent,
        open: !document.querySelector("#sources-r5").hidden,
      };
    });
    assert.deepEqual(
      positions.ids,
      [1, 2, 3, 4, 5, 6].map((n) => `r${n}`),
    );
    assert(
      Math.abs(positions.after - positions.before) < 2,
      JSON.stringify(positions),
    );
    assert(positions.open && positions.source.includes("已展开来源"));
    mark(
      "cached history gap fills chronologically, preserving viewport anchor and expanded sources",
    );
    await page.reload();
    await page.waitForFunction(
      () =>
        window.AgenticRagChat?.controller.state.sessions.get(
          window.AgenticRagChat.controller.state.selectedSessionId,
        )?.loaded && !document.querySelector("#new-chat").disabled,
    );
    const deletedKey = "00000000-0000-4000-8000-000000000001";
    await page.evaluate(
      (key) =>
        sessionStorage.setItem(
          "agenticrag.chat.resume.v1",
          JSON.stringify({ version: 1, sessions: [], creationRequestId: key }),
        ),
      deletedKey,
    );
    let creates = 0;
    await page.route("**/v1/chat-sessions", async (route) => {
      if (route.request().method() === "POST") {
        creates++;
        if (route.request().postDataJSON().creation_request_id === deletedKey)
          return route.fulfill({
            status: 410,
            json: { error_code: "SESSION_GONE" },
          });
      }
      return route.continue();
    });
    await page.reload();
    await page.waitForFunction(() =>
      document.querySelector("#chat-notice").textContent.includes("已删除"),
    );
    assert.equal(creates, 1);
    assert.equal(
      await page.evaluate(
        () =>
          JSON.parse(sessionStorage.getItem("agenticrag.chat.resume.v1"))
            .creationRequestId,
      ),
      null,
    );
    await page.locator("#new-chat").click();
    await page.waitForFunction(
      () =>
        window.AgenticRagChat.controller.state.selectedSessionId &&
        !document.querySelector("#new-chat").disabled,
    );
    assert.equal(creates, 2);
    mark(
      "real HttpError 410 clears deleted creation key; next explicit click creates a fresh session",
    );
    await page.route("**/health/ready", (route) =>
      route.fulfill({
        status: 503,
        json: { status: "unready", dependencies: { redis: "unavailable" } },
      }),
    );
    await page.route("**/v1/runtime/summary", (route) =>
      route.fulfill({
        status: 200,
        json: {
          runtime_config_snapshot_id: "review-snapshot",
          dependencies: { mysql: "available", redis: "unavailable" },
        },
      }),
    );
    await page.locator("#open-system").click();
    await page.waitForFunction(
      () =>
        document.querySelector("#health-status").textContent === "服务尚未就绪",
    );
    assert.match(
      await page.locator("#health-grid").textContent(),
      /redis：不可用/,
    );
    await page.locator("#close-tools").click();
    mark("503 readiness preserves dependency diagnostics in system drawer");
    const sessionId = await page.evaluate(
      () => window.AgenticRagChat.controller.state.selectedSessionId,
    );
    const submitExternal = async (question) => {
      const response = await context.request.post(
        `${base}/v1/chat-sessions/${sessionId}/turns`,
        {
          data: {
            query: question,
            client_request_id: require("node:crypto").randomUUID(),
          },
        },
      );
      assert.equal(response.status(), 202);
      return (await response.json()).run_id;
    };
    const waitExternal = async (runId) => {
      for (let i = 0; i < 80; i++) {
        const run = await (
          await context.request.get(`${base}/v1/query-runs/${runId}`)
        ).json();
        if (["completed", "failed", "cancelled"].includes(run.status)) return;
        await page.waitForTimeout(250);
      }
      throw new Error("external turn did not finish");
    };
    const r1 = await submitExternal("第一轮说明通知期限。");
    await waitExternal(r1);
    await page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await page.locator(`[data-role="user"][data-run-id="${r1}"]`).waitFor();
    const r2 = await submitExternal("第二轮补充通知依据。");
    await waitExternal(r2);
    const r3 = await submitExternal("第三轮总结通知要求。");
    await page.locator("#query-input").fill("当前标签页的新问题");
    const conflict = page.waitForResponse(
      (r) => r.request().method() === "POST" && r.url().endsWith("/turns"),
    );
    await page.locator("#send-button").click();
    assert.equal((await conflict).status(), 409);
    await page.locator(`[data-role="user"][data-run-id="${r3}"]`).waitFor();
    const beforeCatchup = await page
      .locator('[data-role="user"]')
      .evaluateAll((nodes) => nodes.map((e) => e.dataset.runId));
    assert(beforeCatchup.indexOf(r1) < beforeCatchup.indexOf(r3));
    await page.waitForFunction(
      (id) =>
        window.AgenticRagChat.controller.state.sessions.get(id).turns.size ===
          3 &&
        !window.AgenticRagChat.controller.state.sessions.get(id).activeRunId,
      sessionId,
    );
    assert.deepEqual(
      await page
        .locator('[data-role="user"]')
        .evaluateAll((nodes) => nodes.map((e) => e.dataset.runId)),
      [r1, r2, r3],
    );
    assert.equal(
      await page.locator("#query-input").inputValue(),
      "当前标签页的新问题",
    );
    mark(
      "real SESSION_BUSY catch-up observes the other tab and inserts missed turns chronologically",
    );
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#open-sidebar").click();
    await page.locator("#open-system").click();
    await page.locator("#close-tools").click();
    assert.equal(
      await page
        .locator("#open-sidebar")
        .evaluate((e) => document.activeElement === e),
      true,
      "closing mobile tools must restore focus to the visible sidebar toggle",
    );
    mark("mobile tools close restores focus to visible navigation");
    assert.deepEqual(errors, []);
    await fs.mkdir(output, { recursive: true });
    await page.screenshot({ path: path.join(output, "review-mobile.png") });
    await fs.writeFile(
      path.join(output, "review-results.json"),
      JSON.stringify(
        { status: "PASS", browser: await browser.version(), checks },
        null,
        2,
      ) + "\n",
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
