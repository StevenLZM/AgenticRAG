/* Opt-in Chrome acceptance against a disposable deterministic Query fixture.
 * NODE_PATH supplies playwright-core; no project runtime dependency is added.
 * CHAT_PREVIEW_URL must point at the isolated fixture, never a production account.
 */
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright-core");
const base = process.env.CHAT_PREVIEW_URL;
assert(
  base && /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(base),
  "explicit local fixture URL required",
);
const output =
  process.env.CHAT_BROWSER_OUTPUT ||
  "/private/tmp/agenticrag-chat-browser-results";
const checks = [];
const mark = (name) => {
  checks.push(name);
  console.log("PASS", name);
};
(async () => {
  await fs.mkdir(output, { recursive: true });
  const browser = await chromium.launch({
    executablePath:
      process.env.CHAT_CHROME_PATH ||
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true,
  });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 960 },
  });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const input = page.locator("#query-input");
  const selected = () =>
    page
      .locator('#session-list [aria-current="true"]')
      .getAttribute("data-session-id");
  const ready = () =>
    page.waitForFunction(
      () =>
        document.querySelector('#session-list [aria-current="true"]') &&
        !document.querySelector("#new-chat").disabled,
    );
  const waitDone = () =>
    page.waitForFunction(
      () =>
        document.querySelectorAll(".message-assistant").length &&
        !document.querySelectorAll(".message-pending").length,
      { timeout: 30000 },
    );
  const submit = async (text) => {
    await input.fill(text);
    const response = page.waitForResponse(
      (r) => r.request().method() === "POST" && /\/turns$/.test(r.url()),
    );
    await page.locator("#send-button").click();
    const result = await response;
    assert.equal(result.status(), 202);
    return (await result.json()).run_id;
  };
  try {
    await page.goto(base);
    await ready();
    await page.locator("#new-chat").click();
    await ready();
    const a = await selected();
    const runs = [];
    for (const question of [
      "请解释合同要求的通知期限。",
      "为什么需要提前通知？",
      "总结一下前面的约定。",
    ]) {
      runs.push(await submit(question));
      assert.equal(
        await page.locator(".message-pending .source-button:visible").count(),
        0,
      );
      await waitDone();
    }
    assert.equal(await page.locator(".message-user").count(), 3);
    assert.equal(await page.locator(".message-assistant").count(), 3);
    assert.equal(new Set(runs).size, 3);
    assert.equal(await input.inputValue(), "");
    assert.equal(await page.locator(".sources:visible").count(), 0);
    mark(
      "1440px three chronological turns, final-only answers, per-answer collapsed sources",
    );
    page.once("dialog", (dialog) => dialog.accept("合同通知讨论"));
    await page.locator("#rename-chat").click();
    await page.waitForFunction(
      () =>
        document.querySelector("#session-title").textContent === "合同通知讨论",
    );
    await page.locator(".source-button").first().click();
    await page.locator(".source-card").waitFor();
    assert.match(
      await page.locator(".source-meta").first().innerText(),
      /第 1 页/,
    );
    await page.locator("#chat-scroll").evaluate((e) => {
      e.scrollTop = 0;
    });
    await page.screenshot({ path: path.join(output, "desktop-1440.png") });
    mark("rename and answer-bound AST source page");
    const backgroundRun = await submit("切换时请继续完成这次回答。");
    await page.locator("#new-chat").click();
    await ready();
    const b = await selected();
    assert.notEqual(b, a);
    await input.fill("会话 B 保留的草稿");
    await page.waitForTimeout(4200);
    assert.equal(await page.locator(".message-assistant").count(), 0);
    assert.equal(await input.inputValue(), "会话 B 保留的草稿");
    await page.locator(`[data-session-id="${a}"]`).click();
    await waitDone();
    assert.equal(
      await page
        .locator(`.message-assistant[data-run-id="${backgroundRun}"]`)
        .count(),
      1,
    );
    mark("switch preserves background completion and separate draft");
    const refreshRun = await submit("刷新页面仍保留进行中的提问。");
    await page.reload();
    await ready();
    await waitDone();
    assert.equal(await selected(), a);
    assert.equal(
      await page
        .locator(`.message-assistant[data-run-id="${refreshRun}"]`)
        .count(),
      1,
    );
    assert.equal(
      (
        await (
          await context.request.get(`${base}/v1/chat-sessions/${a}/turns`)
        ).json()
      ).items.length,
      5,
    );
    mark("refresh resumes same Run with no duplicate submission");
    await submit("请开始一个可以停止的回答。");
    await page.locator("#cancel-button").click();
    await waitDone();
    assert.match(
      await page.locator(".message-assistant .message-body").last().innerText(),
      /已停止/,
    );
    mark("stop waits for durable cancelled state");
    await page.locator("#chat-scroll").evaluate((e) => {
      e.scrollTop = 0;
      e.dispatchEvent(new Event("scroll"));
    });
    await page.locator("#scroll-latest").waitFor();
    const before = await page
      .locator("#chat-scroll")
      .evaluate((e) => e.scrollTop);
    await input.fill("输入草稿不会抢走历史阅读位置");
    assert.equal(
      await page.locator("#chat-scroll").evaluate((e) => e.scrollTop),
      before,
    );
    await page.locator("#scroll-latest").click();
    assert(
      await page
        .locator("#chat-scroll")
        .evaluate((e) => e.scrollHeight - e.scrollTop - e.clientHeight < 80),
    );
    mark("older reading position and return to latest");
    // Browser key/composition events exercise the real event listeners.
    const turnsBeforeIME = await page.locator(".message-user").count();
    await input.evaluate((e) => {
      e.dispatchEvent(
        new CompositionEvent("compositionstart", { bubbles: true }),
      );
      e.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          isComposing: true,
          bubbles: true,
          cancelable: true,
        }),
      );
      e.dispatchEvent(
        new CompositionEvent("compositionend", { bubbles: true }),
      );
    });
    assert.equal(await page.locator(".message-user").count(), turnsBeforeIME);
    await input.fill("第一行");
    await input.press("Shift+Enter");
    await input.type("第二行");
    assert.equal(await input.inputValue(), "第一行\n第二行");
    mark("Chinese composition Enter guard and Shift+Enter newline");
    // Delay a real source response, then leave its session before it returns.
    let sourceRequested;
    const requested = new Promise((resolve) => {
      sourceRequested = resolve;
    });
    await page.route("**/sources", async (route) => {
      const response = await route.fetch();
      sourceRequested();
      await new Promise((resolve) => setTimeout(resolve, 1200));
      await route.fulfill({ response });
    });
    await page.locator(".source-button").first().click();
    await requested;
    await page.locator(`[data-session-id="${b}"]`).click();
    await page.waitForTimeout(1500);
    assert.equal(await page.locator(".source-card").count(), 0);
    assert.equal(await input.inputValue(), ""); // Draft text intentionally does not survive reload.
    await page.unroute("**/sources");
    mark("late source response cannot cross session boundary");
    // Keep final GETs pending briefly so four broken SSE attempts reach polling.
    let brokenStreams = 0,
      getCount = 0;
    await page.route("**/events", (route) => {
      brokenStreams++;
      return route.abort();
    });
    await page.route(/\/v1\/query-runs\/[^/]+$/, async (route) => {
      if (route.request().method() !== "GET") return route.continue();
      getCount++;
      const response = await route.fetch();
      const body = await response.json();
      if (brokenStreams < 4 && body.status === "completed")
        return route.fulfill({
          json: { ...body, status: "running", answer: null },
        });
      return route.fulfill({ response });
    });
    await submit("断线后自动同步回答。");
    await waitDone();
    assert(brokenStreams >= 4);
    assert(getCount >= 5);
    await page.unroute("**/events");
    await page.unroute(/\/v1\/query-runs\/[^/]+$/);
    mark("four SSE failures fall back to GET polling");
    await page.locator(`[data-session-id="${a}"]`).click();
    await waitDone();
    await page.locator(".source-button").first().click();
    await page.locator(".source-card").waitFor();
    assert(
      process.env.CHAT_SEEDED_DOCUMENT_ID,
      "explicit disposable seeded document ID required",
    );
    assert.equal(
      (
        await context.request.delete(
          `${base}/v1/documents/${process.env.CHAT_SEEDED_DOCUMENT_ID}`,
        )
      ).status(),
      204,
    );
    await page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".sources")].some((e) =>
        e.textContent.includes("来源暂不可用"),
      ),
    );
    assert.equal(await page.locator(".source-excerpt").count(), 0);
    mark("open source reauthorization removes deleted document excerpts");
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(
      await page.locator("#new-chat").evaluate((e) => {
        e.focus();
        return document.activeElement === e;
      }),
      false,
    );
    assert(
      await page.evaluate(() => document.documentElement.scrollWidth <= 390),
    );
    await page.locator("#open-sidebar").click();
    assert.equal(
      await page
        .locator("#new-chat")
        .evaluate((e) => document.activeElement === e),
      true,
    );
    await page.locator("#open-system").focus();
    await page.keyboard.press("Tab");
    assert.equal(
      await page
        .locator("#close-sidebar")
        .evaluate((e) => document.activeElement === e),
      true,
    );
    await page
      .locator("#session-sidebar")
      .evaluate((e) =>
        Promise.all(e.getAnimations().map((animation) => animation.finished)),
      );
    await page.screenshot({
      path: path.join(output, "mobile-sidebar-390.png"),
    });
    await page.keyboard.press("Escape");
    assert.equal(
      await page
        .locator("#open-sidebar")
        .evaluate((e) => document.activeElement === e),
      true,
    );
    await page.screenshot({ path: path.join(output, "mobile-390.png") });
    mark(
      "390px no horizontal overflow, hidden sidebar focus isolation, Tab trap and Escape restore",
    );
    assert.deepEqual(errors, []);
    mark("no uncaught browser JavaScript errors");
    await fs.writeFile(
      path.join(output, "results.json"),
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
