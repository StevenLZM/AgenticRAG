"use strict";
const assert = require("node:assert/strict"),
  path = require("node:path");
const { fromFile } = require("./chat_dom.cjs");
const { createChatView, answerPresentation } = require(
  path.join(process.argv[2], "chat-view.js"),
);
const doc = fromFile(path.join(process.argv[2], "index.html")),
  view = createChatView(doc);
const turn = (i) => ({
  run_id: `run-${i}`,
  question: `问${i}`,
  status: "completed",
  created_at: `2026-10-01T00:00:00.00000${i}Z`,
  source_status: "none",
  answer: { route: "chat", segments: [{ kind: "content", text: `答${i}` }] },
});
view.renderSession({
  turns: new Map([1, 2, 3].map((i) => [`run-${i}`, turn(i)])),
  orderedRunIds: ["run-1", "run-2", "run-3"],
  historyCursor: null,
});
assert.equal(doc.getElementById("chat-messages").children.length, 6);
const scroll = doc.getElementById("chat-scroll");
scroll.scrollTop = 100;
scroll.dispatchEvent({ type: "scroll" });
view.updateTurn({
  ...turn(3),
  answer: {
    route: "chat",
    segments: [{ kind: "content", text: "<img onerror=alert(1)>" }],
  },
});
assert.equal(scroll.scrollTop, 100);
assert.equal(
  doc.getElementById("chat-messages").children[1].textContent.includes("答1"),
  true,
);
assert.equal(
  doc.getElementById("chat-messages").querySelectorAll("img").length,
  0,
);
assert(
  doc
    .getElementById("chat-messages")
    .textContent.includes("<img onerror=alert(1)>"),
);
assert.equal(doc.getElementById("timeline"), null);
view.renderSources("run-3", {
  status: "available",
  items: [
    {
      filename: "<img onerror=x>",
      heading_path: ["章节"],
      excerpt: "old content",
      version_status: "historical",
    },
  ],
});
assert.equal(scroll.scrollTop, 100);
assert.equal(answerPresentation({ ...turn(1), status: "running" }), null);
assert.equal(
  answerPresentation({
    ...turn(1),
    answer: {
      route: "research",
      segments: [{ kind: "content", text: "private" }],
    },
  }),
  null,
);
view.setComposer({ draft: "😀".repeat(32000), canSend: true, canStop: false });
assert.equal(doc.getElementById("send-button").disabled, false);
view.setComposer({ draft: "😀".repeat(32001), canSend: true, canStop: false });
assert.equal(doc.getElementById("send-button").disabled, true);
console.log("chat view contracts passed");
const list = doc.getElementById("chat-messages");
Object.defineProperty(scroll, "scrollHeight", {
  configurable: true,
  get: () => 1000 + list.children.length * 120,
});
scroll.scrollTop = 100;
view.prependTurns([turn(0)]);
assert.equal(scroll.scrollTop, 340);
assert.equal(list.children.length, 8);
