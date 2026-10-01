"use strict";
const fs = require("node:fs"),
  path = require("node:path");
const { fromFile } = require("./chat_dom.cjs");
const dir = path.dirname(process.argv[2]),
  input = JSON.parse(fs.readFileSync(0, "utf8"));
const { createChatView, answerPresentation } = require(
  path.join(dir, "chat-view.js"),
);
const document = fromFile(path.join(dir, "index.html")),
  view = createChatView(document);
const run = { ...input.run, question: "测试问题" };
view.renderSession({ turns: new Map(), orderedRunIds: [] });
if (input.mode !== "load")
  view.updateTurn({ ...run, status: "running", answer: null });
view.updateTurn(run);
const text = document
  .getElementById("chat-messages")
  .children[1].querySelectorAll("div")[1].textContent;
process.stdout.write(
  JSON.stringify({
    rendered: { text, notice: answerPresentation(run) ? null : text },
  }),
);
