"use strict";
const assert = require("node:assert/strict"),
  path = require("node:path");
const { fromFile, flush } = require("./chat_dom.cjs");
const dir = process.argv[2];
const { createConsoleTools } = require(path.join(dir, "console-tools.js"));
async function main() {
  const document = fromFile(path.join(dir, "index.html"));
  const tools = createConsoleTools({
    document,
    fetchImpl: async (url) => {
      const bodies = {
        "/v1/runtime/summary": {
          runtime_config_snapshot_id: "snapshot-test",
          dependencies: { mysql: "available", redis: "unavailable" },
        },
        "/health/live": { status: "alive" },
        "/health/ready": {
          status: "unready",
          dependencies: { mysql: "available", redis: "unavailable" },
        },
      };
      assert(url in bodies);
      return new Response(JSON.stringify(bodies[url]), {
        status: url === "/health/ready" ? 503 : 200,
      });
    },
  });
  tools.open("system");
  await flush();
  assert.equal(
    document.getElementById("health-status").textContent,
    "服务尚未就绪",
  );
  assert.match(
    document.getElementById("health-grid").textContent,
    /redis：不可用/,
  );
  assert.equal(
    document.getElementById("snapshot-id").textContent,
    "snapshot-test",
  );
  tools.dispose();
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
