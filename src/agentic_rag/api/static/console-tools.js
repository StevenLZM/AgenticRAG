(function (root) {
  "use strict";
  function createConsoleTools({
    document,
    fetchImpl = (...args) => fetch(...args),
    onDocumentsChanged = () => {},
    timers = globalThis,
  }) {
    const byId = (id) => document.getElementById(id),
      drawer = byId("tools-drawer");
    let focusBefore = null,
      generation = 0,
      kind = null,
      disposed = false;
    const uploads = new Map();
    function textNode(tag, text, className) {
      const e = document.createElement(tag);
      e.textContent = text;
      e.className = className || "";
      return e;
    }
    async function request(url, options = {}) {
      const response = await fetchImpl(url, { cache: "no-store", ...options });
      if (!response.ok) throw new Error("服务暂时不可用，请稍后重试。");
      return response.status === 204 ? null : response.json();
    }
    function close() {
      generation++;
      kind = null;
      drawer.close();
      focusBefore?.focus();
    }
    async function loadMemories() {
      const stamp = generation;
      const list = byId("memory-list");
      list.replaceChildren(textNode("li", "正在加载记忆…"));
      try {
        const data = await request("/v1/memories");
        if (stamp !== generation || kind !== "memory") return;
        list.replaceChildren();
        if (!data.memories?.length)
          list.append(textNode("li", "暂无已保存的长期记忆。"));
        for (const memory of data.memories || []) {
          const item = document.createElement("li"),
            remove = textNode("button", "删除", "secondary");
          item.append(textNode("span", memory.text || memory.id), remove);
          remove.addEventListener("click", async () => {
            remove.disabled = true;
            try {
              await request(`/v1/memories/${encodeURIComponent(memory.id)}`, {
                method: "DELETE",
              });
              if (kind === "memory") await loadMemories();
            } catch (_) {
              if (stamp === generation)
                list.prepend(
                  textNode("li", "删除记忆失败，请重试。", "error-card"),
                );
              remove.disabled = false;
            }
          });
          list.append(item);
        }
      } catch (_) {
        if (stamp === generation)
          list.replaceChildren(
            textNode("li", "长期记忆服务暂不可用，请稍后重试。", "error-card"),
          );
      }
    }
    async function loadSystem() {
      const stamp = generation;
      byId("health-status").textContent = "正在检查系统…";
      try {
        const [summary, live, ready] = await Promise.all([
          request("/v1/runtime/summary"),
          request("/health/live"),
          request("/health/ready"),
        ]);
        if (stamp !== generation) return;
        byId("health-status").textContent =
          ready.status === "ready" && live.status ? "服务就绪" : "服务尚未就绪";
        byId("snapshot-id").textContent =
          summary.runtime_config_snapshot_id || "暂无运行快照";
        byId("health-grid").replaceChildren();
        for (const [name, value] of Object.entries(
          summary.dependencies || {},
        )) {
          byId("health-grid").append(
            textNode(
              "div",
              `${name}：${value === "available" ? "可用" : "不可用"}`,
              "health-chip",
            ),
          );
        }
      } catch (_) {
        if (stamp === generation) {
          byId("health-status").textContent = "系统状态暂不可用，请稍后重试。";
          byId("snapshot-id").textContent = "";
          byId("health-grid").replaceChildren();
        }
      }
    }
    function renderUploads() {
      const list = byId("upload-list");
      list.replaceChildren();
      for (const [id, upload] of uploads) {
        const item = document.createElement("li"),
          remove = textNode("button", "删除", "secondary");
        item.append(
          textNode("span", `${upload.filename} · ${upload.status}`),
          remove,
        );
        remove.addEventListener("click", async () => {
          remove.disabled = true;
          try {
            await request(`/v1/documents/${encodeURIComponent(id)}`, {
              method: "DELETE",
            });
            uploads.delete(id);
            renderUploads();
            onDocumentsChanged();
          } catch (_) {
            byId("ingestion-status").textContent = "删除失败，请稍后重试。";
            remove.disabled = false;
          }
        });
        list.append(item);
      }
    }
    async function pollJob(jobId, documentId) {
      for (let n = 0; n < 120 && !disposed; n++) {
        await new Promise((r) => timers.setTimeout(r, 1000));
        if (disposed) return;
        try {
          const job = await request(
            `/v1/ingestion-jobs/${encodeURIComponent(jobId)}`,
          );
          const upload = uploads.get(documentId);
          if (!upload) return;
          upload.status = job.status;
          renderUploads();
          byId("ingestion-status").textContent = `入库状态：${job.status}`;
          if (
            ["completed", "failed", "cancelled", "quarantined"].includes(
              job.status,
            )
          ) {
            onDocumentsChanged();
            return;
          }
        } catch (_) {
          byId("ingestion-status").textContent =
            "入库状态暂不可用，后台任务继续执行。";
          return;
        }
      }
    }
    byId("document-upload").addEventListener("submit", async (event) => {
      event.preventDefault();
      const file = byId("document-file").files?.[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      byId("ingestion-status").textContent = "正在上传…";
      try {
        const job = await request("/v1/documents", {
          method: "POST",
          body: form,
        });
        uploads.set(job.document_id, {
          filename: file.name,
          status: job.status,
        });
        renderUploads();
        byId("ingestion-status").textContent = `入库状态：${job.status}`;
        onDocumentsChanged();
        if (
          !["completed", "failed", "cancelled", "quarantined"].includes(
            job.status,
          )
        )
          void pollJob(job.job_id, job.document_id);
      } catch (_) {
        byId("ingestion-status").textContent =
          "上传失败，请检查文件格式并重试。";
      }
    });
    byId("close-tools").addEventListener("click", close);
    drawer.addEventListener("cancel", (event) => {
      event.preventDefault();
      close();
    });
    byId("reload-memories").addEventListener(
      "click",
      () => void loadMemories(),
    );
    return {
      open(next) {
        if (!["documents", "memory", "system"].includes(next)) return;
        generation++;
        kind = next;
        focusBefore = document.activeElement;
        for (const value of ["documents", "memory", "system"])
          byId(`${value}-tool`).hidden = value !== next;
        byId("tools-title").textContent = {
          documents: "文档知识库",
          memory: "长期记忆",
          system: "系统状态",
        }[next];
        if (!drawer.open) drawer.showModal();
        byId("close-tools").focus();
        if (next === "memory") void loadMemories();
        if (next === "system") void loadSystem();
      },
      close,
      dispose() {
        disposed = true;
        generation++;
      },
    };
  }
  const api = { createConsoleTools };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AgenticRagConsoleTools = api;
})(typeof window !== "undefined" ? window : globalThis);
