/* Same-origin chat transport. Observation never cancels a server task. */
(function (root) {
  "use strict";
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);
  const PHASES = new Set(["processing", "retrieving", "researching", "auditing"]);
  class HttpError extends Error {
    constructor(status, errorCode, location) {
      super(`HTTP ${status}`);
      this.status = status; this.errorCode = errorCode; this.location = location;
    }
  }
  const part = value => encodeURIComponent(value);
  function createChatApi({fetchImpl = (...args) => fetch(...args), timers = globalThis} = {}) {
    async function request(path, {method = "GET", body, signal} = {}) {
      const options = {method, signal, cache: "no-store", headers: {Accept: "application/json"}};
      if (body !== undefined) { options.body = JSON.stringify(body); options.headers["Content-Type"] = "application/json"; }
      const response = await fetchImpl(path, options);
      if (!response.ok) {
        let value = {};
        try { value = await response.json(); } catch (_) { /* malformed error bodies stay private */ }
        throw new HttpError(response.status, typeof value.error_code === "string" ? value.error_code : "REQUEST_FAILED", response.headers.get("Location"));
      }
      return response.status === 204 ? null : response.json();
    }
    function page(path, options = {}) {
      const params = new URLSearchParams();
      if (options.cursor) params.set("cursor", options.cursor);
      if (options.limit) params.set("limit", options.limit);
      return request(path + (params.size ? `?${params}` : ""), {signal: options.signal});
    }
    const sessionPath = id => `/v1/chat-sessions/${part(id)}`;
    const runPath = id => `/v1/query-runs/${part(id)}`;
    function delay(ms, signal) {
      return new Promise((resolve, reject) => {
        if (signal?.aborted) { reject(new DOMException("Aborted", "AbortError")); return; }
        const stop = () => { timers.clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); };
        const timer = timers.setTimeout(() => { signal?.removeEventListener("abort", stop); resolve(); }, ms);
        signal?.addEventListener("abort", stop, {once: true});
      });
    }
    const api = {
      createSession: (key, signal) => request("/v1/chat-sessions", {method:"POST", body:{creation_request_id:key}, signal}),
      listSessions: options => page("/v1/chat-sessions", options),
      getSession: (id, signal) => request(sessionPath(id), {signal}),
      renameSession: (id, title, signal) => request(sessionPath(id), {method:"PATCH", body:{title}, signal}),
      deleteSession: (id, signal) => request(sessionPath(id), {method:"DELETE", signal}),
      listTurns: (id, options) => page(`${sessionPath(id)}/turns`, options),
      submitTurn: (id, question, key, signal) => request(`${sessionPath(id)}/turns`, {method:"POST", body:{query:question,client_request_id:key}, signal}),
      findSubmission: (id, key, signal) => request(`${sessionPath(id)}/submissions/${part(key)}`, {signal}),
      getSources: (id, runId, signal) => request(`${sessionPath(id)}/turns/${part(runId)}/sources`, {signal}),
      getRun: (id, signal) => request(runPath(id), {signal}),
      cancelRun: (id, signal) => request(`${runPath(id)}/cancel`, {method:"POST", signal})
    };
    api.watchRun = async ({runId, cursor = 0, signal, onPhase = () => {}, onRun = () => {}, onConnection = () => {}}) => {
      let failures = 0, finished = false, polling = false, initial = true;
      let lastId = Number.isSafeInteger(cursor) && cursor >= 0 ? cursor : 0;
      const check = async () => {
        const run = await api.getRun(runId, signal);
        if (signal?.aborted) return;
        if (run.run_id !== runId) throw new Error("Run identity mismatch");
        finished = TERMINAL.has(run.status);
        await onRun(run);
      };
      const stopError = error => error instanceof HttpError && [400,401,403,404,410,422].includes(error.status);
      async function consume() {
        const response = await fetchImpl(`${runPath(runId)}/events`, {signal, cache:"no-store",
          headers:{Accept:"text/event-stream", "Last-Event-ID":String(lastId)}});
        if (!response.ok) throw new HttpError(response.status, "STREAM_UNAVAILABLE", null);
        if (!response.body) throw new Error("Missing event stream");
        const reader = response.body.getReader(), decoder = new TextDecoder();
        let buffer = "";
        try {
          while (!signal?.aborted && !finished) {
            const {value, done} = await reader.read();
            buffer += decoder.decode(value, {stream: !done});
            buffer = buffer.replace(/\r\n/g, "\n");
            if (buffer.length > 1024 * 1024) throw new Error("Event frame too large");
            let index;
            while ((index = buffer.indexOf("\n\n")) !== -1) {
              const frame = buffer.slice(0,index); buffer = buffer.slice(index+2);
              if (frame.startsWith(":")) { failures = 0; continue; }
              let id = null, type = "", data = [];
              for (const line of frame.split("\n")) {
                if (line.startsWith("id:")) id = Number(line.slice(3).trim());
                if (line.startsWith("event:")) type = line.slice(6).trim();
                if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
              }
              let payload;
              try { payload = JSON.parse(data.join("\n")); } catch (_) { continue; }
              if (!Number.isSafeInteger(id) || id <= lastId || payload?.run_id !== runId) continue;
              failures = 0; lastId = id; onConnection("connected");
              if (type === "QUERY_PHASE_CHANGED" && PHASES.has(payload.phase)) await onPhase({run_id:runId, id, phase:payload.phase});
              if (["ANSWER_FINALIZED", "RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED"].includes(type)) await check();
              if (finished || signal?.aborted) return;
            }
            if (done) return;
          }
        } finally { try { await reader.cancel(); } catch (_) { /* connection already closed */ } reader.releaseLock(); }
      }
      try {
        while (!signal?.aborted && !finished) {
          if (initial || polling) {
            initial = false;
            try { await check(); } catch (error) { if (stopError(error)) throw error; if (signal?.aborted) return; }
          }
          if (finished || signal?.aborted) return;
          if (polling) { onConnection("polling"); await delay(2000, signal); continue; }
          try { await consume(); } catch (error) { if (stopError(error)) throw error; if (signal?.aborted) return; }
          if (finished || signal?.aborted) return;
          // EOF is a transport fact, never a durable completion signal.
          try { await check(); } catch (error) { if (stopError(error)) throw error; if (signal?.aborted) return; }
          if (finished || signal?.aborted) return;
          onConnection("reconnecting");
          if (failures < 3) await delay([500,1000,2000][failures++], signal);
          else { polling = true; await delay(2000, signal); }
        }
      } catch (error) { if (!signal?.aborted) throw error; }
    };
    return api;
  }
  const exported = {createChatApi, HttpError};
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  else root.AgenticRagChatApi = exported;
})(typeof window !== "undefined" ? window : globalThis);
