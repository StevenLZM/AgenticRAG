const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const watchdog = setTimeout(() => { console.error('Console flow did not complete'); process.exit(1); }, 3000);

async function scenario(route, completedAtSubmit = false) {
  const nodes = {};
  const element = () => ({textContent: '', children: [], classList: {add() {}, remove() {}},
    addEventListener() {}, replaceChildren() { this.children = []; },
    appendChild(child) { this.children.push(child); }, append(...children) { this.children.push(...children); }});
  let initialize;
  let releaseStream;
  const pause = new Promise(resolve => { releaseStream = resolve; });
  let streamed;
  const ready = new Promise(resolve => { streamed = resolve; });
  const completed = {run_id: 'run-1', status: 'completed', answer: {
    route, audited: route === 'chat' ? null : true,
    segments: [{kind: 'content', text: '你好！', evidence_ids: []}],
    evidence_parent_ids: [], citation_coverage: null
  }};
  let submitted;
  let initialText;
  const context = {
    document: {getElementById(id) { return nodes[id] ||= element(); }, createElement: element,
      addEventListener(type, callback) { initialize = callback; }},
    window: {setTimeout}, TextDecoder,
    fetch: async (url, options) => {
      const json = data => ({ok: true, json: async () => data});
      if (url === '/v1/query') {
        submitted = JSON.parse(options.body);
        initialText = nodes.answer.textContent;
        return json(completedAtSubmit ? completed : {run_id: 'run-1', status: 'queued'});
      }
      if (url.endsWith('/events')) {
        let count = 0;
        return {ok: true, body: {getReader() { return {async read() {
          if (count++ === 0) {
            const events = [
              {event_type: 'PROGRESS', summary: 'progress update'},
              {event_type: 'INTERNAL_TOOL_PAYLOAD', summary: 'private'},
              {event_type: 'QUERY_ROUTED', route, summary: 'completed'},
              {event_type: 'RETRIEVAL_DEGRADED', summary: 'degraded'}
            ];
            return {done: false, value: Buffer.from(events.map((e, i) => `id: ${i + 1}\ndata: ${JSON.stringify(e)}\n\n`).join(''))};
          }
          streamed();
          await pause;
          return {done: true};
        }}; }}};
      }
      if (url === '/v1/query-runs/run-1') return json(completed);
      if (url === '/v1/runtime/summary') return json({dependencies: {}, memory_available: true});
      if (url === '/v1/memories') return json({memories: []});
      return json({status: 'ready'});
    }
  };
  vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
  initialize();
  await context.window.submitQuery('你好');
  assert.equal(initialText, '正在处理消息…');
  await ready;
  assert.equal(submitted.wait_seconds, 0);
  const waiting = {chat: '正在生成聊天回复…', fast_rag: '正在检索资料…', research: '正在深入研究…'};
  assert.equal(nodes.answer.textContent, completedAtSubmit ? '你好！' : waiting[route]);
  assert.equal(nodes.timeline.children.length, 2);
  assert.ok(!nodes.timeline.children.some(n => n.textContent.includes('进度更新')));
  assert.equal(nodes['degradation-banner'].hidden, false);
  releaseStream();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(nodes.answer.textContent, '你好！');
  if (route === 'chat') {
    assert.match(nodes.evidence.textContent, /不适用/);
    assert.match(nodes.audit.textContent, /不适用/);
  }
}

(async () => {
  for (const route of ['chat', 'fast_rag', 'research']) await scenario(route);
  await scenario('chat', true);
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => clearTimeout(watchdog));
