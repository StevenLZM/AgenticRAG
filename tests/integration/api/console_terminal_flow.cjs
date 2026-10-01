const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const {run, mode, events = []} = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = {};
const element = () => ({textContent: '', hidden: true, children: [],
  classList: {add() {}, remove() {}}, addEventListener() {},
  replaceChildren() { this.children = []; },
  appendChild(child) { this.children.push(child); }});
let initialize;
let releaseStream;
let streamed;
const pause = new Promise(resolve => { releaseStream = resolve; });
const ready = new Promise(resolve => { streamed = resolve; });
const context = {
  document: {getElementById(id) { return nodes[id] ||= element(); }, createElement: element,
    addEventListener(_type, callback) { initialize = callback; }},
  window: {setTimeout}, TextDecoder,
  fetch: async (url, options) => {
    const json = data => ({ok: true, json: async () => data});
    if (url === '/v1/query') {
      assert.equal(options.method, 'POST');
      return json(mode === 'sync' ? run : {run_id: run.run_id, status: 'queued'});
    }
    if (url === `/v1/query-runs/${run.run_id}/events`) {
      let delivered = false;
      return {ok: true, body: {getReader() { return {async read() {
        if (!delivered) {
          delivered = true;
          const replay = [{event_type: 'QUERY_ROUTED', route: run.answer.route}, ...events];
          return {done: false, value: Buffer.from(replay.map((event, i) => (
            `id: ${i + 1}\ndata: ${JSON.stringify(event)}\n\n`
          )).join(''))};
        }
        streamed();
        await pause;
        return {done: true};
      }}; }}};
    }
    if (url === `/v1/query-runs/${run.run_id}`) return json(run);
    if (url === '/v1/runtime/summary') return json({dependencies: {}, memory_available: true});
    if (url === '/v1/memories') return json({memories: []});
    if (url === '/health/live' || url === '/health/ready') return json({status: 'ready'});
    throw new Error(`Unexpected request: ${url}`);
  }
};
const view = () => ({text: nodes.answer.textContent,
  notice: nodes['degradation-banner'].hidden ? null : nodes['degradation-banner'].textContent,
  evidence: nodes.evidence.textContent, audit: nodes.audit.textContent});

(async () => {
  vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), context);
  initialize();
  const result = {};
  if (mode !== 'load') {
    await context.window.submitQuery('今天北京天气怎么样');
    await ready;
    if (mode === 'sync') result.initial = view();
    releaseStream();
    await new Promise(resolve => setImmediate(resolve));
    result.afterStream = view();
  }
  await context.window.loadRun(run.run_id);
  result.reloaded = view();
  process.stdout.write(JSON.stringify(result));
})().catch(error => { console.error(error); process.exitCode = 1; });
