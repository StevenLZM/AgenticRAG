"use strict";
const assert = require("node:assert/strict"), path = require("node:path");
const {createChatApi} = require(path.join(process.argv[2], "chat-api.js"));
const {answerPresentation} = require(path.join(process.argv[2], "chat-view.js"));
const encoder = new TextEncoder();
const base = {run_id:"r", thread_id:"s", status:"running", answer_protocol_version:"answer-v2"};
const answer = {protocol_version:"answer-v2", epoch:1, seq:0, text:"", stream_status:"frozen", faithfulness:"pending", mode:"full", evidence_ids:[]};
let streamController, observedPrefix;
const prefix = new Promise(resolve => { observedPrefix = resolve; });
const updates = [];
const terminal = {...base, status:"completed", answer:{...answer, seq:3, text:"first second", stream_status:"completed", faithfulness:"failed"}};
function frame(seq, type, data) { return encoder.encode(`id: 1:${seq}\nevent: ${type}\ndata: ${JSON.stringify({run_id:"r",epoch:1,seq,...data})}\n\n`); }
const api = createChatApi({fetchImpl:async(url, options)=> {
  if (!url.endsWith("events")) return new Response(JSON.stringify({...base,answer}));
  assert.equal(options.headers["Last-Event-ID"], "1:0");
  return new Response(new ReadableStream({start(c){ streamController=c; c.enqueue(frame(1,"answer.delta",{text:"first",mode:"full"})); }}));
}});
const task = api.watchRun({runId:"r", onRun:run=>{updates.push(run);if(run.answer?.text==="first")observedPrefix();}});
(async()=>{
  await prefix;
  assert.equal(updates.at(-1).status,"running");
  assert.match(answerPresentation(updates.at(-1)).text,/核验中/);
  streamController.enqueue(frame(1,"answer.delta",{text:"duplicate"}));
  streamController.enqueue(frame(2,"answer.delta",{text:" second"}));
  streamController.enqueue(frame(3,"answer.completed",{run:terminal}));
  streamController.close();
  await task;
  assert.equal(updates.at(-1).answer.text,"first second");
  assert.match(answerPresentation(updates.at(-1)).text,/核验未通过/);
  assert(!updates.some(r=>r.answer?.text?.includes("duplicate")));
  console.log("supervisor streaming browser contracts passed");
})().catch(error=>{console.error(error);process.exitCode=1;});
