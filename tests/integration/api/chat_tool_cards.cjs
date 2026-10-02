"use strict";
const assert = require("node:assert/strict"), path = require("node:path");
const { fromFile } = require("./chat_dom.cjs");
const { createChatView, answerPresentation } = require(path.join(process.argv[2], "chat-view.js"));
const doc = fromFile(path.join(process.argv[2], "index.html"));
const view = createChatView(doc);
const turn = {
  run_id: "map-1", status: "completed", question: "附近哪里散步？",
  answer: {
    route: "chat", tool_audited: true,
    segments: [{kind:"content",text:"西湖，距中心 250 米。"}],
    external_sources: [{id:"s1",provider:"高德地图",tool_id:"mcp.amap_maps.maps_text_search",observed_at:"2026-10-02T00:00:00Z"}],
    cards: [{id:"c1",source_id:"s1",kind:"place",title:"西湖 <img onerror=alert(1)>",address:"南山路",location:"120.15,30.25",coordinate_system:"GCJ-02",distance_m:250,url:"https://uri.amap.com/marker?position=120.15%2C30.25&coordinate=gaode",details:{city:"杭州",category:"景点",raw:"never expose"}}],
  },
};
assert.equal(answerPresentation({...turn, answer:{...turn.answer,tool_audited:null}}), null);
assert(answerPresentation({...turn,answer:{...turn.answer,route:"research"}}));
assert.equal(answerPresentation({...turn,status:"running"}),null);
view.updateTurn(turn, false);
const list = doc.getElementById("chat-messages");
assert.equal(list.querySelectorAll("a").length, 1);
assert.equal(list.querySelectorAll("img").length, 0);
assert(list.textContent.includes("<img onerror=alert(1)>"));
assert(!list.textContent.includes("never expose"));
assert.equal(list.querySelector("a").getAttribute("rel"), "noopener noreferrer");
assert.equal(list.querySelector("a").getAttribute("target"), "_blank");
let details = list.querySelector("details");
assert(details);
assert.equal(details.open, false);
details.open = true;
view.updateTurn(turn, false);
assert.equal(list.querySelector("details"), details);
assert.equal(details.open, true);
// Reloading persisted JSON recreates safe cards, with collapsed details.
const saved = JSON.parse(JSON.stringify(turn));
view.renderSession({turns:new Map([[saved.run_id,saved]]),orderedRunIds:[saved.run_id],historyCursor:null});
assert.equal(list.querySelectorAll("a").length,1);
assert.equal(list.querySelector("details").open,false);
const poiOnly = JSON.parse(JSON.stringify(turn));
delete poiOnly.answer.cards[0].location;
delete poiOnly.answer.cards[0].coordinate_system;
poiOnly.answer.cards[0].poi_id = "B023B08WDR";
poiOnly.answer.cards[0].url = "https://uri.amap.com/marker?poiid=B023B08WDR&src=agentic_rag&callnative=0";
view.updateTurn(poiOnly,false);
assert.equal(list.querySelectorAll("a").length,1);
assert(list.querySelector("a").getAttribute("href").includes("poiid=B023B08WDR"));
for (const url of ["javascript:alert(1)","https://uri.amap.com.evil.com/marker", "https://user@uri.amap.com/marker", "http://uri.amap.com/marker", "https://uri.amap.com/redirect"]) {
  const unsafe = JSON.parse(JSON.stringify(turn));
  unsafe.answer.cards[0].url = url;
  view.updateTurn(unsafe,false);
  assert.equal(list.querySelectorAll("a").length,0,url);
}
// Mixed answers keep the document source button alongside map provenance.
view.updateTurn({...turn,answer:{...turn.answer,route:"research",audited:true,evidence_parent_ids:["p1"],segments:[{kind:"content",text:"文档事实",evidence_ids:["e1"]}]}},false);
assert.equal(list.querySelector('[data-action="sources"]').hidden,false);
assert.equal(list.querySelectorAll("a").length,1);
// Legacy messages need no new fields and leave no stale cards.
view.updateTurn({...turn,answer:{route:"chat",segments:[{kind:"content",text:"普通聊天"}]}},false);
assert.equal(list.querySelectorAll("a").length,0);
assert(list.textContent.includes("普通聊天"));
console.log("tool card DOM, safe links, mixed audits and persisted restoration passed");
