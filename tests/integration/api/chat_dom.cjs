"use strict";
const fs = require("node:fs");
class Element {
  constructor(tag, doc) {
    this.tagName = tag.toUpperCase();
    this.doc = doc;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.dataset = {};
    this._text = "";
    this.hidden = false;
    this.value = "";
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 2000;
    this.clientHeight = 600;
    this.open = false;
    this.classes = new Set();
    this.classList = {
      add: (...x) => x.forEach((v) => this.classes.add(v)),
      remove: (...x) => x.forEach((v) => this.classes.delete(v)),
      contains: (x) => this.classes.has(x),
      toggle: (x, on) => {
        const yes = on ?? !this.classes.has(x);
        yes ? this.classes.add(x) : this.classes.delete(x);
        return yes;
      },
    };
  }
  set textContent(x) {
    this._text = String(x ?? "");
    this.children = [];
  }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  set innerHTML(v) {
    throw new Error("Unsafe innerHTML: " + v);
  }
  append(...nodes) {
    for (const n of nodes) {
      n.parentElement = this;
      this.children.push(n);
    }
  }
  appendChild(n) {
    this.append(n);
    return n;
  }
  prepend(...nodes) {
    for (const n of nodes) n.parentElement = this;
    this.children.unshift(...nodes);
  }
  replaceChildren(...nodes) {
    this.children = [];
    this._text = "";
    this.append(...nodes);
  }
  remove() {
    if (this.parentElement)
      this.parentElement.children = this.parentElement.children.filter(
        (n) => n !== this,
      );
  }
  setAttribute(k, v) {
    this.attributes[k] = String(v);
    if (k === "id") this.id = String(v);
    if (k.startsWith("data-"))
      this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] =
        String(v);
  }
  getAttribute(k) {
    return this.attributes[k] ?? null;
  }
  removeAttribute(k) {
    delete this.attributes[k];
  }
  addEventListener(k, fn) {
    (this.listeners[k] ??= []).push(fn);
  }
  removeEventListener(k, fn) {
    this.listeners[k] = (this.listeners[k] || []).filter((f) => f !== fn);
  }
  dispatchEvent(e) {
    e.target ??= this;
    e.currentTarget = this;
    e.preventDefault ??= () => {
      e.defaultPrevented = true;
    };
    for (const fn of this.listeners[e.type] || []) fn(e);
    if (e.bubbles && this.parentElement) this.parentElement.dispatchEvent(e);
    return !e.defaultPrevented;
  }
  click() {
    this.dispatchEvent({ type: "click", bubbles: true });
  }
  focus() {
    this.doc.activeElement = this;
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
    this.dispatchEvent({ type: "close" });
  }
  matches(sel) {
    if (sel.startsWith("#")) return this.id === sel.slice(1);
    const attr = sel.match(/\[([\w-]+)(?:="([^"]*)")?\]/);
    const tag = sel.match(/^[a-z]+/);
    return (
      (!tag || this.tagName === tag[0].toUpperCase()) &&
      (!attr ||
        (this.getAttribute(attr[1]) !== null &&
          (attr[2] === undefined || this.getAttribute(attr[1]) === attr[2])))
    );
  }
  querySelectorAll(sel) {
    const found = [];
    for (const c of this.children) {
      if (sel.split(",").some((s) => c.matches(s.trim()))) found.push(c);
      found.push(...c.querySelectorAll(sel));
    }
    return found;
  }
  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }
  closest(sel) {
    return this.matches(sel) ? this : this.parentElement?.closest(sel) || null;
  }
  contains(node) {
    return node === this || this.children.some((c) => c.contains(node));
  }
  requestSubmit() {
    this.dispatchEvent({ type: "submit" });
  }
}
function documentFor(html) {
  const document = {
    listeners: {},
    hidden: false,
    activeElement: null,
    visibilityState: "visible",
  };
  document.body = new Element("body", document);
  document.createElement = (tag) => new Element(tag, document);
  document.getElementById = (id) => document.body.querySelector(`#${id}`);
  document.querySelectorAll = (sel) => document.body.querySelectorAll(sel);
  document.addEventListener = Element.prototype.addEventListener;
  document.removeEventListener = Element.prototype.removeEventListener;
  document.dispatchEvent = Element.prototype.dispatchEvent;
  for (const m of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const el = document.createElement(m[1]);
    el.id = m[2];
    document.body.append(el);
  }
  return document;
}
const flush = async (n = 12) => {
  for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r));
};
module.exports = {
  Element,
  documentFor,
  flush,
  fromFile: (path) => documentFor(fs.readFileSync(path, "utf8")),
};
