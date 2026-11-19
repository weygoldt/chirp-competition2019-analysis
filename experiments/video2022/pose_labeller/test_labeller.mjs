// Logic test of labeller.html under node, with a minimal fake DOM (no browser needed).
//   node experiments/video2022/pose_labeller/test_labeller.mjs
// Simulates mouse and key events and checks what the page saves.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "labeller.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

const W = 2048, H = 1536, TOP = 31;
const state = {
  items: [
    {item_id: "a", image: "images/a.jpg", width: W, height: H, prefill: [{head: [100, 100], center: [150, 100], tail: [200, 100]}]},
    {item_id: "b", image: "images/b.jpg", width: W, height: H, prefill: []},
  ],
  labels: {}, labeller: "test",
};
const posted = [];
const handlers = {window: {}, cv: {}};
const el = () => ({textContent: "", innerHTML: "", style: {display: ""}});
const elements = {pos: el(), state: el(), next: el(), msg: el(), help: el()};
const ctx = new Proxy({}, {get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; }});
const cv = {
  width: 0, height: 0, getContext: () => ctx,
  getBoundingClientRect: () => ({left: 0, top: TOP}),
  addEventListener: (type, fn) => { handlers.cv[type] = fn; },
};
elements.cv = cv;
class FakeImage { set src(v) { this._src = v; this.width = W; this.height = H; setTimeout(() => this.onload && this.onload(), 0); } }
const sandbox = {
  document: {getElementById: id => elements[id]},
  innerWidth: 1600, innerHeight: 1231,  // canvas 1600 x 1200 -> scale 0.78125, image fills it
  addEventListener: (type, fn) => { handlers.window[type] = fn; },
  performance: {now: () => 1000},
  navigator: {sendBeacon: () => true},
  Image: FakeImage, Blob: class {},
  fetch: async (url, opts) => {
    if (url === "/api/session") return {json: async () => JSON.parse(JSON.stringify(state))};
    const body = JSON.parse(opts.body); posted.push(body);
    return {json: async () => ({ok: true, label: {...body, labeller: "test"}})};
  },
  setTimeout, console, JSON, Math,
};
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const tick = () => new Promise(r => setTimeout(r, 5));
const S = 1600 / W;  // view scale
const scr = (x, y) => ({clientX: x * S, clientY: y * S + TOP});
async function click(x, y) {
  handlers.cv.mousedown({button: 0, ...scr(x, y)});
  handlers.window.mouseup({target: cv, ...scr(x, y)});
  await tick();
}
async function key(k) { handlers.window.keydown({key: k, preventDefault() {}}); await tick(); await tick(); }

await tick(); await tick();
assert.match(elements.pos.innerHTML, /frame <b>1<\/b> \/ 2/);

// item a: keep SLEAP's fish, add a new one by three clicks
await click(1000, 800); await click(1050, 800); await click(1100, 800);
await key("Enter");
let p = posted.at(-1);
assert.equal(p.item_id, "a"); assert.equal(p.status, "done"); assert.equal(p.fish.length, 2);
assert.equal(p.fish[0].origin, "prefill");
assert.deepEqual(p.fish[0].head, [100, 100]);
assert.equal(p.fish[1].origin, "new");
assert.ok(Math.abs(p.fish[1].head[0] - 1000) < 1e-6 && Math.abs(p.fish[1].tail[1] - 800) < 1e-6);

// item b (opened by Enter): skip the head, click centre and tail; then a second fish: head only
assert.match(elements.pos.innerHTML, /frame <b>2<\/b>/);
await key("s"); await click(500, 500); await click(600, 500);
await key("n"); await click(1500, 300);
await key("Enter");
p = posted.at(-1);
assert.equal(p.item_id, "b"); assert.equal(p.fish.length, 2);
assert.equal(p.fish[0].head, null); assert.deepEqual(p.fish[0].center, [500, 500]);
assert.deepEqual(p.fish[1].head, [1500, 300]); assert.equal(p.fish[1].center, null);

// back to a: drag SLEAP's tail -> origin becomes prefill_edited; right click deletes the new fish's centre
await key("ArrowLeft");
assert.match(elements.pos.innerHTML, /frame <b>1<\/b>/);
handlers.cv.mousedown({button: 0, ...scr(200, 100)});
handlers.window.mousemove(scr(210, 120));
handlers.window.mouseup({target: cv, ...scr(210, 120)});
handlers.cv.contextmenu({preventDefault() {}, ...scr(1050, 800)});
await key("Enter");
p = posted.at(-1);
assert.equal(p.item_id, "a");
assert.equal(p.fish[0].origin, "prefill_edited"); assert.deepEqual(p.fish[0].tail.map(Math.round), [210, 120]);
assert.equal(p.fish[1].center, null);

// undo restores the deleted point; x marks a frame unusable
await key("ArrowRight"); await key("ArrowLeft");
handlers.cv.contextmenu({preventDefault() {}, ...scr(1000, 800)});
await key("u");
await key("x");
p = posted.at(-1);
assert.equal(p.status, "unusable"); assert.deepEqual(p.fish[1].head, [1000, 800]);
console.log(`labeller logic ok (${posted.length} saves)`);
