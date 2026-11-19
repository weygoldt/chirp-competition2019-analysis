// Logic test of labeller.html under node, with a minimal fake DOM (no browser needed).
//   node experiments/video2022/pose_labeller/test_labeller.mjs
// Simulates mouse and key events on a 7-point skeleton and checks what the page saves.
import fs from "node:fs";
import vm from "node:vm";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, "labeller.html"), "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

const W = 2048, H = 1536, TOP = 31;
const NODES = ["snout", "spine1", "spine2", "spine3", "tail", "left_fin", "right_fin"];
const state = {
  items: [
    {item_id: "a", image: "images/a.jpg", width: W, height: H,
     prefill: [{points: {snout: [100, 100], spine1: [150, 100], tail: [300, 100]}, origin: ""}]},
    {item_id: "b", image: "images/b.jpg", width: W, height: H, prefill: []},
  ],
  labels: {}, labeller: "test",
  skeleton: {nodes: NODES, edges: [["snout", "spine1"]], colors: {}, note: "test"},
};
const posted = [];
const handlers = {window: {}, cv: {}};
const el = () => ({textContent: "", innerHTML: "", style: {display: ""}});
const elements = {pos: el(), state: el(), next: el(), msg: el(), help: el(), nodes: el(), note: el()};
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
  setTimeout, console, JSON, Math, Object,
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
assert.match(elements.nodes.innerHTML, /right_fin/);

// item a: keep the suggested fish, add a new one with all 7 points
for (let i = 0; i < 7; i++) await click(1000 + 20 * i, 800);
await key("Enter");
let p = posted.at(-1);
assert.equal(p.item_id, "a"); assert.equal(p.status, "done"); assert.equal(p.fish.length, 2);
assert.equal(p.fish[0].origin, "prefill");
assert.deepEqual(Object.keys(p.fish[0].points), ["snout", "spine1", "tail"]);
assert.equal(p.fish[1].origin, "new");
assert.deepEqual(Object.keys(p.fish[1].points), NODES);
assert.ok(Math.abs(p.fish[1].points.right_fin[0] - 1120) < 1e-6);

// item b: skip the snout, then spine1..tail; skip both fins with s s
assert.match(elements.pos.innerHTML, /frame <b>2<\/b>/);
await key("s");
for (let i = 0; i < 4; i++) await click(500 + 30 * i, 500);
await key("s"); await key("s");
// second fish: snout only, then n
await key("n"); await click(1500, 300);
await key("Enter");
p = posted.at(-1);
assert.equal(p.item_id, "b"); assert.equal(p.fish.length, 2);
assert.deepEqual(Object.keys(p.fish[0].points), ["spine1", "spine2", "spine3", "tail"]);
assert.deepEqual(p.fish[0].points.spine1, [500, 500]);
assert.deepEqual(Object.keys(p.fish[1].points), ["snout"]);

// back to a: click on the suggested fish's snout, then fill its missing points (spine2, spine3, then left fin)
await key("ArrowLeft");
assert.match(elements.pos.innerHTML, /frame <b>1<\/b>/);
await click(100, 100);           // selects fish 1 (no move)
await click(200, 105); await click(250, 105);    // spine2, spine3
await click(160, 120);           // left_fin
// drag its tail, delete the new fish's spine1 with a right click
handlers.cv.mousedown({button: 0, ...scr(300, 100)});
handlers.window.mousemove(scr(310, 120));
handlers.window.mouseup({target: cv, ...scr(310, 120)});
handlers.cv.contextmenu({preventDefault() {}, ...scr(1020, 800)});
await key("Enter");
p = posted.at(-1);
assert.equal(p.item_id, "a");
assert.equal(p.fish[0].origin, "prefill_edited");
assert.deepEqual(p.fish[0].points.spine2, [200, 105]); assert.deepEqual(p.fish[0].points.left_fin, [160, 120]);
assert.deepEqual(p.fish[0].points.tail.map(Math.round), [310, 120]);
assert.equal(p.fish[0].points.right_fin, undefined);
assert.equal(p.fish[1].points.spine1, undefined);

// a click far from the active fish starts a new fish instead of placing the active fish's next point
await key("ArrowRight");
await click(1500, 300);          // select fish 2 of item b (snout only)
await click(1540, 310);          // its spine1 (near)
await click(400, 1200);          // far away: a new fish
await key("Enter");
p = posted.at(-1);
assert.equal(p.item_id, "b"); assert.equal(p.fish.length, 3);
assert.deepEqual(Object.keys(p.fish[1].points), ["snout", "spine1"]);
assert.deepEqual(p.fish[2].points, {snout: [400, 1200]});
await key("ArrowLeft");

// undo restores a deleted point; x marks a frame unusable
await key("ArrowRight"); await key("ArrowLeft");
handlers.cv.contextmenu({preventDefault() {}, ...scr(1000, 800)});
await key("u");
await key("x");
p = posted.at(-1);
assert.equal(p.status, "unusable"); assert.deepEqual(p.fish[1].points.snout, [1000, 800]);
console.log(`labeller logic ok (${posted.length} saves)`);
