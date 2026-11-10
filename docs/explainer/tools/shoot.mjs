// Screenshots of a built page in headless Chromium (Playwright).
//   node tools/shoot.mjs page.html out.png [--w 900] [--scale 1] [--eval "js"] [--wait 1500] [--clip x,y,w,h] [--errors]
// --eval runs after load (e.g. "__fig.hold(); __fig.setStage(2)"); several --eval/--wait pairs run in order.
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");
import { resolve } from "node:path";

const args = process.argv.slice(2);
const [file, out] = args;
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const steps = [];
for (let i = 2; i < args.length; i++) {
  if (args[i] === "--eval") steps.push({ eval: args[++i] });
  else if (args[i] === "--wait") steps.push({ wait: +args[++i] });
}
const w = +opt("--w", 900), scale = +opt("--scale", 1), clip = opt("--clip", null);
const browser = await chromium.launch();
const pg = await browser.newPage({ viewport: { width: w, height: 1000 }, deviceScaleFactor: scale, reducedMotion: args.includes("--still") ? "reduce" : "no-preference" });
const errors = [];
pg.on("pageerror", (e) => errors.push(String(e)));
pg.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await pg.goto("file://" + resolve(file));
await pg.waitForTimeout(600);
for (const s of steps) {
  if (s.eval) await pg.evaluate(s.eval);
  if (s.wait) await pg.waitForTimeout(s.wait);
}
const shotOpt = { path: out, fullPage: !clip };
if (clip) { const [x, y, cw, ch] = clip.split(",").map(Number); shotOpt.clip = { x, y, width: cw, height: ch }; }
await pg.screenshot(shotOpt);
if (errors.length) console.log("ERRORS:\n" + errors.join("\n"));
else console.log("ok", out);
await browser.close();
