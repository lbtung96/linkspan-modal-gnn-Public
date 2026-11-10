// Page shell for the explainer figures: theme tokens, the framed card, legend and caption.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

// the kit as inline source (exports stripped), so the page draws with the build's code
export function kitSource() {
  return readFileSync(join(HERE, "iso.mjs"), "utf8").replace(/^export (const|function) /gm, "$1 ");
}
export function benchSource() {
  return readFileSync(join(HERE, "bench.mjs"), "utf8").replace(/^import .*$/gm, "").replace(/^export \{.*$/gm, "").replace(/^export (const|function) /gm, "$1 ");
}

export const CSS = `
:root{color-scheme:dark}
.iso{
  --page:#0c0c0c;--card:#141414;--text:#a8a8a8;--strong:#e8e8e8;--muted:#717171;
  --paper:#141414;--top:#1e1e1e;--s0:#0f0f0f;--s1:#131313;--s2:#171717;--s3:#1a1a1a;
  --lit-top:#2a2a2a;--lit-shade:#202020;
  --hi:#d6d6d6;--mid:#6c6c6c;--lo:#404040;--faint:#2b2b2b;--lit:#ffffff;
  --halo:#040404;--halo-o:.9;--dot:#e6e6e6;--cap:#262626;
  --accent:#ffffff;--trace:#cfcfcf;--trace2:#8a8a8a;--screen:#0d0f0e;
  --w:.6px;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
.iso[data-theme="light"]{
  color-scheme:light;
  --page:#f1f1ee;--card:#fbfbfa;--text:#55555a;--strong:#18181a;--muted:#8b8b90;
  --paper:#fbfbfa;--top:#ffffff;--s0:#d9d9d6;--s1:#e3e3e0;--s2:#ececea;--s3:#f4f4f2;
  --lit-top:#ffffff;--lit-shade:#efefec;
  --hi:#25252a;--mid:#87878c;--lo:#b2b2b6;--faint:#d3d3d5;--lit:#000000;
  --halo:#85857c;--halo-o:.3;--dot:#38383c;--cap:#e2e2df;
  --accent:#000000;--trace:#222226;--trace2:#8d8d92;--screen:#f3f4f1;
}
html,body{margin:0}
body.iso{background:var(--page);color:var(--text);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:900px;margin:0 auto;padding:28px 16px 48px}
.iso-svg{display:block;width:100%;height:auto;overflow:visible;user-select:none;-webkit-user-select:none}
.iso-svg path,.iso-svg circle{vector-effect:non-scaling-stroke}
.iso-fill{fill:var(--paper)}
.iso-shade,.iso-top,.iso-side{stroke-width:.35;stroke-linejoin:round}
.iso-shade[data-shade="0"]{fill:var(--s0);stroke:var(--s0)}
.iso-shade[data-shade="1"]{fill:var(--s1);stroke:var(--s1)}
.iso-shade[data-shade="2"]{fill:var(--s2);stroke:var(--s2)}
.iso-shade[data-shade="3"]{fill:var(--s3);stroke:var(--s3)}
.iso-side{fill:url(#cylG);stroke:none}
.iso-top{fill:var(--top);stroke:var(--top)}
.iso-solid[data-lit]>.iso-top{fill:var(--lit-top);stroke:var(--lit-top)}
.iso-solid[data-lit]>.iso-shade{fill:var(--lit-shade);stroke:var(--lit-shade)}
.iso-solid[data-lit]>.iso-edge{stroke:var(--lit)}
.iso-halo{fill:var(--halo);filter:blur(9px);opacity:var(--halo-o);transform:translateY(3px)}
.iso-line{fill:none;stroke:var(--lo);stroke-width:var(--w);stroke-linecap:round;stroke-linejoin:round}
.iso-line[data-tone="hi"]{stroke:var(--hi)}
.iso-line[data-tone="mid"]{stroke:var(--mid)}
.iso-line[data-tone="lo"]{stroke:var(--lo)}
.iso-line[data-tone="faint"]{stroke:var(--faint)}
.iso-line[data-tone="lit"]{stroke:var(--lit)}
.iso-dots circle{fill:var(--dot);opacity:.32}
.iso-dots[data-tone="hi"] circle{opacity:.85}
.iso-dots[data-tone="mid"] circle{opacity:.45}
.iso-dots[data-tone="lo"] circle{opacity:.22}
.iso-dots[data-tone="lit"] circle{fill:var(--lit);opacity:1}
.iso-face-text{fill:var(--mid);font-family:var(--mono);letter-spacing:.04em}
.iso-face-text[data-tone="lo"]{fill:var(--lo)}
.iso-face-text[data-tone="hi"]{fill:var(--hi)}
.tb-edge{fill:none;stroke:var(--mid);stroke-linecap:round;stroke-linejoin:round}
.tb-body{fill:none;stroke:var(--s2);stroke-linecap:round;stroke-linejoin:round}
.tb-shine{fill:none;stroke:var(--top);stroke-linecap:round;stroke-linejoin:round}
.iso-cap{fill:var(--cap);stroke:var(--mid);stroke-width:var(--w)}
.iso-tube[data-lit] .tb-edge{stroke:var(--lit)}
.screen-bg{fill:var(--screen);stroke:var(--lo);stroke-width:var(--w)}
.trace{fill:none;stroke:var(--trace);stroke-width:.8;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.trace2{fill:none;stroke:var(--trace2);stroke-width:.6;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.pulse{fill:var(--lit)}
.plate{position:relative;padding:52px 22px;background:var(--card);box-shadow:0 0 0 1px var(--mid);outline:none}
.plate:focus-visible{box-shadow:0 0 0 1px var(--mid),0 0 0 3px var(--hi)}
.corner{position:absolute;color:var(--muted);font-family:var(--mono);font-size:10.5px;line-height:1.2;letter-spacing:.04em;white-space:nowrap}
.corner[data-c="fig"]{top:18px;left:20px;color:var(--text)}
.corner[data-c="title"]{top:18px;right:20px;text-transform:uppercase;letter-spacing:.1em}
.corner[data-c="hint"]{bottom:18px;left:20px;text-transform:uppercase;letter-spacing:.1em}
.corner[data-c="readout"]{bottom:18px;right:20px;color:var(--text);font-variant-numeric:tabular-nums;text-align:right}
.controls{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:12px;font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.controls .grp{display:flex;gap:4px;align-items:center}
.controls button{font:inherit;color:var(--text);background:transparent;border:1px solid var(--lo);padding:3px 8px;border-radius:0;cursor:pointer}
.controls button[aria-pressed="true"]{color:var(--strong);border-color:var(--hi)}
.controls input[type=range]{width:160px;accent-color:var(--hi)}
.legend{display:grid;gap:10px;margin-top:14px;color:var(--muted);font-size:13px;line-height:1.55}
.keys{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:6px 18px}
.key{display:flex;gap:8px;align-items:flex-start}
.key svg{width:16px;height:10px;flex:none;margin-top:5px;overflow:visible}
.key svg path{fill:none;stroke:var(--hi);stroke-width:1;stroke-linecap:round}
.key[data-m="lit"] svg path{stroke:var(--lit)}
.key[data-m="mid"] svg path{stroke:var(--mid)}
.key[data-m="lo"] svg path{stroke:var(--lo)}
.key b{color:var(--text);font-weight:500;font-family:var(--mono);font-size:12px}
.caption{color:var(--text)}
.caption code,.key code{font-family:var(--mono);font-size:12px}
@media (max-width:560px){.plate{padding:46px 6px 74px;overflow:hidden}.plate .iso-svg{width:118%;margin:0 -9%}.corner[data-c="readout"]{bottom:16px;max-width:calc(100% - 40px);white-space:normal}.corner[data-c="hint"]{bottom:48px}}
`;

const keyMark = { lit: "M1 5h14", hi: "M1 5h14", mid: "M1 5h14", lo: "M1 5h14" };

export function page({ title, theme, fig, cardTitle, hint, readout, label, svg, controls, keys, caption, script, description }) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${title}</title><meta name="description" content="${description || label}">
<style>${CSS}</style></head>
<body class="iso" data-theme="${theme}">
<div class="wrap">
<figure style="margin:0">
<div class="plate" tabindex="0" role="application" aria-label="${label}">
<span class="corner" data-c="fig">${fig}</span><span class="corner" data-c="title">${cardTitle}</span>
${svg}
<span class="corner" data-c="hint">${hint}</span><span class="corner" data-c="readout" data-readout aria-live="polite">${readout}</span>
</div>
<div class="controls">${controls || ""}</div>
<figcaption class="legend">
<div class="keys">${keys.map((k) => `<div class="key" data-m="${k.mark}"><svg viewBox="0 0 16 10"><path d="${keyMark[k.mark] || keyMark.hi}"/></svg><span>${k.label}</span></div>`).join("")}</div>
<div class="caption">${caption}</div>
</figcaption>
</figure>
</div>
<script>${script}</script>
</body></html>`;
}
