"""The style sheet and the player script of the search page (:mod:`schemagraph.agent.viz`).

Both are constants: the page never builds CSS or script from data. The script reads the replay
steps from a JSON data block, and changes the page only by toggling classes, setting
``textContent`` and opening disclosures, so no recorded string (SQL, prompts, model names) is
ever parsed as HTML. The page's Content-Security-Policy admits exactly this script, by its
SHA-256 hash.
"""

from __future__ import annotations

# Milliseconds per replay step at 1x speed.
STEP_MS = 1400

CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1f24; --muted: #5b6470; --line: #c3c9d1; --panel: #f3f4f6;
  --accent: #b35900; --good: #1a7f37; --bad: #cf222e; --bar: #7d8590; --code: #f6f8fa;
  --select: #8250df; --backup: #1a7f37;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e6e8eb; --muted: #9aa4b1; --line: #3b424c; --panel: #1b1f26;
    --accent: #ffb454; --good: #56d364; --bad: #ff7b72; --bar: #8b949e; --code: #161b22;
    --select: #c297ff; --backup: #56d364;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 96rem; padding: 16px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; overflow-wrap: anywhere;
}
a { color: inherit; }
h1 { font-size: 1.3rem; margin: 0 0 .5rem; }
h2 { font-size: 1.1rem; margin: 1.5rem 0 .5rem; }
h3 { font-size: 1rem; margin: 0 0 .5rem; }
.facts { display: grid; grid-template-columns: max-content 1fr; gap: .15rem 1rem; margin: 0; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; }
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; }
th, td { text-align: left; padding: .2rem .75rem .2rem 0; vertical-align: top; }
th { color: var(--muted); font-weight: 600; white-space: nowrap; }
.legend { margin: .5rem 0 1rem; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 50%;
  border: 1px solid var(--fg); vertical-align: -1px; margin-right: .4rem; }
.key { display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; color: var(--muted); margin: .5rem 0; }
.key svg { vertical-align: -3px; margin-right: .3rem; }
.key .node { stroke: var(--fg); }
.key .chosen-ring { stroke: var(--accent); fill: none; stroke-width: 2; }
.good { color: var(--good); }
.bad { color: var(--bad); }
.muted { color: var(--muted); }
pre { background: var(--code); border: 1px solid var(--line); border-radius: 6px; padding: .6rem;
  overflow-x: auto; font: 12.5px/1.45 ui-monospace, "SFMono-Regular", Consolas, monospace;
  overflow-wrap: normal; white-space: pre-wrap; }
ul { margin: .25rem 0; padding-left: 1.25rem; }
details > summary { cursor: pointer; color: var(--muted); }

/* the player: the tree on the left, the node panel on the right */
.player { display: grid; grid-template-columns: minmax(0, 3fr) minmax(20rem, 2fr); gap: 1rem;
  align-items: start; }
@media (max-width: 60rem) { .player { grid-template-columns: minmax(0, 1fr); } }
.controls { display: none; flex-wrap: wrap; align-items: center; gap: .4rem; margin: 0 0 .5rem; }
.js .controls { display: flex; }
.controls button, .controls select { font: inherit; padding: .2rem .6rem; border-radius: 6px;
  border: 1px solid var(--line); background: var(--panel); color: var(--fg); cursor: pointer; }
.controls input[type=range] { flex: 1 1 12rem; }
.counter { color: var(--muted); font-variant-numeric: tabular-nums; min-width: 5rem; }
.caption { display: none; min-height: 3em; margin: 0 0 .5rem; padding: .4rem .6rem;
  border-left: 4px solid var(--select); background: var(--panel); border-radius: 0 6px 6px 0; }
.js .caption { display: block; }
.kind-tell .caption { border-left-color: var(--backup); }
.stage { border: 1px solid var(--line); border-radius: 6px; background: var(--bg); }
.stage svg { display: block; margin: 0 auto; max-width: 100%; height: auto;
  font: 11px ui-monospace, "SFMono-Regular", Consolas, monospace; }
.stage text { fill: var(--fg); }
.stage .label { fill: var(--muted); text-anchor: middle; }
.stage .score { text-anchor: middle; dominant-baseline: central; font-size: 9px;
  paint-order: stroke; stroke: var(--bg); stroke-width: 2.5px; }
.stage .ex { dominant-baseline: central; font-size: 12px; font-weight: 700; }
.stage .ex.good { fill: var(--good); }
.stage .ex.bad { fill: var(--bad); }
.stage .edge { stroke: var(--line); fill: none; stroke-width: 1.5; }
.stage .root rect { fill: var(--panel); stroke: var(--muted); stroke-width: 1.5; }
.stage .root text { text-anchor: middle; dominant-baseline: central; fill: var(--muted); }
.stage .node { fill: var(--c); stroke: var(--fg); stroke-width: 1; }
.stage .node.hollow { fill: none; stroke: var(--c); stroke-width: 3; }
.stage .node.dashed { fill: none; stroke: var(--c); stroke-width: 2.5; stroke-dasharray: 4 3; }
.stage .chosen-ring { stroke: var(--accent); fill: none; stroke-width: 2.5; }
.stage a:focus-visible .node, .stage a:hover .node { stroke-width: 3; }
.panel { border: 1px solid var(--line); border-radius: 6px; padding: .75rem;
  max-height: 88vh; overflow-y: auto; position: sticky; top: 8px; }
.panel-empty { color: var(--muted); display: none; }
.js .panel-empty { display: block; }
.node-detail { border-top: 1px solid var(--line); padding: .75rem 0; }
.node-detail:first-of-type { border-top: 0; }
.node-detail:target { background: var(--panel); }
.marks { color: var(--accent); font-weight: 600; }
.prompt-text { margin: .25rem 0; }
.pending-note { display: none; }
.schema-context { margin-top: .75rem; }
.spec { margin: .5rem 0 1rem; }

/* the replay: classes the player script toggles */
.js .panel .node-detail { display: none; border-top: 0; padding-top: 0; }
.js .panel .node-detail.selected { display: block; }
.js .panel-empty[hidden] { display: none; }
.js .node-detail.pending .outcome { display: none; }
.js .node-detail.pending .pending-note { display: block; }
.js .node-g, .js .edge { transition: opacity .35s ease; }
.js .node-g:not(.shown), .js .edge:not(.shown) { opacity: 0; pointer-events: none; }
.js .node-g.pending .node { fill: var(--bg); stroke: var(--c); stroke-dasharray: 3 3;
  stroke-width: 2.5; }
.js .node-g.pending .score, .js .node-g.pending .ex { opacity: 0; }
.js .player-main:not(.finished) .chosen-ring { opacity: 0; }
.js .edge.on-path { stroke: var(--select); stroke-width: 3.5; }
.js .kind-tell .edge.on-path { stroke: var(--backup); }
.js .node-g.on-path .node, .js .root.on-path rect { stroke: var(--select); stroke-width: 3.5; }
.js .kind-tell .node-g.on-path .node, .js .kind-tell .root.on-path rect { stroke: var(--backup); }
.js .node-g.current .node { stroke: var(--accent); stroke-width: 4; }
.js .node-g.current .halo { opacity: 1; animation: halo 1.2s ease-out infinite; }
.halo { fill: none; stroke: var(--accent); stroke-width: 2; opacity: 0; }
@keyframes halo { from { stroke-opacity: .9; } to { stroke-opacity: 0; } }
@media (prefers-reduced-motion: reduce) {
  .js .node-g, .js .edge { transition: none; }
  .js .node-g.current .halo { animation: none; }
}
"""

# The player. It reads {"nodes": n, "start": text, "events": [{kind, node, path, caption}]}
# from the #search-events data block: ``node`` and ``path`` are node indexes (the page's
# element ids are g-<i>, e-<i> and node-<i>), ``kind`` is ask, tell or pick.
PLAYER_JS = """
(() => {
  "use strict";
  const STEP_MS = __STEP_MS__;
  const byId = (id) => document.getElementById(id);
  const data = JSON.parse(byId("search-events").textContent);
  const events = data.events;
  const player = byId("player");
  const slider = byId("step");
  const playButton = byId("play");
  const speed = byId("speed");
  let step = 0;
  let timer = null;
  let selected = null;

  document.body.classList.add("js");
  slider.max = String(events.length);
  // each node's full hover text; while it generates, its title shows data-pending instead
  const titles = [];
  for (let index = 0; index < data.nodes; index++) {
    const title = byId("g-" + index).querySelector("title");
    const pending = title.getAttribute("data-pending");
    titles.push({ title: title, full: title.textContent, pending: pending });
  }

  function select(index) {
    if (selected !== null) byId("node-" + selected).classList.remove("selected");
    selected = index;
    if (index !== null) byId("node-" + index).classList.add("selected");
    byId("panel-empty").hidden = index !== null;
  }

  function render() {
    const asked = new Set();
    const told = new Set();
    for (const event of events.slice(0, step)) {
      if (event.kind === "ask") asked.add(event.node);
      if (event.kind === "tell") told.add(event.node);
    }
    const current = step > 0 ? events[step - 1] : null;
    const path = new Set(current ? current.path : []);
    for (let index = 0; index < data.nodes; index++) {
      const group = byId("g-" + index);
      const edge = byId("e-" + index);
      const shown = asked.has(index);
      const pending = shown && !told.has(index);
      const isCurrent = current !== null && current.node === index;
      group.classList.toggle("shown", shown);
      group.classList.toggle("pending", pending);
      byId("node-" + index).classList.toggle("pending", pending);
      titles[index].title.textContent = pending ? titles[index].pending : titles[index].full;
      group.classList.toggle("on-path", path.has(index));
      group.classList.toggle("current", isCurrent);
      edge.classList.toggle("shown", shown);
      edge.classList.toggle("on-path", path.has(index) || isCurrent);
    }
    byId("root-node").classList.toggle("on-path", current !== null && current.kind !== "pick");
    for (const kind of ["ask", "tell", "pick"]) {
      player.classList.toggle("kind-" + kind, current !== null && current.kind === kind);
    }
    player.classList.toggle("finished", step === events.length);
    byId("caption").textContent = current ? current.caption : data.start;
    byId("counter").textContent = step + " / " + events.length;
    slider.value = String(step);
    select(current === null ? null : current.node);
  }

  function go(target) {
    step = Math.max(0, Math.min(events.length, target));
    render();
  }

  function pause() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    playButton.textContent = "\\u25B6 Play";
  }

  function play() {
    if (step >= events.length) go(0);
    playButton.textContent = "\\u23F8 Pause";
    timer = setInterval(() => {
      if (step >= events.length) pause();
      else go(step + 1);
    }, STEP_MS / Number(speed.value));
  }

  const toggle = () => (timer === null ? play() : pause());
  const moves = {
    first: () => 0,
    back: () => step - 1,
    forward: () => step + 1,
    last: () => events.length,
  };
  for (const [id, target] of Object.entries(moves)) {
    byId(id).addEventListener("click", () => { pause(); go(target()); });
  }
  playButton.addEventListener("click", toggle);
  speed.addEventListener("change", () => { if (timer !== null) { pause(); play(); } });
  slider.addEventListener("input", () => { pause(); go(Number(slider.value)); });
  for (let index = 0; index < data.nodes; index++) {
    byId("g-" + index).addEventListener("click", (event) => {
      event.preventDefault();
      pause();
      select(index);
    });
  }
  // a prompt's schema link opens the page's one copy of that DDL
  for (const link of document.querySelectorAll("a.ctx-link")) {
    link.addEventListener("click", () => {
      const target = byId(link.getAttribute("href").slice(1));
      if (target) target.open = true;
    });
  }
  const keys = {
    " ": toggle,
    ArrowRight: () => { pause(); go(step + 1); },
    ArrowLeft: () => { pause(); go(step - 1); },
    Home: () => { pause(); go(0); },
    End: () => { pause(); go(events.length); },
  };
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const tag = target.tagName;
    // keys keep their own meaning in form controls, links, disclosures and the node panel
    if (!(event.key in keys) || ["INPUT", "SELECT", "SUMMARY", "A", "a"].includes(tag)) return;
    if (target.closest && target.closest(".panel")) return;
    if (event.key === " " && tag === "BUTTON") return;  // the button's own click
    event.preventDefault();
    keys[event.key]();
  });
  render();
})();
""".replace("__STEP_MS__", str(STEP_MS))
