const $ = id => document.getElementById(id);
const PRESETS = ["king","queen","man","woman","crown","apple","pear","grape"];
const state = { tokens: [], sel: null, axes: {}, custom: null, vec: null };
const dark = () => matchMedia("(prefers-color-scheme:dark)").matches;
const layoutBase = () => ({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: dark() ? "#ececf1" : "#1a1a1f" }, margin: { l: 50, r: 10, t: 10, b: 40 } });
const j = async u => { const r = await fetch(u); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); };
const vis = s => s.replace(/ /g, "␣");
const word = t => t.trim().toLowerCase();

async function init() {
  const m = await j("/api/meta");
  $("meta").textContent = `${m.model_id}, ${m.vocab.toLocaleString()} tokens × ${m.dims} dims`;
  state.axes = m.axes;
  $("axdefs").innerHTML = Object.entries(m.axes).map(([n, d]) => d.pairs
    ? `<div><b>${n}</b> = mean(${d.pairs.map(p => p[0] + "−" + p[1]).join(", ")})</div>`
    : `<div><b>${n}</b> = mean(${d.pos.join(", ")}) − mean(${d.neg.join(", ")})</div>`).join("");
  PRESETS.forEach(w => { const b = document.createElement("button"); b.textContent = w;
    b.onclick = () => { $("text").value = $("text").value.trim() ? $("text").value + " " + w : w; run(); }; $("presets").append(b); });
  const pw = document.createElement("span"); pw.className = "mut"; pw.id = "variants"; $("presets").append(pw);
  fillSelectors(Object.keys(m.axes));
  $("text").addEventListener("input", debounce(run, 250));
  $("go").onclick = analogy; $("cadd").onclick = addCustom;
  $("ax").onchange = $("ay").onchange = plot2d;
  $("copy").onclick = () => navigator.clipboard.writeText(JSON.stringify(state.vec));
  run(); analogy();
}
function fillSelectors(names) {
  const keep = [$("ax").value, $("ay").value];
  for (const s of [$("ax"), $("ay")]) s.innerHTML = names.map(n => `<option>${n}</option>`).join("");
  $("ax").value = keep[0] || names[1]; $("ay").value = keep[1] || names[0];
  if (!$("ax").value) $("ax").selectedIndex = 1; if (!$("ay").value) $("ay").selectedIndex = 0;
}
function debounce(f, ms) { let t; return () => { clearTimeout(t); t = setTimeout(f, ms); }; }

async function run() {
  const text = $("text").value;
  if (!text.trim()) { $("chips").innerHTML = ""; return; }
  state.tokens = await j("/api/tokenize?text=" + encodeURIComponent(text));
  $("chips").innerHTML = "";
  state.tokens.forEach((t, i) => {
    const c = document.createElement("div"); c.className = "chip"; c.dataset.i = i;
    c.title = t.leading_space ? "Leading space is part of this token" : "No leading space";
    c.innerHTML = `<b>${vis(t.token)}</b><small>ID ${t.token_id}${t.leading_space ? " · ␣" : ""}</small>`;
    c.onclick = () => select(i); $("chips").append(c);
  });
  const ws = [...new Set(text.split(/\s+/).map(word).filter(Boolean))].slice(0, 6);
  if (ws.length) {
    const v = await j("/api/words?words=" + ws.join(","));
    $("variants").textContent = v.map(r => `"${r.word}"=${r.no_space.ids} vs " ${r.word}"=${r.with_space.ids}`).join("  ·  ");
  }
  select(Math.min(state.sel ?? 0, state.tokens.length - 1));
  plots();
}

async function select(i) {
  state.sel = i; const t = state.tokens[i]; if (!t) return;
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("sel", +c.dataset.i === i));
  $("panel").hidden = false;
  const [d, nn] = await Promise.all([j("/api/token/" + t.token_id), j(`/api/neighbors/${t.token_id}?k=10`)]);
  state.vec = d.vector;
  $("big").textContent = `Token: '${d.token}' | ID: ${d.token_id} | dims: ${d.dims}`;
  $("norm").textContent = "L2 norm " + d.norm.toFixed(3);
  heat(d.vector);
  $("vals").innerHTML = "<tr><th>dim</th><th>value</th></tr>" + d.vector.slice(0, 16).map((v, k) => `<tr><td>${k}</td><td>${v.toFixed(5)}</td></tr>`).join("");
  $("nn").innerHTML = nn.map(n => `<tr><td>${vis(n.token)}</td><td>${n.id}</td><td>${n.similarity.toFixed(3)}</td></tr>`).join("");
  bars(t);
}
function heat(v) {
  const c = $("heat"), x = c.getContext("2d"), m = Math.max(...v.map(Math.abs)) || 1;
  v.forEach((val, k) => { const a = Math.abs(val) / m, s = Math.sqrt(a);
    x.fillStyle = val >= 0 ? `rgba(217,72,15,${s})` : `rgba(28,126,214,${s})`;
    x.fillStyle = val >= 0 ? `rgb(${255 - 38 * s | 0},${255 - 183 * s | 0},${255 - 240 * s | 0})` : `rgb(${255 - 227 * s | 0},${255 - 129 * s | 0},${255 - 41 * s | 0})`;
    x.fillRect(k, 0, 1, 1); });
  c.onmousemove = e => { const r = c.getBoundingClientRect(), k = Math.min(767, Math.max(0, Math.floor((e.clientX - r.left) / r.width * 768)));
    $("hover").textContent = `dim ${k}: ${v[k].toFixed(5)}`; };
}
async function axesFor(words) {
  let u = "/api/axes?tokens=" + encodeURIComponent(words.join(","));
  if (state.custom) u += `&custom_name=${encodeURIComponent(state.custom.name)}&pos=${encodeURIComponent(state.custom.pos)}&neg=${encodeURIComponent(state.custom.neg)}`;
  return (await j(u)).scores;
}
async function bars(t) {
  const w = word(t.token), s = (await axesFor([w]))[w], names = Object.keys(s);
  Plotly.react("bars", [{ type: "bar", orientation: "h", y: names, x: names.map(n => s[n]),
    marker: { color: names.map(n => s[n] >= 0 ? "#d9480f" : "#1c7ed6") } }],
    { ...layoutBase(), xaxis: { zeroline: true, range: [-1, 1], title: "cosine with axis" }, yaxis: { autorange: "reversed" } }, { displayModeBar: false });
}
async function plots() {
  const typed = [...new Set($("text").value.split(/\s+/).map(word).filter(Boolean))];
  const ws = [...new Set([...PRESETS, ...typed])].slice(0, 40);
  const variants = await j("/api/words?words=" + ws.join(","));
  const ids = variants.map(v => v.with_space.ids.length === 1 ? v.with_space.ids[0] : v.with_space.ids[0]);
  const pts = await j("/api/pca?ids=" + ids.join(","));
  Plotly.react("p3", [{ type: "scatter3d", mode: "markers+text", text: ws, textposition: "top center",
    x: pts.map(p => p.x), y: pts.map(p => p.y), z: pts.map(p => p.z),
    marker: { size: 4, color: ws.map(w => PRESETS.includes(w) ? "#4f46e5" : "#d9480f") } }],
    { ...layoutBase(), margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } } }, { displayModeBar: false });
  state.planeWords = ws; plot2d();
}
async function plot2d() {
  const ws = state.planeWords; if (!ws) return;
  const s = await axesFor(ws), X = $("ax").value, Y = $("ay").value;
  Plotly.react("p2", [{ type: "scatter", mode: "markers+text", text: ws, textposition: "top center",
    x: ws.map(w => s[w][X]), y: ws.map(w => s[w][Y]), marker: { size: 9, color: "#4f46e5" } }],
    { ...layoutBase(), xaxis: { title: X, zeroline: true }, yaxis: { title: Y, zeroline: true } }, { displayModeBar: false });
}
async function addCustom() {
  const name = $("cname").value.trim(), pos = $("cpos").value.trim();
  if (!name || !pos) return;
  state.custom = { name, pos, neg: $("cneg").value.trim() };
  state.axes[name] = { pos: pos.split(","), neg: state.custom.neg.split(",") };
  $("axdefs").insertAdjacentHTML("beforeend", `<div><b>${name}</b> (custom) = mean(${pos}) − mean(${state.custom.neg || "neutral set"})</div>`);
  fillSelectors(Object.keys(state.axes)); plot2d(); if (state.tokens[state.sel]) bars(state.tokens[state.sel]);
}
async function analogy() {
  const [a, b, c] = ["a", "b", "c"].map(i => $(i).value.trim());
  const r = await j(`/api/analogy?a=${a}&b=${b}&c=${c}&k=10`);
  $("ana").innerHTML = r.map(n => `<tr><td>${vis(n.token)}</td><td>${n.id}</td><td>${n.similarity.toFixed(3)}</td></tr>`).join("");
}
init();
