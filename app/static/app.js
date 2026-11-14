const $ = id => document.getElementById(id);
const PRESETS = { en: ["king","queen","man","woman","crown","apple","pear","grape"],
                  tr: ["kral","kraliçe","adam","kadın","taç","elma","armut","üzüm"] };
const T = { en: { tip: "A leading space is part of the token in GPT-2/Qwen: 'king' ≠ ' king' (different IDs and vectors).",
                  tok: "Token", variants: "variants", axis: "cosine with axis", cname: "name", cpos: "positive words, comma-separated", cneg: "negative words (optional)", neutral: "neutral set" },
            tr: { tip: "GPT-2/Qwen'de baştaki boşluk token'ın parçasıdır: 'king' ≠ ' king' (farklı ID ve vektör).",
                  tok: "Token", variants: "varyantlar", axis: "eksenle kosinüs", cname: "ad", cpos: "pozitif kelimeler, virgülle", cneg: "negatif kelimeler (opsiyonel)", neutral: "nötr küme" } };
const plang = () => (state.meta && state.meta.lang === "tr") ? "tr" : TL_LANG();
const L = () => T[TL_LANG()] || T.en;
const state = { tokens: [], sel: null, axes: {}, custom: null, vec: null, model: "gpt2", meta: null };
try { state.model = localStorage.getItem("tl-model") || "gpt2"; } catch (e) {}
const dark = () => matchMedia("(prefers-color-scheme:dark)").matches;
const layoutBase = () => ({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: dark() ? "#ececf1" : "#1a1a1f" }, margin: { l: 70, r: 10, t: 10, b: 40 } });
const j = async u => { u += (u.includes("?") ? "&" : "?") + "model=" + encodeURIComponent(state.model);
  const r = await fetch(u); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); };
const vis = s => s.replace(/ /g, "␣");
const word = t => t.trim().toLowerCase();
const debounce = (f, ms) => { let t; return () => { clearTimeout(t); t = setTimeout(f, ms); }; };

async function init() {
  const ms = await (await fetch("/api/models")).json();
  if (!ms.models.some(m => m.slug === state.model)) state.model = ms.default;
  $("model").innerHTML = ms.models.map(m => `<option value="${m.slug}">${m.label}</option>`).join("");
  $("model").value = state.model;
  $("model").onchange = async () => { state.model = $("model").value; try { localStorage.setItem("tl-model", state.model); } catch (e) {}
    state.sel = null; await loadModel(); await run(); analogy(); };
  $("text").addEventListener("input", debounce(run, 250));
  $("go").onclick = analogy; $("cadd").onclick = addCustom;
  $("ax").onchange = $("ay").onchange = plot2d;
  $("copy").onclick = () => navigator.clipboard.writeText(JSON.stringify(state.vec));
  document.addEventListener("tl-lang", () => { applyLang(); });
  await loadModel(); applyLang(); await run(); analogy();
}
function applyLang() {
  $("tip").title = L().tip; $("tip").textContent = "ⓘ";
  $("cname").placeholder = L().cname; $("cpos").placeholder = L().cpos; $("cneg").placeholder = L().cneg;
  renderPresets(); renderAxDefs();
}
async function loadModel() {
  const m = await j("/api/meta"); state.meta = m; state.axes = { ...m.axes }; state.custom = null;
  $("meta").textContent = `${m.model_id} · ${m.vocab.toLocaleString()} tokens × ${m.dims} dims`;
  const heat = $("heat"); heat.width = m.dims;
  renderAxDefs(); renderPresets(); fillSelectors(Object.keys(state.axes), true);
  const tr = m.lang === "tr";   // Turkish-only model: switch defaults to Turkish words
  $("text").value = tr ? "kral kraliçe adam kadın elma" : "king queen man woman apple";
  [$("a").value, $("b").value, $("c").value] = tr ? ["kral", "adam", "kadın"] : ["king", "man", "woman"];
}
function renderPresets() {
  const box = $("presets"); const old = $("variants")?.textContent || ""; box.innerHTML = "";
  PRESETS[plang()].forEach(w => { const b = document.createElement("button"); b.textContent = w;
    b.onclick = () => { $("text").value = $("text").value.trim() ? $("text").value + " " + w : w; run(); }; box.append(b); });
  const s = document.createElement("span"); s.className = "mut"; s.id = "variants"; s.textContent = old; box.append(s);
}
function renderAxDefs() {
  $("axdefs").innerHTML = Object.entries(state.axes).map(([n, d]) => d.pairs
    ? `<div><b>${n}</b> = mean(${d.pairs.map(p => p[0] + "−" + p[1]).join(", ")})</div>`
    : `<div><b>${n}</b> = mean(${d.pos.join(", ")}) − mean(${(d.neg || []).filter(Boolean).join(", ") || (state.meta?.neutral || []).join(", ") + " (" + L().neutral + ")"})</div>`).join("");
}
function fillSelectors(names, reset) {
  const keep = reset ? [] : [$("ax").value, $("ay").value];
  for (const s of [$("ax"), $("ay")]) s.innerHTML = names.map(n => `<option>${n}</option>`).join("");
  $("ax").value = names.includes(keep[0]) ? keep[0] : names[1]; $("ay").value = names.includes(keep[1]) ? keep[1] : names[0];
}

async function run() {
  const text = $("text").value;
  if (!text.trim()) { $("chips").innerHTML = ""; return; }
  state.tokens = await j("/api/tokenize?text=" + encodeURIComponent(text));
  $("chips").innerHTML = "";
  state.tokens.forEach((t, i) => {
    const c = document.createElement("div"); c.className = "chip"; c.dataset.i = i;
    c.innerHTML = `<b>${vis(t.token)}</b><small>ID ${t.token_id}${t.leading_space ? " · ␣" : ""}</small>`;
    c.onclick = () => select(i); $("chips").append(c);
  });
  const ws = [...new Set(text.split(/\s+/).map(word).filter(Boolean))].slice(0, 6);
  if (ws.length) {
    const v = await j("/api/words?words=" + encodeURIComponent(ws.join(",")));
    $("variants").textContent = v.map(r => `"${r.word}"=${r.no_space.ids.join("+")} / "␣${r.word}"=${r.with_space.ids.join("+")}`).join("  ·  ");
  }
  select(Math.max(0, Math.min(state.sel ?? 0, state.tokens.length - 1)));
  plots();
}
async function select(i) {
  state.sel = i; const t = state.tokens[i]; if (!t) return;
  document.querySelectorAll(".chip").forEach(c => c.classList.toggle("sel", +c.dataset.i === i));
  $("panel").hidden = false;
  const [d, nn] = await Promise.all([j("/api/token/" + t.token_id), j(`/api/neighbors/${t.token_id}?k=10`)]);
  state.vec = d.vector;
  $("big").textContent = `${L().tok}: '${d.token}' | ID: ${d.token_id} | dims: ${d.dims}`;
  $("norm").textContent = "L2 norm " + d.norm.toFixed(3);
  heat(d.vector);
  $("vals").innerHTML = "<tr><th>dim</th><th>value</th></tr>" + d.vector.slice(0, 16).map((v, k) => `<tr><td>${k}</td><td>${v.toFixed(5)}</td></tr>`).join("");
  $("nn").innerHTML = nn.map(n => `<tr><td>${vis(n.token)}</td><td>${n.id}</td><td>${n.similarity.toFixed(3)}</td></tr>`).join("");
  bars(t);
}
function heat(v) {
  const c = $("heat"), x = c.getContext("2d"), n = v.length; c.width = n; c.height = 1;
  const m = Math.max(...v.map(Math.abs)) || 1;
  v.forEach((val, k) => { const s = Math.sqrt(Math.abs(val) / m);
    x.fillStyle = val >= 0 ? `rgb(${255 - 38 * s | 0},${255 - 183 * s | 0},${255 - 240 * s | 0})` : `rgb(${255 - 227 * s | 0},${255 - 129 * s | 0},${255 - 41 * s | 0})`;
    x.fillRect(k, 0, 1, 1); });
  c.onmousemove = e => { const r = c.getBoundingClientRect(), k = Math.min(n - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * n)));
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
    { ...layoutBase(), xaxis: { zeroline: true, range: [-1, 1], title: L().axis }, yaxis: { autorange: "reversed" } }, { displayModeBar: false });
}
async function plots() {
  const typed = [...new Set($("text").value.split(/\s+/).map(word).filter(Boolean))];
  const ws = [...new Set([...PRESETS[plang()], ...typed])].slice(0, 40);
  const variants = await j("/api/words?words=" + encodeURIComponent(ws.join(",")));
  const ids = variants.map(v => v.with_space.ids[0]);
  const pts = await j("/api/pca?ids=" + ids.join(","));
  Plotly.react("p3", [{ type: "scatter3d", mode: "markers+text", text: ws, textposition: "top center",
    x: pts.map(p => p.x), y: pts.map(p => p.y), z: pts.map(p => p.z),
    marker: { size: 4, color: ws.map(w => PRESETS[plang()].includes(w) ? "#0891b2" : "#d9480f") } }],
    { ...layoutBase(), margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } } }, { displayModeBar: false });
  state.planeWords = ws; plot2d();
}
async function plot2d() {
  const ws = state.planeWords; if (!ws) return;
  const s = await axesFor(ws), X = $("ax").value, Y = $("ay").value;
  if (!s[ws[0]] || !(X in s[ws[0]]) || !(Y in s[ws[0]])) return;
  Plotly.react("p2", [{ type: "scatter", mode: "markers+text", text: ws, textposition: "top center",
    x: ws.map(w => s[w][X]), y: ws.map(w => s[w][Y]), marker: { size: 9, color: "#0891b2" } }],
    { ...layoutBase(), xaxis: { title: X, zeroline: true }, yaxis: { title: Y, zeroline: true } }, { displayModeBar: false });
}
async function addCustom() {
  const name = $("cname").value.trim(), pos = $("cpos").value.trim();
  if (!name || !pos) return;
  state.custom = { name, pos, neg: $("cneg").value.trim() };
  state.axes[name] = { pos: pos.split(",").map(x => x.trim()), neg: state.custom.neg.split(",").map(x => x.trim()) };
  renderAxDefs(); fillSelectors(Object.keys(state.axes)); plot2d(); if (state.tokens[state.sel]) bars(state.tokens[state.sel]);
}
async function analogy() {
  const [a, b, c] = ["a", "b", "c"].map(i => encodeURIComponent($(i).value.trim()));
  const r = await j(`/api/analogy?a=${a}&b=${b}&c=${c}&k=10`);
  $("ana").innerHTML = r.map(n => `<tr><td>${vis(n.token)}</td><td>${n.id}</td><td>${n.similarity.toFixed(3)}</td></tr>`).join("");
}

// ---- live server load
const hist = { cpu: [], ram: [] }, N = 60;
function spark(id, a, b, max, c1, c2) {
  const c = $(id), x = c.getContext("2d"), w = c.width, h = c.height; x.clearRect(0, 0, w, h);
  [[a, c1], [b, c2]].forEach(([arr, col]) => { x.beginPath(); x.strokeStyle = col; x.lineWidth = 2;
    arr.forEach((v, i) => { const px = i / (N - 1) * w, py = h - Math.min(1, v / max) * (h - 4) - 2; i ? x.lineTo(px, py) : x.moveTo(px, py); }); x.stroke(); });
}
async function poll() {
  if (document.hidden || !$("stats").open) return;
  try {
    const s = await (await fetch("/api/stats")).json();
    hist.cpu.push([s.cpu_proc, s.cpu_sys]); hist.ram.push([s.rss_mb, s.mem_used_mb]);
    if (hist.cpu.length > N) { hist.cpu.shift(); hist.ram.shift(); }
    $("cpuv").textContent = `${s.cpu_proc.toFixed(0)}% / ${s.cpu_sys.toFixed(0)}%`;
    $("ramv").textContent = `${s.rss_mb.toFixed(0)} MB / ${s.mem_used_mb.toFixed(0)} of ${s.mem_total_mb.toFixed(0)} MB`;
    const cm = Math.max(100, ...hist.cpu.flat()), rm = s.mem_total_mb;
    spark("cpuc", hist.cpu.map(v => v[0]), hist.cpu.map(v => v[1]), cm, "#e11d48", "#9a9aa8");
    spark("ramc", hist.ram.map(v => v[0]), hist.ram.map(v => v[1]), rm, "#e11d48", "#9a9aa8");
    $("statx").textContent = `${s.cpus} vCPU · models in memory: ${s.models_loaded} · red = process, grey = whole machine · updates every 2 s while this section is open`;
  } catch (e) {}
}
setInterval(poll, 2000);
document.addEventListener("DOMContentLoaded", init);
