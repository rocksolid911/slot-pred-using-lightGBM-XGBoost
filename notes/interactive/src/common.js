/* Shared helpers for every lesson page: maths, live TeX, charts, lab wiring. */
const D = window.SLOT_DATA;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const f = (x, d = 2) => Number(x).toFixed(d);
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const R = String.raw;
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

/* ---------- maths ---------- */
function erf(x) { const s = x < 0 ? -1 : 1; x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
const phi = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function invPhi(p) {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416], pl = 0.02425;
  let q, r;
  if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  q = p - 0.5; r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}
/* a visit with expected m minutes and log-normal noise sigma (mean 1) */
const visitPdf = (x, m, s) => phi((Math.log(x / m) + s * s / 2) / s) / (x * s);
const quantileOf = (m, q, s) => m * Math.exp(-s * s / 2 + invPhi(q) * s);
const roundUp = x => Math.max(5, Math.ceil(x / 5 - 1e-9) * 5);
function score(m, slot, s) {           /* closed forms for one visit type */
  const d = (Math.log(slot / m) + s * s / 2) / s, cov = Phi(d);
  const idle = slot * cov - m * Phi(d - s);
  return { late: 1 - cov, idle, over: m - slot + idle, absErr: m - slot + 2 * idle };
}
function population(hist, slotOf, s) {
  let n = 0, late = 0, idle = 0, over = 0, slot = 0, abs = 0;
  for (const [m, k] of hist) { const sl = slotOf(m), r = score(m, sl, s);
    n += k; late += k * r.late; idle += k * r.idle; over += k * r.over; slot += k * sl; abs += k * r.absErr; }
  return { late: late / n, idle: idle / n, over: over / n, slot: slot / n, mae: abs / n, n };
}
function range(a, b, step) { const out = []; for (let x = a; x <= b + 1e-9; x += step) out.push(+x.toFixed(6)); return out; }

/* ---------- live TeX ---------- */
const texQueue = new Set(); let texFrame = 0, mjChain = Promise.resolve();
function tex(el, s, display = true) {
  el.textContent = display ? "\\[" + s + "\\]" : "\\(" + s + "\\)";
  texQueue.add(el);
  if (!texFrame) texFrame = requestAnimationFrame(flushTex);
}
function flushTex() {
  texFrame = 0;
  const MJ = window.MathJax;
  if (!MJ || !MJ.typesetPromise) return;
  const els = [...texQueue]; texQueue.clear();
  mjChain = mjChain.then(() => { try { MJ.typesetClear(els); } catch (e) {} return MJ.typesetPromise(els); }).catch(() => {});
}
window.addEventListener("load", () => { const t = setInterval(() => { if (window.MathJax && window.MathJax.typesetPromise) { clearInterval(t); flushTex(); } }, 300); });

/* ---------- charts ---------- */
function hexA(hex, a) { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h, 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; }
function makeChart(id, xTitle, yTitle, extra = {}) {
  return new Chart($(id), { type: "scatter", data: { datasets: [] }, options: {
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { labels: { boxWidth: 12, font: { family: "IBM Plex Sans", size: 12 }, filter: i => !i.text.startsWith("_") } },
               tooltip: { enabled: extra.tooltip !== false, filter: i => !i.dataset.label.startsWith("_") } },
    scales: { x: { type: "linear", title: { display: !!xTitle, text: xTitle }, min: extra.xmin, max: extra.xmax },
              y: { title: { display: !!yTitle, text: yTitle }, min: extra.ymin, max: extra.ymax, display: extra.yDisplay !== false } } } });
}
function paint(c) {
  const m = css("--muted"), g = css("--grid");
  for (const ax of Object.values(c.options.scales)) { ax.ticks = { ...(ax.ticks || {}), color: m }; ax.grid = { color: g }; if (ax.title) ax.title.color = m; }
  c.options.plugins.legend.labels.color = m;
  c.update("none");
}
const line = (label, data, color, o = {}) => ({ label, data, showLine: true, pointRadius: 0, borderColor: color, backgroundColor: color, borderWidth: 2, ...o });
const vline = (label, x, y0, y1, color, dash = []) => line(label, [{ x, y: y0 }, { x, y: y1 }], color, { borderDash: dash, borderWidth: 1.5 });
const area = (label, data, color, a = 0.22) => ({ label, data, showLine: true, pointRadius: 0, borderWidth: 0, fill: "origin", backgroundColor: hexA(color, a) });

/* ---------- lab wiring ---------- */
function bindOut(id, fmt) { const el = $(id), out = $(id + "Out"); const set = () => out.textContent = fmt(+el.value); el.addEventListener("input", set); set(); }
const LAB_UPDATES = [];
function registerLab(fn, inputIds = []) { LAB_UPDATES.push(fn); inputIds.forEach(id => $(id).addEventListener("input", fn)); }
function startLabs() {
  const all = () => LAB_UPDATES.forEach(fn => fn());
  all();
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) mq.addEventListener("change", all);
  new MutationObserver(all).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
}
