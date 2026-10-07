/* RASTA-QF frontend helpers */

const C = {
  cyan: "#22d3ee", violet: "#a78bfa", green: "#34d399", red: "#fb7185",
  amber: "#fbbf24", blue: "#60a5fa", slate: "#64748b", pink: "#f472b6",
};

Chart.defaults.color = "#8b9bb8";
Chart.defaults.borderColor = "rgba(103,132,187,0.12)";
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11.5;
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.boxHeight = 10;
Chart.defaults.animation.duration = 600;

function gridOpts(extra = {}) {
  return Object.assign({
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 9 } },
      y: { grid: { color: "rgba(103,132,187,0.10)" }, ticks: { maxTicksLimit: 6 } },
    },
    plugins: { legend: { position: "top" } },
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
  }, extra);
}

function lineChart(id, labels, datasets, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return null;
  return new Chart(el, {
    type: "line",
    data: { labels, datasets: datasets.map((d, i) => Object.assign({
      borderColor: d.color || Object.values(C)[i % 7],
      backgroundColor: (d.color || Object.values(C)[i % 7]) + "22",
      borderWidth: d.width ?? 2, pointRadius: d.points ? 2 : 0,
      tension: 0.35, fill: !!d.fill,
    }, d)) },
    options: gridOpts(opts),
  });
}

function barChart(id, labels, datasets, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return null;
  return new Chart(el, {
    type: "bar",
    data: { labels, datasets: datasets.map((d, i) => Object.assign({
      backgroundColor: d.color || Object.values(C)[i % 7],
      borderRadius: 6, borderSkipped: false, maxBarThickness: 34,
    }, d)) },
    options: gridOpts(opts),
  });
}

function areaChart(id, labels, datasets, opts = {}) {
  return lineChart(id, labels, datasets.map(d => Object.assign({ fill: true }, d)),
    Object.assign({ scales: { x: { grid: { display: false } }, y: { stacked: true, max: 1 } } }, opts));
}

const api = (url, opts) => fetch(url, opts).then(r => r.json());

/* ---------------- pipeline control ---------------- */
let pollTimer = null;

async function refreshStatus() {
  try {
    const st = await api("/api/status");
    document.querySelectorAll("[data-mod-dot]").forEach(el => {
      const m = st.modules[el.dataset.modDot];
      el.className = "dot " + (m ? m.status : "pending");
    });
    document.querySelectorAll("[data-mod-bar]").forEach(el => {
      const m = st.modules[el.dataset.modBar];
      if (!m) return;
      el.style.width = (m.status === "done" ? 100 : m.progress) + "%";
      const tx = document.querySelector(`[data-mod-msg="${el.dataset.modBar}"]`);
      if (tx) tx.textContent = m.status === "running"
        ? `${Math.round(m.progress)}% — ${m.message || "computing"}…`
        : (m.message || m.status);
      const bt = document.querySelector(`[data-mod-badge="${el.dataset.modBar}"]`);
      if (bt) {
        bt.className = "badge " + ({ done: "ok", running: "warn", error: "bad" }[m.status] || "muted");
        bt.textContent = m.status.toUpperCase();
      }
    });
    return st;
  } catch (e) { return { modules: {}, running: false }; }
}

function startPolling(onIdle) {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const st = await refreshStatus();
    if (!st.running) {
      clearInterval(pollTimer);
      pollTimer = null;
      const btn = document.getElementById("runBtn");
      if (btn) { btn.disabled = false; btn.innerHTML = "▶&nbsp; Run Full Pipeline"; }
      if (onIdle) onIdle();
    }
  }, 1500);
}

async function runPipeline(modules = ["all"], onDone) {
  const btn = document.getElementById("runBtn");
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>&nbsp; Running…'; }
  await api("/api/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modules }),
  });
  startPolling(onDone);
}

/* ---------------- module data loading ---------------- */
async function moduleData(name) {
  const r = await api(`/api/results/${name}`);
  if (r.status !== "done" || !r.results) return null;
  return r.results;
}

function showEmpty(containerId, msg = "No results yet — run the pipeline to compute this module.") {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<div class="empty"><div class="big">◇</div>${msg}<br><br>
    <button class="btn primary" onclick="runPipeline(['all'], ()=>location.reload())">▶&nbsp; Run Full Pipeline</button></div>`;
}

function heatColor(v, lo = -1, hi = 1) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const r = Math.round(251 * t + 96 * (1 - t));
  const g = Math.round(113 * t + 165 * (1 - t));
  const b = Math.round(133 * t + 250 * (1 - t));
  return `rgba(${r},${g},${b},${0.12 + 0.5 * Math.abs(t - 0.5) * 2})`;
}

function fmt(x, d = 2) { return Number(x).toFixed(d); }

/* boot: status dots + polling if running */
document.addEventListener("DOMContentLoaded", async () => {
  const st = await refreshStatus();
  if (st.running) startPolling();
});
