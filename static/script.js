const SHAPE = {
  circle: { icon: "⬤", color: "#ff6b8a", text: "Rounded contours and radial symmetry detected." },
  rectangle: { icon: "▬", color: "#4ecdc4", text: "Parallel straight edges and right-angle corners confirmed." },
  triangle: { icon: "▲", color: "#ffe66d", text: "Three converging edges forming a peaked triangular structure." },
};

let chart = null;

// ── Tab switching ─────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll(".tab-section").forEach(s => s.style.display = "none");
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("section-" + tab).style.display = "block";
  document.getElementById("tab-" + tab).classList.add("active");
  if (tab === "metrics") loadMetrics();
}

// ── File upload / drag & drop ─────────────────────
const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");

document.getElementById("drop-zone").addEventListener("dragover", e => { e.preventDefault(); uploadArea.classList.add("drag-over"); });
document.getElementById("drop-zone").addEventListener("dragleave", () => uploadArea.classList.remove("drag-over"));
document.getElementById("drop-zone").addEventListener("drop", e => { e.preventDefault(); uploadArea.classList.remove("drag-over"); handleFile(e.dataTransfer.files[0]); });
document.getElementById("drop-zone").addEventListener("click", e => { if (e.target.tagName !== "LABEL") fileInput.click(); });
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file || !file.type.match(/image\/(jpeg|png)/)) return alert("JPG or PNG only.");
  const reader = new FileReader();
  reader.onload = e => document.getElementById("preview-img").src = e.target.result;
  reader.readAsDataURL(file);

  document.getElementById("results-section").style.display = "none";
  document.getElementById("loading").style.display = "block";

  const form = new FormData();
  form.append("file", file);
  fetch("/predict", { method: "POST", body: form })
    .then(r => r.json())
    .then(showResults)
    .catch(err => alert("Error: " + err.message))
    .finally(() => document.getElementById("loading").style.display = "none");
}

// ── Render prediction results ─────────────────────
function showResults(data) {
  document.getElementById("results-section").style.display = "block";
  document.getElementById("edge-img").src = "data:image/png;base64," + data.edge_image;

  const card = document.getElementById("result-card");
  if (data.low_confidence) {
    card.className = "result-card warn";
    card.style.cssText = "";
    card.innerHTML = `<div class="shape-icon" style="background:rgba(255,215,64,.12)">⚠️</div>
      <div class="result-meta">
        <p class="result-label">Result</p>
        <p class="result-shape-name" style="color:#ffd740;font-size:1.5rem">Uncertain</p>
        <p class="conf-label">Model not confident — highest: <strong>${data.confidence}%</strong></p>
      </div>`;
  } else {
    const { icon, color, text } = SHAPE[data.predicted_class] || SHAPE.circle;
    card.className = "result-card";
    card.style.cssText = `border-color:${color}44; background:linear-gradient(135deg,${color}18,${color}08)`;
    card.innerHTML = `
      <div class="shape-icon" style="background:${color}22;color:${color}">${icon}</div>
      <div class="result-meta">
        <p class="result-label">Predicted Shape</p>
        <p class="result-shape-name" style="color:${color}">${data.predicted_class.toUpperCase()}</p>
        <div class="conf-track"><div class="conf-fill" id="cbar" style="width:0%;background:${color}"></div></div>
        <p class="conf-label">Confidence: <strong>${data.confidence}%</strong></p>
        <p class="conf-label" style="margin-top:6px;font-style:italic">${text}</p>
      </div>`;
    setTimeout(() => { const b = document.getElementById("cbar"); if (b) b.style.width = data.confidence + "%"; }, 80);
  }

  // Chart
  if (chart) chart.destroy();
  const labels = Object.keys(data.probabilities);
  const colors = labels.map(l => SHAPE[l]?.color || "#7c6ff7");
  chart = new Chart(document.getElementById("prob-chart"), {
    type: "bar",
    data: { labels: labels.map(l => l[0].toUpperCase() + l.slice(1)), datasets: [{ data: Object.values(data.probabilities), backgroundColor: colors.map(c => c + "99"), borderColor: colors, borderWidth: 2, borderRadius: 8, borderSkipped: false }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.parsed.y.toFixed(2)}%` } } }, scales: { y: { beginAtZero: true, max: 100, ticks: { color: "#6a6d8e", callback: v => v + "%" }, grid: { color: "rgba(255,255,255,.05)" } }, x: { ticks: { color: "#6a6d8e" }, grid: { display: false } } } },
  });
}

// ── Metrics tab ───────────────────────────────────
async function loadMetrics() {
  const el = document.getElementById("metrics-content");
  el.innerHTML = '<div class="loading-placeholder">Loading…</div>';
  try {
    const data = await fetch("/metrics").then(r => r.json());
    if (data.error) { el.innerHTML = `<div class="error-box"><p>📊 ${data.error}</p><p style="margin-top:8px;font-size:.85rem">Run <code>python eval.py</code> first.</p></div>`; return; }
    const cls = ["circle", "rectangle", "triangle"], clr = { circle: "#ff6b8a", rectangle: "#4ecdc4", triangle: "#ffe66d" };
    const r = data.classification_report || {}, wav = r["weighted avg"] || {}, cm = data.confusion_matrix || [];
    const pct = v => v != null ? (v * 100).toFixed(1) + "%" : "—";

    el.innerHTML = `
      <div class="metrics-top">
        ${[["🎯", "Test Accuracy", data.accuracy + "%"], ["⚖️", "Weighted F1", pct(wav["f1-score"])], ["📐", "Weighted Precision", pct(wav["precision"])]].map(([e, l, v]) => `<div class="metric-tile"><div class="metric-emoji">${e}</div><div class="metric-value">${v}</div><div class="metric-label">${l}</div></div>`).join("")}
      </div>
      <div class="glass-card">
        <h3 class="card-title"><span class="dot dot-purple"></span>Per-Class Report</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>${cls.map(c => { const d = r[c] || {}; return `<tr><td><div class="cls-label"><div class="cls-dot" style="background:${clr[c]}"></div>${c[0].toUpperCase() + c.slice(1)}</div></td><td style="color:${clr[c]}">${pct(d.precision)}</td><td style="color:${clr[c]}">${pct(d.recall)}</td><td style="color:${clr[c]}">${pct(d["f1-score"])}</td><td style="color:var(--muted)">${d.support ?? '—'}</td></tr>`; }).join("")}</tbody>
        </table></div>
      </div>
      <div class="glass-card">
        <h3 class="card-title"><span class="dot dot-cyan"></span>Confusion Matrix</h3>
        <div class="cm-wrap"><table class="cm-table">
          <thead><tr><th></th>${cls.map(c => `<th style="color:${clr[c]}">${c[0].toUpperCase() + c.slice(1)}</th>`).join("")}</tr></thead>
          <tbody>${cm.map((row, i) => `<tr><th style="text-align:right;color:${clr[cls[i]]};padding-right:10px">${cls[i][0].toUpperCase() + cls[i].slice(1)}</th>${row.map((v, j) => `<td class="cm-cell ${i === j ? 'cm-diag' : v > 0 ? 'cm-miss' : 'cm-zero'}">${v}</td>`).join("")}</tr>`).join("")}</tbody>
        </table></div>
      </div>`;
  } catch { el.innerHTML = '<div class="error-box">Could not load metrics.</div>'; }
}
