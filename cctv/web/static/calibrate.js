// Calibration helper page: click lane-dash corners, fit a homography on the server, inspect the result.
// The maths lives in cctv/calibrate.py; this page only collects points and draws what comes back.

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
let spec = { points: [] };   // same format as data/calibration/<id>.points.json
let fit = null;              // last server response

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json();
  if (!r.ok) throw new Error(body.detail || r.status);
  return body;
}

function draw() {
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  if (fit) {
    fit.grid.forEach((line, i) => {
      ctx.strokeStyle = line.length > 2 ? "#00e5ff" : "#ffd400";
      ctx.lineWidth = 1;
      ctx.beginPath();
      line.forEach(([u, v], k) => (k ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
      ctx.stroke();
    });
    ctx.fillStyle = "#00ff66";
    fit.reprojected.forEach(([u, v]) => { ctx.beginPath(); ctx.arc(u, v, 2, 0, 7); ctx.fill(); });
  }
  ctx.strokeStyle = "#ff3b30";
  ctx.lineWidth = 1.5;
  spec.points.forEach(p => { ctx.beginPath(); ctx.arc(p.img[0], p.img[1], 4, 0, 7); ctx.stroke(); });
}

function renderTable() {
  const tbody = document.getElementById("pts");
  tbody.innerHTML = spec.points.map((p, i) => `<tr>
      <td>${p.img[0].toFixed(1)}</td><td>${p.img[1].toFixed(1)}</td>
      <td><input type="number" min="0" value="${p.line}" data-i="${i}" data-k="line"></td>
      <td><input type="number" min="0" value="${p.dash}" data-i="${i}" data-k="dash"></td>
      <td><select data-i="${i}" data-k="end"><option${p.end === "start" ? " selected" : ""}>start</option><option${p.end === "end" ? " selected" : ""}>end</option></select></td>
      <td class="num">${fit && fit.errors_m[i] !== undefined ? fit.errors_m[i].toFixed(3) : ""}</td>
      <td><button data-del="${i}" aria-label="Remove point ${i + 1}">Remove</button></td></tr>`).join("");
  tbody.querySelectorAll("input, select").forEach(el => el.addEventListener("change", () => {
    const p = spec.points[+el.dataset.i];
    p[el.dataset.k] = el.dataset.k === "end" ? el.value : parseInt(el.value, 10);
    fit = null; draw();
  }));
  tbody.querySelectorAll("[data-del]").forEach(b => b.addEventListener("click", () => {
    spec.points.splice(+b.dataset.del, 1); fit = null; renderTable(); draw();
  }));
}

canvas.addEventListener("click", ev => {
  const r = canvas.getBoundingClientRect();
  const u = (ev.clientX - r.left) * canvas.width / r.width;
  const v = (ev.clientY - r.top) * canvas.height / r.height;
  const line = parseInt(document.getElementById("next-line").value, 10) || 0;
  const dashEl = document.getElementById("next-dash");
  const dash = parseInt(dashEl.value, 10) || 0;
  // Alternate start/end on the same dash, then move to the next dash.
  const prev = spec.points[spec.points.length - 1];
  const end = prev && prev.line === line && prev.dash === dash && prev.end === "start" ? "end" : "start";
  spec.points.push({ img: [+u.toFixed(1), +v.toFixed(1)], line, dash, end });
  if (end === "end") dashEl.value = dash + 1;
  fit = null; renderTable(); draw();
});

async function loadCamera(id) {
  fit = null;
  spec = { camera_id: id, image_size: [640, 480], points: [], unknown_phase_lines: [] };
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; spec.image_size = [img.naturalWidth, img.naturalHeight]; draw(); };
  img.src = `/api/cameras/${encodeURIComponent(id)}/reference.jpg`;
  renderTable();
  document.getElementById("result").textContent = "Load the saved points or click at least 4 points, then fit.";
}

document.getElementById("load").addEventListener("click", async () => {
  const id = document.getElementById("cam").value;
  const d = await api(`/api/cameras/${encodeURIComponent(id)}/calibration`);
  if (!d.points) { document.getElementById("result").textContent = "No saved points for this camera yet."; return; }
  spec = d.points; fit = null; renderTable(); draw();
});

document.getElementById("clear").addEventListener("click", () => { spec.points = []; fit = null; renderTable(); draw(); });

document.getElementById("fit").addEventListener("click", async () => {
  const lines = [...new Set(spec.points.map(p => p.line))].filter(l => l !== 0);
  const body = { ...spec, unknown_phase_lines: spec.unknown_phase_lines && spec.unknown_phase_lines.length ? spec.unknown_phase_lines : lines };
  const out = document.getElementById("result");
  try {
    fit = await api("/api/calibrate/fit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const scale = fit.scale_uncertainty ? `${(fit.scale_uncertainty * 100).toFixed(1)}%` : "unknown";
    out.innerHTML = `Fitted ${spec.points.length} points. Reprojection error ${fit.rmse_m.toFixed(3)} m RMS,
      worst ${fit.max_err_m.toFixed(3)} m. Speed scale uncertainty ${scale}.
      ${spec.transverse_lines ? "" : " No transverse lines given, so a sideways shear is unconstrained: along-lane speeds are fine, lane changes are not."}`;
  } catch (e) {
    fit = null;
    out.textContent = `Fit failed: ${e.message}`;
  }
  renderTable(); draw();
});

document.getElementById("download").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(spec, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${spec.camera_id || "camera"}.points.json`;
  a.click();
});

(async function main() {
  const d = await api("/api/cameras");
  const sel = document.getElementById("cam");
  const cams = d.cameras.filter(c => c.calibrated);
  sel.innerHTML = cams.map(c => `<option value="${c.id}">${c.id}: ${c.name}</option>`).join("");
  sel.addEventListener("change", () => loadCamera(sel.value));
  if (cams.length) loadCamera(cams[0].id);
  else document.getElementById("result").textContent = "No camera has a reference frame yet (data/calibration/<id>.ref.jpg).";
})();
