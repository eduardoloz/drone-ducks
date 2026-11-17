// LIE traffic speed analytics: map page.
// Plain JavaScript on purpose: no build step, easy to read and change.
// Data comes from the FastAPI endpoints in cctv/web/app.py.

const TZ = "America/New_York";
const RAMP = ["#00204d", "#31446b", "#666970", "#958f78", "#cbba69", "#ffea46"]; // slow -> fast
const RAMP_MIN = 20, RAMP_MAX = 75; // mph range the colour ramp covers

const SUITABILITY = {
  measured_good: "Video, measured good enough for per-vehicle speeds",
  measured_marginal: "Video, marginal (low resolution or dropped frames)",
  measured_unusable: "Video, unusable when measured",
  inconclusive_dark: "Video, measured after dark: re-measure in daylight",
  offline_when_measured: "Video, offline when measured",
  candidate_video: "Video, not measured yet",
  candidate_video_secondary: "Video of a ramp, service road or cross street",
  snapshot_only: "Still images every ~2 s: counts at best, no speeds",
  excluded: "Excluded",
};

let map, cameras = [], markers = {}, selectedId = null;

// ---------------------------------------------------------------- helpers
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
function speedColour(mph) {
  const x = Math.min(1, Math.max(0, (mph - RAMP_MIN) / (RAMP_MAX - RAMP_MIN))) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i;
  const a = hexToRgb(RAMP[i]), b = hexToRgb(RAMP[i + 1]);
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
function localTime(iso) {
  if (!iso) return "–";
  return new Intl.DateTimeFormat("en-US", { timeZone: TZ, month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit" }).format(new Date(iso));
}
const fmt = (x, d = 0) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

// Light at the camera when the clip was recorded (sun elevation, cctv/daylight.py). The detector is
// trained on daylight photos; at night it sees headlights, so we say so next to the numbers.
function lightNote(run) {
  if (!run || !run.light || run.light === "day") return "";
  const sun = run.sun_elev_deg !== null && run.sun_elev_deg !== undefined ? ` (sun ${fmt(run.sun_elev_deg, 1)}°)` : "";
  const what = run.light === "night"
    ? "Recorded at night. Vehicles show mostly as headlights; detection and tracking are much weaker than in daylight, so counts are undercounts and speeds are less reliable."
    : "Recorded at twilight. Contrast is falling and headlights are on; treat these numbers with extra care.";
  return `<div class="callout small"><strong>${run.light === "night" ? "Night" : "Twilight"}${sun}.</strong> ${what}</div>`;
}
function offsetNote(run) {
  if (!run || run.view_shift_px === null || run.view_shift_px === undefined) return "";
  return `<p class="muted small">The camera sat ${fmt(run.view_shift_px, 1)} px off its calibrated preset; the calibration was shifted to match.</p>`;
}

// -------------------------------------------------------------------- map
function initMap() {
  map = L.map("map", { zoomSnap: 0.5 }).setView([40.79, -73.35], 10);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
}

function markerStyle(cam) {
  const latest = cam.latest_valid;
  if (latest && latest.median_mph !== undefined) {
    return { radius: 10, fillColor: speedColour(latest.median_mph), fillOpacity: 0.95,
             color: cam.calibrated ? "#ffffff" : "#33363b", weight: cam.calibrated ? 3 : 1 };
  }
  if (cam.feed_type === "jpeg") return { radius: 3.5, fillColor: "#9aa0a6", fillOpacity: 0.8, color: "#62666d", weight: 1 };
  const good = cam.speed_suitability === "measured_good";
  return { radius: good ? 6 : 5, fillColor: "#ffffff", fillOpacity: good ? 0.9 : 0.5, color: "#00653a", weight: good ? 2.5 : 1.5 };
}

function drawMarkers() {
  for (const cam of cameras) {
    const m = L.circleMarker([cam.lat, cam.lon], markerStyle(cam)).addTo(map);
    const v = cam.latest_valid;
    let label = v && v.median_mph !== undefined ? `median ${v.median_mph} mph` : (SUITABILITY[cam.speed_suitability] || "");
    if (cam.latest && cam.latest.camera_moved) label += " (newest clip: camera had moved)";
    m.bindTooltip(`<strong>${esc(cam.name)}</strong><br>${esc(label)}`);
    m.on("click", () => selectCamera(cam.id));
    markers[cam.id] = m;
  }
  const withData = cameras.filter(c => c.latest_valid);
  if (withData.length) withData.forEach(c => markers[c.id].bringToFront());
  map.fitBounds(L.latLngBounds(cameras.map(c => [c.lat, c.lon])), { padding: [20, 20] });
}

// ------------------------------------------------------------------ panel
function renderOverview(data) {
  const counts = data.summary.by_speed_suitability;
  const rows = Object.entries(counts).sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `<tr><td>${esc(SUITABILITY[k] || k)}</td><td class="num">${n}</td></tr>`).join("");
  const live = cameras.filter(c => c.latest_valid);
  const list = live.map(c => `<li><button data-cam="${esc(c.id)}"><strong>${esc(c.name)}</strong><br>
      <span class="muted small">median ${fmt(c.latest_valid.median_mph)} mph from ${c.latest_valid.with_speed} vehicles, limit ${c.posted_limit_mph}</span></button></li>`).join("");
  document.getElementById("panel").innerHTML = `
    <h2>${data.summary.total} cameras on the LIE</h2>
    <p class="muted">Queens-Midtown Tunnel to Riverhead. Pick a camera on the map. Filled circles have processed footage
      and are coloured by median speed; a white ring means the camera is calibrated.</p>
    <div class="legend"><span>${RAMP_MIN} mph</span><span class="ramp" aria-hidden="true"></span><span>${RAMP_MAX} mph</span></div>
    <h3>Cameras with speed estimates</h3>
    ${list ? `<ul class="cam-list">${list}</ul>` : `<p>No footage processed yet. Run <code>python -m cctv.pipeline</code> (see README).</p>`}
    <h3>What each camera can give us</h3>
    <table><tbody>${rows}</tbody></table>
    <div class="callout small">Only real video (about 15 frames per second) supports per-vehicle speeds. Still-image
      cameras update every ~2 seconds, too slowly to follow a car, so they can at most count traffic.</div>`;
  document.querySelectorAll("[data-cam]").forEach(b => b.addEventListener("click", () => selectCamera(b.dataset.cam)));
}

async function selectCamera(id) {
  selectedId = id;
  if (replayTimer) { clearInterval(replayTimer); replayTimer = null; }
  const d = await api(`/api/cameras/${encodeURIComponent(id)}`);
  const cam = d.camera, cal = d.calibration, newest = d.runs[0], latest = d.latest_valid;
  const meas = cam.measured || {};
  map.panTo([cam.lat, cam.lon]);
  const feed = cam.feed.type === "hls"
    ? `HLS video${meas.width ? `, ${meas.width}×${meas.height}` : ""}${meas.unique_fps ? `, ${fmt(meas.unique_fps, 1)} unique frames/s` : ""}`
    : "Still image (NYC DOT), refreshed about every 2 s";
  const calText = cal ? `RMSE ${fmt(cal.rmse_m, 2)} m on ${esc(cal.created_utc.slice(0, 10))}; speed scale ±${fmt(cal.scale_uncertainty * 100, 1)}%` : "Not calibrated";
  let html = `
    <div class="cam-head">
      <div class="grow">
        <h2>${esc(cam.name)}</h2>
        <p class="muted">${esc(cam.county)}${cam.exit ? `, exit ${esc(cam.exit)}` : ""}, camera ${esc(cam.id)} (${esc(cam.owner)})</p>
      </div>
      <div class="limit-sign" title="Posted limit used for flags"><span>SPEED<br>LIMIT</span><strong>${cam.posted_limit_mph}</strong></div>
    </div>
    <dl class="facts">
      <dt>Feed</dt><dd>${feed}</dd>
      <dt>Speed use</dt><dd>${esc(SUITABILITY[cam.speed_suitability] || cam.speed_suitability)}</dd>
      <dt>Calibration</dt><dd>${calText}</dd>
      ${(d.views || []).length > 1 ? `<dt>Known views</dt><dd>${d.views.map(v => `${esc(v.view_id)}${v.calibrated ? " (calibrated)" : " (not calibrated)"}`).join(", ")}
        <span class="muted small">: pan-tilt-zoom presets; each clip is matched to one</span></dd>` : ""}
      <dt>Position</dt><dd>${fmt(cam.chainage_km, 1)} km from the Queens-Midtown Tunnel</dd>
    </dl>`;
  if (newest && newest.camera_moved) {
    const where = newest.view_id ? `shows preset ${esc(newest.view_id)}, which is not calibrated yet`
                                 : "matches none of the known views";
    html += `<div class="callout"><strong>The camera has been re-aimed.</strong> The newest clip
      (${localTime(newest.clip_start_utc)}) ${where}, so its ${newest.vehicles} tracks got no
      speeds${newest.flow_veh_per_hour ? ` (counts only: about ${newest.flow_veh_per_hour.toLocaleString("en-US")} vehicles/hour tracked)` : ""}.
      ${latest ? "Below: the newest clip that matched the calibration." : "Recalibrate this view to get speeds."}</div>
      ${lightNote(newest)}`;
  }
  if (latest) {
    const pct = latest.with_speed ? fmt(latest.pct_over_limit) : "–";
    html += `
      <h3>Latest clip with speeds</h3>
      <p class="muted small">${localTime(latest.clip_start_utc)}, ${fmt(latest.duration_s)} s of video</p>
      ${lightNote(latest)}${offsetNote(latest)}
      <div class="stats">
        <div class="stat"><strong>${fmt(latest.median_mph)}</strong><span>median mph</span></div>
        <div class="stat"><strong>${fmt(latest.p85_mph)}</strong><span>85th percentile</span></div>
        <div class="stat"><strong>${latest.with_speed}/${latest.vehicles}</strong><span>vehicles with a speed</span></div>
        <div class="stat"><strong>${pct}%</strong><span>flagged over limit</span></div>
      </div>
      <h3>Speed distribution</h3><div id="hist"></div>
      <h3>Tracks on the calibrated stretch</h3>
      <p class="muted small">Replay of the latest clip's tracks drawn on the reference frame. Labels are estimates in mph.</p>
      <div id="overlay"></div>
      <h3>Speed estimates from this camera</h3><div id="cam-events"></div>`;
  } else if (!newest) {
    html += `<div class="callout">No processed footage for this camera yet.${cam.feed.type === "hls" ? " Record a clip and run the pipeline (see the README)." : ""}</div>`;
  }
  if (cam.feed.type === "hls") {
    html += `<h3>Live video</h3>
      <p class="muted small">Streams directly from NYSDOT's server. Live overlays need the real-time worker (see docs/ARCHITECTURE.md).</p>
      <button id="live-btn">Watch live video</button><div id="live" class="overlay-wrap" hidden></div>`;
  }
  document.getElementById("panel").innerHTML = html;
  document.getElementById("panel").scrollTop = 0;
  if (latest) {
    const s = await api(`/api/stats?camera_id=${encodeURIComponent(id)}`);
    renderHistogram(document.getElementById("hist"), s.histogram, cam.posted_limit_mph);
    try {
      renderOverlay(document.getElementById("overlay"), await api(`/api/runs/${latest.run_id}/overlay`));
    } catch (e) {  // 403 on a public site while the clip is younger than the publication delay
      document.getElementById("overlay").innerHTML = `<p class="muted">Track replay is published with a delay; check back later.</p>`;
    }
    const ev = await api(`/api/events?camera_id=${encodeURIComponent(id)}&limit=40`);
    document.getElementById("cam-events").innerHTML = eventsTable(ev.events, false);
  }
  const btn = document.getElementById("live-btn");
  if (btn) btn.addEventListener("click", () => watchLive(cam.feed.url, btn));
}

// ------------------------------------------------------------ histogram
function renderHistogram(el, hist, limit) {
  if (!hist.length) { el.innerHTML = `<p class="muted">No accepted speeds.</p>`; return; }
  const W = 520, H = 170, pad = { l: 30, r: 10, t: 10, b: 26 };
  const lo = hist[0].from, hi = hist[hist.length - 1].to, maxN = Math.max(...hist.map(h => h.count));
  const x = v => pad.l + (v - lo) / (hi - lo) * (W - pad.l - pad.r);
  const y = n => H - pad.b - n / maxN * (H - pad.t - pad.b);
  const bars = hist.map(h => `<rect x="${x(h.from) + 1}" y="${y(h.count)}" width="${x(h.to) - x(h.from) - 2}" height="${H - pad.b - y(h.count)}" fill="${speedColour((h.from + h.to) / 2)}"><title>${h.from}-${h.to} mph: ${h.count}</title></rect>`).join("");
  const ticks = hist.map(h => `<text x="${x(h.from)}" y="${H - 8}" text-anchor="middle">${h.from}</text>`).join("") + `<text x="${x(hi)}" y="${H - 8}" text-anchor="middle">${hi}</text>`;
  const lim = limit >= lo && limit <= hi ? `<line x1="${x(limit)}" x2="${x(limit)}" y1="${pad.t}" y2="${H - pad.b}" stroke="currentColor" stroke-dasharray="4 3"/><text x="${x(limit) + 4}" y="${pad.t + 10}">limit ${limit}</text>` : "";
  el.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Histogram of estimated speeds">${bars}${ticks}${lim}
    <text x="4" y="${pad.t + 8}">${maxN}</text></svg><p class="muted small">Vehicles per ${hist[0].to - hist[0].from} mph bin. Dashed line: posted limit.</p>`;
}

// --------------------------------------------------------------- overlay
// Replays the latest clip: a slider (or Play) moves through time and we draw each
// vehicle's last 1.5 s of positions, with its speed estimate at the current point.
let replayTimer = null;

function renderOverlay(el, ov) {
  const [w, h] = ov.image_size;
  const dur = Math.ceil(ov.duration_s || Math.max(0, ...ov.tracks.map(t => t.t_end || 0)));
  el.innerHTML = `<div class="overlay-wrap"><img src="${ov.reference_url}" alt="Reference frame of the camera view, empty road" width="${w}" height="${h}">
      <svg viewBox="0 0 ${w} ${h}" aria-hidden="true"></svg></div>
    <div class="row-replay legend">
      <button id="replay-play">Play</button>
      <input id="replay-t" type="range" min="0" max="${dur}" step="0.1" value="0" aria-label="Replay time in seconds" style="flex:1">
      <span id="replay-label">0.0 s</span>
    </div>
    <div class="legend"><span>Coloured: accepted estimate (mph).</span><span>Dashed grey: rejected track.</span><span><span class="flag">yellow</span> flagged over limit.</span></div>`;
  const svg = el.querySelector("svg"), slider = el.querySelector("#replay-t"), label = el.querySelector("#replay-label");
  const roi = ov.roi_polygon ? `<polygon points="${ov.roi_polygon.map(p => p.join(",")).join(" ")}" fill="none" stroke="#ffffff" stroke-opacity="0.6" stroke-dasharray="6 4"/>` : "";
  const draw = T => {
    const parts = [roi];
    for (const t of ov.tracks) {
      const trail = t.points.filter(p => p[2] <= T && p[2] >= T - 1.5);
      if (trail.length < 2 || T > t.t_end + 0.3) continue;
      const pts = trail.map(p => `${p[0]},${p[1]}`).join(" ");
      const head = trail[trail.length - 1];
      const colour = t.accepted ? (t.over_limit ? "#f2b705" : speedColour(t.speed_mph)) : "#9aa0a6";
      parts.push(`<polyline points="${pts}" fill="none" stroke="${colour}" stroke-width="3" stroke-dasharray="${t.accepted ? "" : "4 3"}"/>`);
      parts.push(`<circle cx="${head[0]}" cy="${head[1]}" r="4" fill="${colour}" stroke="#000"/>`);
      if (t.accepted) parts.push(`<text x="${head[0] + 6}" y="${head[1] - 6}">${fmt(t.speed_mph)}</text>`);
    }
    svg.innerHTML = parts.join("");
    label.textContent = `${Number(T).toFixed(1)} s`;
  };
  slider.addEventListener("input", () => draw(+slider.value));
  el.querySelector("#replay-play").addEventListener("click", ev => {
    if (replayTimer) { clearInterval(replayTimer); replayTimer = null; ev.target.textContent = "Play"; return; }
    ev.target.textContent = "Pause";
    replayTimer = setInterval(() => {
      let T = +slider.value + 0.1;
      if (T > dur) T = 0;
      slider.value = T; draw(T);
    }, 100);
  });
  draw(+slider.value || Math.min(5, dur));
  slider.value = Math.min(5, dur);
}

// ---------------------------------------------------------------- events
function eventsTable(events, withCamera) {
  if (!events.length) return `<p class="muted">No speed estimates yet.</p>`;
  const rows = events.map(e => {
    const overBy = e.over_limit ? e.speed_mph - e.margin_mph - e.limit_mph : null;
    return `<tr class="${e.over_limit ? "flagged" : ""}">
      <td>${localTime(e.ts_utc)}</td>${withCamera ? `<td>${esc(e.camera_name)}</td>` : ""}
      <td>${esc(e.cls)}</td><td class="num">${fmt(e.speed_mph, 1)} ± ${fmt(e.sigma_mph, 1)}</td>
      <td class="num">${fmt(e.limit_mph)}</td>
      <td>${overBy !== null ? `<span class="flag">over by ≥ ${fmt(overBy, 1)}</span>` : ""}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>Time (NY)</th>${withCamera ? "<th>Camera</th>" : ""}<th>Class</th>
    <th class="num">Estimate, mph</th><th class="num">Limit</th><th>Flag</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function loadEventsView() {
  const over = document.getElementById("over-only").checked;
  const ev = await api(`/api/events?limit=300${over ? "&over_only=true" : ""}`);
  document.getElementById("events-table").innerHTML = eventsTable(ev.events, true);
}

// ------------------------------------------------------------------ live
function watchLive(url, btn) {
  const box = document.getElementById("live");
  box.hidden = false;
  btn.disabled = true;
  const video = document.createElement("video");
  video.controls = true; video.muted = true; video.autoplay = true; video.playsInline = true;
  box.appendChild(video);
  const start = () => {
    if (window.Hls && Hls.isSupported()) { const hls = new Hls(); hls.loadSource(url); hls.attachMedia(video); }
    else if (video.canPlayType("application/vnd.apple.mpegurl")) video.src = url;
    else box.innerHTML = `<p class="muted">This browser cannot play HLS video.</p>`;
  };
  if (window.Hls) return start();
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/hls.js/1.5.13/hls.min.js";
  s.onload = start;
  document.head.appendChild(s);
}

// ---------------------------------------------------------------- routing
function route() {
  const view = location.hash === "#events" ? "events" : "map";
  document.getElementById("view-map").hidden = view !== "map";
  document.getElementById("view-events").hidden = view !== "events";
  document.querySelectorAll("nav a[data-view]").forEach(a => a.classList.toggle("active", a.dataset.view === view));
  if (view === "events") loadEventsView();
  else if (map) setTimeout(() => map.invalidateSize(), 0);
}

async function main() {
  initMap();
  const data = await api("/api/cameras");
  cameras = data.cameras;
  drawMarkers();
  renderOverview(data);
  document.getElementById("over-only").addEventListener("change", loadEventsView);
  window.addEventListener("hashchange", route);
  route();
}

main().catch(err => {
  document.getElementById("panel").innerHTML = `<p>Could not load data: ${esc(err.message)}. Is the API running on this port?</p>`;
});
