const statusEl = document.getElementById("status");
const engagementValue = document.getElementById("engagementValue");
const countAttentive = document.getElementById("countAttentive");
const countPartial = document.getElementById("countPartial");
const countAway = document.getElementById("countAway");
const facesDetected = document.getElementById("facesDetected");
const inferenceFps = document.getElementById("inferenceFps");
const calibrateBtn = document.getElementById("calibrateBtn");
const recordBtn = document.getElementById("recordBtn");
const calibrateNote = document.getElementById("calibrateNote");
const reportLink = document.getElementById("reportLink");
const canvas = document.getElementById("sparkline");
const ctx = canvas.getContext("2d");

const history = [];
const HISTORY_MAX = 120;
let isRecording = false;

function engagementColor(pct) {
  if (pct >= 70) return "#4cc38a";
  if (pct >= 40) return "#f0a63f";
  return "#e5484d";
}

function drawSparkline() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;

  ctx.beginPath();
  history.forEach((pct, i) => {
    const x = (i / (HISTORY_MAX - 1)) * w;
    const y = h - (pct / 100) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = engagementColor(history[history.length - 1]);
  ctx.lineWidth = 2;
  ctx.stroke();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    statusEl.textContent = "connected";
    statusEl.className = "status ok";
  };

  ws.onclose = () => {
    statusEl.textContent = "disconnected — retrying…";
    statusEl.className = "status error";
    setTimeout(connect, 1500);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.error) {
      statusEl.textContent = `error: ${data.error}`;
      statusEl.className = "status error";
    }

    isRecording = !!data.recording;
    recordBtn.textContent = isRecording ? "Stop Session" : "Start Session";
    recordBtn.classList.toggle("recording", isRecording);

    if (data.calibrating) {
      calibrateNote.textContent = "Calibrating… hold still and face the screen.";
    } else if (data.calibrated) {
      calibrateNote.textContent = "Calibrated.";
    } else {
      calibrateNote.textContent = "Not calibrated — recommended before a session.";
    }

    const m = data.metrics;
    if (!m) return;

    const pct = Math.round(m.engagement_smoothed_pct);
    engagementValue.textContent = `${pct}%`;
    engagementValue.style.color = engagementColor(pct);

    countAttentive.textContent = m.attentive;
    countPartial.textContent = m.partial;
    countAway.textContent = m.away;
    facesDetected.textContent = m.faces_detected;
    inferenceFps.textContent = m.inference_fps.toFixed(1);

    history.push(m.engagement_smoothed_pct);
    if (history.length > HISTORY_MAX) history.shift();
    drawSparkline();
  };
}

calibrateBtn.addEventListener("click", async () => {
  await fetch("/api/calibrate", { method: "POST" });
});

recordBtn.addEventListener("click", async () => {
  reportLink.innerHTML = "";
  if (isRecording) {
    const res = await fetch("/api/session/stop", { method: "POST" });
    const data = await res.json();
    if (data.report_url) {
      reportLink.innerHTML = `Session saved. <a href="${data.report_url}" target="_blank">View report</a>`;
    }
  } else {
    await fetch("/api/session/start", { method: "POST" });
  }
});

connect();
