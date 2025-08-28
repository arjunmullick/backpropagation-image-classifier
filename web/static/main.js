const runSelect = document.getElementById('runSelect');
const refreshRunsBtn = document.getElementById('refreshRuns');
const lossCanvas = document.getElementById('lossChart');
const accCanvas = document.getElementById('accChart');
const curvesImg = document.getElementById('curvesImg');
const fileInput = document.getElementById('fileInput');
const drawCanvas = document.getElementById('drawCanvas');
const clearCanvasBtn = document.getElementById('clearCanvas');
const predictCanvasBtn = document.getElementById('predictCanvas');
const predDiv = document.getElementById('prediction');
const probsDiv = document.getElementById('probs');

async function fetchRuns() {
  const res = await fetch('/runs');
  const data = await res.json();
  runSelect.innerHTML = '';
  data.runs.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r; opt.textContent = r; runSelect.appendChild(opt);
  });
  if (data.runs.length > 0) {
    runSelect.value = data.runs[data.runs.length - 1];
    updateCurvesImg();
  }
}

function drawSeries(ctx, values, color, label) {
  if (!values || values.length === 0) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0,0,W,H);
  // axes
  ctx.strokeStyle = '#aaa';
  ctx.beginPath(); ctx.moveTo(40, 10); ctx.lineTo(40, H-20); ctx.lineTo(W-10, H-20); ctx.stroke();
  // scale
  const maxVal = Math.max(...values) || 1;
  const minVal = Math.min(...values) || 0;
  const pad = 0.05 * (maxVal - minVal || 1);
  const vMax = maxVal + pad, vMin = Math.max(0, minVal - pad);
  const plotW = W - 60, plotH = H - 40;
  ctx.strokeStyle = color;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 40 + (i / Math.max(1, values.length - 1)) * plotW;
    const y = (H - 20) - ((v - vMin) / (vMax - vMin)) * plotH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#333';
  ctx.fillText(label, 45, 20);
}

async function pollMetrics() {
  const run = runSelect.value;
  if (!run) return;
  try {
    const res = await fetch(`/metrics?run=${encodeURIComponent(run)}`);
    if (!res.ok) return;
    const m = await res.json();
    const lossCtx = lossCanvas.getContext('2d');
    const accCtx = accCanvas.getContext('2d');
    drawSeries(lossCtx, m.train_loss, '#0070f3', 'Loss (train)');
    drawSeries(lossCtx, m.val_loss, '#f39c12', ''); // overlay second series
    drawSeries(accCtx, m.train_acc, '#0070f3', 'Acc (train)');
    drawSeries(accCtx, m.val_acc, '#f39c12', '');
  } catch (e) { /* ignore */ }
}

function updateCurvesImg() {
  const run = runSelect.value;
  if (!run) return;
  curvesImg.src = `/curves.png?run=${encodeURIComponent(run)}&_=${Date.now()}`;
  curvesImg.style.display = 'block';
}

refreshRunsBtn.addEventListener('click', fetchRuns);
runSelect.addEventListener('change', () => { updateCurvesImg(); pollMetrics(); });

// Prediction via file upload
fileInput.addEventListener('change', async () => {
  if (!fileInput.files || fileInput.files.length === 0) return;
  const form = new FormData();
  form.append('run', runSelect.value);
  form.append('image', fileInput.files[0]);
  const res = await fetch('/predict', { method: 'POST', body: form });
  const data = await res.json();
  showPrediction(data);
});

// Drawing canvas setup
const dctx = drawCanvas.getContext('2d');
dctx.fillStyle = '#000'; dctx.fillRect(0,0,drawCanvas.width, drawCanvas.height);
let drawing = false; let last = null;

function drawDot(x, y) {
  dctx.fillStyle = '#fff';
  dctx.beginPath();
  dctx.arc(x, y, 6, 0, Math.PI*2);
  dctx.fill();
}

drawCanvas.addEventListener('pointerdown', e => { drawing = true; const r = drawCanvas.getBoundingClientRect(); drawDot(e.clientX - r.left, e.clientY - r.top); });
drawCanvas.addEventListener('pointermove', e => { if (!drawing) return; const r = drawCanvas.getBoundingClientRect(); drawDot(e.clientX - r.left, e.clientY - r.top); });
drawCanvas.addEventListener('pointerup', () => drawing = false);
drawCanvas.addEventListener('pointerleave', () => drawing = false);

clearCanvasBtn.addEventListener('click', () => { dctx.fillStyle = '#000'; dctx.fillRect(0,0,drawCanvas.width, drawCanvas.height); predDiv.textContent=''; probsDiv.textContent=''; });

predictCanvasBtn.addEventListener('click', async () => {
  const dataURL = drawCanvas.toDataURL('image/png');
  const form = new FormData();
  form.append('run', runSelect.value);
  form.append('image_base64', dataURL);
  const res = await fetch('/predict', { method: 'POST', body: form });
  const data = await res.json();
  showPrediction(data);
});

function showPrediction(data) {
  if (data.error) { predDiv.textContent = 'Error: ' + data.error; return; }
  predDiv.textContent = 'Prediction: ' + data.prediction;
  const lines = data.probs.map(o => `  ${o.label}: ${o.p.toFixed(4)}`);
  probsDiv.textContent = lines.join('\n');
}

// init
fetchRuns();
setInterval(pollMetrics, 2000);

