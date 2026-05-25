const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');

const statusText            = document.getElementById('status');
const frameCountText        = document.getElementById('frameCount');
const predictionText        = document.getElementById('prediction');
const confidenceText        = document.getElementById('confidence');
const confidenceBar         = document.getElementById('confidenceBar');
const startBtn              = document.getElementById('startBtn');
const stopBtn               = document.getElementById('stopBtn');
const progressFill          = document.getElementById('progressFill');
const progressPercent       = document.getElementById('progressPercent');
const overlayStatus         = document.getElementById('overlayStatus');
const predictionHistoryBox  = document.getElementById('predictionHistory');
const previousConfidence    = document.getElementById('previousConfidence');

const RECORD_SECONDS = 3;
const BACKEND_URL    = 'http://127.0.0.1:5000/predict';

let mediaRecorder  = null;
let recordedChunks = [];
let historyEntries = [];
let lastConfidence = '--';
let cameraStream   = null;
let recordTimer    = null;
let progressTimer  = null;

canvas.width  = 640;
canvas.height = 480;

// ── Helpers ──────────────────────────────────────────────

function updateProgressUI(text) {
  frameCountText.textContent  = text;
  progressPercent.textContent = '';
  progressFill.style.width    = '0%';
}

function renderHistory() {
  if (historyEntries.length === 0) {
    predictionHistoryBox.innerHTML =
      '<div class="history-empty">No prediction history yet</div>';
  } else {
    predictionHistoryBox.innerHTML = historyEntries.slice().reverse().map(e => `
      <div class="history-item">
        <div class="history-word">${e.word}</div>
        <div class="history-meta">Confidence: ${e.confidence}%</div>
      </div>
    `).join('');
  }
  if (previousConfidence) {
    previousConfidence.textContent = `Previous confidence: ${lastConfidence}`;
  }
}

// ── Camera ───────────────────────────────────────────────

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });
    video.srcObject = cameraStream;
    await video.play();

    statusText.textContent    = 'Camera ready. Click "Start Recording" to begin.';
    overlayStatus.textContent = 'Camera ready';
    updateProgressUI('Ready');

    (function drawPreview() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      requestAnimationFrame(drawPreview);
    })();

  } catch (err) {
    statusText.textContent    = 'Camera failed: ' + err.message;
    overlayStatus.textContent = 'Camera blocked';
  }
}

// ── Start button ─────────────────────────────────────────

startBtn.addEventListener('click', () => {
  if (!cameraStream) { statusText.textContent = 'Camera not ready.'; return; }

  recordedChunks = [];

  mediaRecorder = new MediaRecorder(cameraStream, { mimeType: 'video/webm' });

  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    clearTimeout(recordTimer);
    clearInterval(progressTimer);
    progressFill.style.width    = '100%';
    progressPercent.textContent = '100%';

    // ── Change to "Predicting..." as soon as recording stops ──
    predictionText.textContent = 'Predicting...';
    confidenceText.textContent = 'Confidence: --';
    confidenceBar.style.width  = '0%';
    overlayStatus.textContent  = 'Predicting';
    statusText.textContent     = 'Sending video to backend...';

    try {
      startBtn.disabled = true;
      stopBtn.disabled  = true;

      const blob     = new Blob(recordedChunks, { type: 'video/webm' });
      const formData = new FormData();
      formData.append('video', blob, 'input.webm');

      const response = await fetch(BACKEND_URL, { method: 'POST', body: formData });
      const result   = await response.json();

      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);

      const word       = result.prediction || 'Unknown';
      const confidence = result.confidence;   // number like 73.4

      predictionText.textContent = word;
      confidenceText.textContent = `Confidence: ${confidence}%`;
      confidenceBar.style.width  = `${confidence}%`;

      historyEntries.push({ word, confidence });
      lastConfidence = `${confidence}%`;
      renderHistory();

      statusText.textContent    = result.message || 'Prediction complete.';
      overlayStatus.textContent = 'Completed';
      updateProgressUI('Completed');

    } catch (err) {
      predictionText.textContent = 'Error';
      confidenceText.textContent = 'Confidence: --';
      confidenceBar.style.width  = '0%';
      statusText.textContent     = `Backend error: ${err.message}`;
      overlayStatus.textContent  = 'Error';
    } finally {
      startBtn.disabled = false;
      stopBtn.disabled  = true;
    }
  };

  mediaRecorder.start(1000);

  // Progress bar during recording
  let elapsed = 0;
  progressFill.style.width    = '0%';
  progressPercent.textContent = '0%';

  progressTimer = setInterval(() => {
    elapsed += 0.1;
    const pct = Math.min((elapsed / RECORD_SECONDS) * 100, 100);
    progressFill.style.width    = `${pct}%`;
    progressPercent.textContent = `${Math.round(pct)}%`;
  }, 100);

  recordTimer = setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  }, RECORD_SECONDS * 1000);

  predictionText.textContent = 'Recording...';
  confidenceText.textContent = 'Confidence: --';
  confidenceBar.style.width  = '0%';
  statusText.textContent     = 'Recording... Click Stop when done.';
  overlayStatus.textContent  = 'Recording';
  updateProgressUI('Recording video...');

  startBtn.disabled = true;
  stopBtn.disabled  = false;
});

// ── Stop button ──────────────────────────────────────────

stopBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
});

// ── Init ─────────────────────────────────────────────────

stopBtn.disabled = true;
updateProgressUI('Starting...');
renderHistory();
startCamera();