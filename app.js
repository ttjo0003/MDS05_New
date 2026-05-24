const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const statusText = document.getElementById('status');
const frameCountText = document.getElementById('frameCount');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

const progressFill = document.getElementById('progressFill');
const progressPercent = document.getElementById('progressPercent');
const overlayStatus = document.getElementById('overlayStatus');

const predictionHistoryBox = document.getElementById('predictionHistory');
const previousConfidenceDisplay = document.getElementById('previousConfidence');

const RECORD_SECONDS = 3;
let recordTimer = null;
let progressTimer = null;

canvas.width = 640;
canvas.height = 480;

const BACKEND_URL = 'http://127.0.0.1:5000/predict';

let mediaRecorder = null;
let recordedChunks = [];
let predictedWords = [];
let latestCompletedConfidence = '--';
let cameraStream = null;

statusText.textContent = 'Starting camera...';
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

function updateProgressUI(text) {
  frameCountText.textContent = text;
  progressPercent.textContent = '';
  progressFill.style.width = '0%';
}

function renderPredictionHistory() {
  if (!predictionHistoryBox) return;

  if (predictedWords.length === 0) {
    predictionHistoryBox.innerHTML = `<div class="history-empty">No prediction history yet</div>`;
  } else {
    predictionHistoryBox.innerHTML = `
      <div class="history-item">
        <div class="history-word">${predictedWords.join(' ')}</div>
      </div>
    `;
  }

  if (previousConfidenceDisplay) {
    previousConfidenceDisplay.textContent = `Previous confidence: ${latestCompletedConfidence}`;
  }
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480
      },
      audio: false
    });

    video.srcObject = cameraStream;

    await video.play();

    statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
    overlayStatus.textContent = 'Camera ready';
    updateProgressUI('Ready');

    function drawPreview() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      requestAnimationFrame(drawPreview);
    }

    drawPreview();

  } catch (err) {
    console.error('Camera start error:', err);
    statusText.textContent = 'Camera failed: ' + err.message;
    overlayStatus.textContent = 'Camera blocked';
  }
}

startBtn.addEventListener('click', () => {
  if (!cameraStream) {
    statusText.textContent = 'Camera is not ready yet.';
    return;
  }

  recordedChunks = [];

  mediaRecorder = new MediaRecorder(cameraStream, {
    mimeType: 'video/webm'
  });

  mediaRecorder.ondataavailable = event => {
    console.log("Chunk received:", event.data.size);

    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    console.log("Recorder stopped");
    console.log("Recorded chunks:", recordedChunks.length);

    clearTimeout(recordTimer);
    clearInterval(progressTimer);
    progressFill.style.width = '100%';
    progressPercent.textContent = '100%';
    try {
      startBtn.disabled = true;
      stopBtn.disabled = true;

      statusText.textContent = 'Sending video to backend...';
      overlayStatus.textContent = 'Sending';

      const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });

      const formData = new FormData();
      formData.append('video', videoBlob, 'input.webm');
      console.log("Video blob size:", videoBlob.size);

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP ${response.status}`);
      }

      const finalPrediction = result.prediction || 'Unknown';
      const finalConfidence = result.confidence || 'Generated';

      predictionText.textContent = finalPrediction;
      confidenceText.textContent = `Confidence: ${finalConfidence}`;

      predictedWords.push(finalPrediction);
      latestCompletedConfidence = finalConfidence;
      renderPredictionHistory();

      statusText.textContent = result.message || 'Prediction received.';
      overlayStatus.textContent = 'Completed';
      updateProgressUI('Completed');

    } catch (err) {
      console.error(err);
      predictionText.textContent = 'Error';
      confidenceText.textContent = 'Confidence: --';
      statusText.textContent = `Backend error: ${err.message}`;
      overlayStatus.textContent = 'Error';
    } finally {
      startBtn.disabled = false;
      stopBtn.disabled = true;
    }
  };

  mediaRecorder.start(1000);

  let elapsed = 0;
  progressFill.style.width = '0%';
  progressPercent.textContent = '0%';

  progressTimer = setInterval(() => {
    elapsed += 0.1;
    const percent = Math.min((elapsed / RECORD_SECONDS) * 100, 100);
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${Math.round(percent)}%`;
  }, 100);

  recordTimer = setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  }, RECORD_SECONDS * 1000);
  console.log("Recorder started:", mediaRecorder.state);

  predictionText.textContent = 'Recording...';
  confidenceText.textContent = 'Confidence: --';
  statusText.textContent = 'Recording gesture video... Click Stop when done.';
  overlayStatus.textContent = 'Recording';
  updateProgressUI('Recording video...');

  startBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener('click', () => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    statusText.textContent = 'Processing video...';
    overlayStatus.textContent = 'Processing';
    mediaRecorder.stop();
  }
});

stopBtn.disabled = true;
updateProgressUI('Starting...');
renderPredictionHistory();
startCamera();