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

canvas.width = 640;
canvas.height = 480;

const TARGET_FRAMES = 30;
const MAX_CAPTURE_FRAMES = 30;
const MIN_VALID_FRAMES = 20;
const BACKEND_URL = 'http://127.0.0.1:5000/predict';

let rawBuffer = [];
let isPredicting = false;
let isSending = false;

let predictedWords = [];
let latestCompletedConfidence = '--';

statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
predictionText.textContent = 'Waiting...';
confidenceText.textContent = 'Confidence: --';

function updateProgressUI(current, total) {
  const percent = Math.min((current / total) * 100, 100);
  frameCountText.textContent = `Captured valid frames: ${current} / ${total}`;
  progressPercent.textContent = `${Math.round(percent)}%`;
  progressFill.style.width = `${percent}%`;
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

function resetCaptureState() {
  rawBuffer = [];
  updateProgressUI(0, MAX_CAPTURE_FRAMES);
}

function flattenLandmarks(landmarks, count, useVisibility = false) {
  const arr = [];

  if (landmarks && landmarks.length) {
    for (let i = 0; i < Math.min(landmarks.length, count); i++) {
      const lm = landmarks[i];
      arr.push([
        lm.x || 0,
        lm.y || 0,
        useVisibility ? (lm.visibility ?? 1.0) : 1.0
      ]);
    }
  }

  while (arr.length < count) {
    arr.push([0, 0, 0]);
  }

  return arr.slice(0, count);
}

function extractUniSignFrame(results) {
  const pose = results.poseLandmarks || [];
  const leftHand = results.leftHandLandmarks || [];
  const rightHand = results.rightHandLandmarks || [];
  const face = results.faceLandmarks || [];

  const selectedPose = [
    pose[0],   // nose
    pose[11],  // left shoulder
    pose[12],  // right shoulder
    pose[13],  // left elbow
    pose[14],  // right elbow
    pose[15],  // left wrist
    pose[16],  // right wrist
    pose[23],  // left hip
    pose[24]   // right hip
  ];

  return {
    body: flattenLandmarks(selectedPose, 9, true),
    left: flattenLandmarks(leftHand, 21),
    right: flattenLandmarks(rightHand, 21),
    face_all: flattenLandmarks(face.slice(0, 18), 18)
  };
}

function hasValidHands(results) {
  return !!(
    (results.leftHandLandmarks && results.leftHandLandmarks.length > 0) ||
    (results.rightHandLandmarks && results.rightHandLandmarks.length > 0)
  );
}

function uniformSampleFrames(frames, targetCount = TARGET_FRAMES) {
  if (!Array.isArray(frames) || frames.length < targetCount) {
    return null;
  }

  const sampled = [];
  for (let i = 0; i < targetCount; i++) {
    const idx = Math.round((i * (frames.length - 1)) / (targetCount - 1));
    sampled.push(frames[idx]);
  }

  return sampled;
}

function convertToPayload(sampledFrames) {
  return {
    body: sampledFrames.map(f => f.body),
    left: sampledFrames.map(f => f.left),
    right: sampledFrames.map(f => f.right),
    face_all: sampledFrames.map(f => f.face_all),
    attention_mask: new Array(sampledFrames.length).fill(1)
  };
}

async function sendToBackend(sampledFrames) {
  if (isSending) return;

  try {
    isSending = true;
    statusText.textContent = 'Sending live UniSign sequence to backend...';
    overlayStatus.textContent = 'Sending';

    const payload = convertToPayload(sampledFrames);

    const response = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }

    const finalPrediction = result.prediction || 'Unknown';
    const finalConfidence = result.confidence ?? '--';

    predictionText.textContent = finalPrediction;
    confidenceText.textContent = `Confidence: ${finalConfidence}`;

    predictedWords.push(finalPrediction);
    latestCompletedConfidence = finalConfidence;
    renderPredictionHistory();

    statusText.textContent = result.message || 'Prediction received.';
    overlayStatus.textContent = 'Completed';

  } catch (err) {
    console.error(err);
    predictionText.textContent = 'Error';
    confidenceText.textContent = 'Confidence: --';
    statusText.textContent = `Backend error: ${err.message}`;
    overlayStatus.textContent = 'Error';
  } finally {
    resetCaptureState();
    isPredicting = false;
    isSending = false;
    startBtn.disabled = false;
    stopBtn.disabled = false;
  }
}

async function finalizePrediction() {
  if (rawBuffer.length < MIN_VALID_FRAMES) {
    statusText.textContent = `Not enough valid frames. Need ${MIN_VALID_FRAMES}, got ${rawBuffer.length}.`;
    predictionText.textContent = 'Try again';
    confidenceText.textContent = 'Confidence: --';
    overlayStatus.textContent = 'Too few frames';
    isPredicting = false;
    startBtn.disabled = false;
    stopBtn.disabled = false;
    return;
  }

  const sampledFrames = uniformSampleFrames(rawBuffer, TARGET_FRAMES);

  if (!sampledFrames) {
    statusText.textContent = 'Failed to build valid 30-frame sequence.';
    predictionText.textContent = 'Try again';
    overlayStatus.textContent = 'Sampling failed';
    return;
  }

  await sendToBackend(sampledFrames);
}

startBtn.addEventListener('click', () => {
  if (isSending) return;

  rawBuffer = [];
  isPredicting = true;

  predictionText.textContent = 'Recording...';
  confidenceText.textContent = 'Confidence: --';
  statusText.textContent = 'Recording gesture. Perform one complete sign.';
  overlayStatus.textContent = 'Recording';

  startBtn.disabled = true;
  stopBtn.disabled = false;

  updateProgressUI(0, MAX_CAPTURE_FRAMES);
});

stopBtn.addEventListener('click', async () => {
  if (!isPredicting || isSending) return;

  isPredicting = false;
  statusText.textContent = 'Processing gesture sequence...';
  overlayStatus.textContent = 'Processing';

  startBtn.disabled = true;
  stopBtn.disabled = true;

  await finalizePrediction();
});

const holistic = new Holistic({
  locateFile: file =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

holistic.onResults(async (results) => {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.poseLandmarks) {
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
      color: '#00BFFF',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.poseLandmarks, {
      color: '#FFD700',
      radius: 2
    });
  }

  if (results.leftHandLandmarks) {
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.leftHandLandmarks, {
      color: '#FF0000',
      radius: 2
    });
  }

  if (results.rightHandLandmarks) {
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
      color: '#9000ff',
      lineWidth: 2
    });
    drawLandmarks(ctx, results.rightHandLandmarks, {
      color: '#FF0000',
      radius: 2
    });
  }

  ctx.restore();

  if (isPredicting && !isSending) {
    if (hasValidHands(results)) {
      const frame = extractUniSignFrame(results);
      rawBuffer.push(frame);

      updateProgressUI(rawBuffer.length, MAX_CAPTURE_FRAMES);
      statusText.textContent = `Recording valid gesture frames... ${rawBuffer.length}/${MAX_CAPTURE_FRAMES}`;
      overlayStatus.textContent = 'Recording';

      if (rawBuffer.length >= MAX_CAPTURE_FRAMES) {
        isPredicting = false;
        startBtn.disabled = true;
        stopBtn.disabled = true;
        statusText.textContent = 'Max capture reached. Processing automatically...';
        overlayStatus.textContent = 'Auto processing';
        await finalizePrediction();
      }
    } else {
      statusText.textContent = 'No hands detected. Keep your hand visible.';
      overlayStatus.textContent = 'Waiting for hands';
    }
  }
});

const camera = new Camera(video, {
  onFrame: async () => {
    await holistic.send({ image: video });
  },
  width: 640,
  height: 480
});

camera.start()
  .then(() => {
    statusText.textContent = 'Camera started. Click "Start Prediction" to begin.';
    overlayStatus.textContent = 'Camera ready';
  })
  .catch((err) => {
    console.error('Camera start error:', err);
    statusText.textContent = 'Camera failed: ' + err.message;
    overlayStatus.textContent = 'Camera blocked';
  });

updateProgressUI(0, MAX_CAPTURE_FRAMES);
renderPredictionHistory();