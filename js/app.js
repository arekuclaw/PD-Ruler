import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/vision_bundle.mjs";

// --- Constants ---------------------------------------------------------------
// MediaPipe FaceLandmarker iris landmark indices (478-point model).
const LEFT_IRIS_CENTER = 468;
const RIGHT_IRIS_CENTER = 473;
const HANDLE_RADIUS = 9;       // drawn handle radius (canvas px)
const HIT_RADIUS = 22;         // touch/click grab tolerance

// --- State -------------------------------------------------------------------
const state = {
  img: null,            // HTMLImageElement
  scale: 1,             // displayed canvas px per natural image px
  handles: {            // positions in canvas (display) coordinates
    leftPupil: null,
    rightPupil: null,
    cardA: null,
    cardB: null,
  },
  dragging: null,
};

let faceLandmarker = null;
let landmarkerReady = null;

// --- DOM ---------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const cardEdge = $("cardEdge");
const customWrap = $("customWrap");
const customMm = $("customMm");
const video = $("video");
const camStatus = $("camStatus");

// --- Model bootstrap ---------------------------------------------------------
async function initLandmarker() {
  if (landmarkerReady) return landmarkerReady;
  landmarkerReady = (async () => {
    setStatus("Loading face model…");
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
    });
    setStatus("");
  })();
  return landmarkerReady;
}

// --- Image loading -----------------------------------------------------------
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (file) loadImageFile(file);
});

function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => onImageReady(img);
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

async function onImageReady(img) {
  state.img = img;
  layoutCanvas();
  $("step-camera").hidden = true;
  $("step-measure").hidden = false;
  $("step-result").hidden = false;
  // Default card handles: a horizontal segment near the bottom third.
  const w = canvas.width, h = canvas.height;
  state.handles.cardA = { x: w * 0.30, y: h * 0.80 };
  state.handles.cardB = { x: w * 0.70, y: h * 0.80 };
  // Default pupils (overwritten by detection if a face is found).
  state.handles.leftPupil = { x: w * 0.42, y: h * 0.40 };
  state.handles.rightPupil = { x: w * 0.58, y: h * 0.40 };
  draw();
  await detectPupils();
  await detectCard();
}

// --- Camera live preview -----------------------------------------------------
let stream = null;
let useFrontCamera = true;

async function openCamera() {
  $("step-input").hidden = true;
  $("step-camera").hidden = false;
  await startStream();
}

async function startStream() {
  stopStream();
  setCamStatus("Starting camera…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: useFrontCamera ? "user" : "environment" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    setCamStatus("");
    // Only offer "switch" when more than one camera is present.
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      $("switchCamBtn").hidden = cams.length < 2;
    } catch { /* device enumeration optional */ }
  } catch (err) {
    console.error(err);
    setCamStatus("Camera unavailable — " + err.message + ". Use Upload photo instead.");
  }
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
}

function captureFrame() {
  if (!stream || !video.videoWidth) {
    setCamStatus("Camera not ready yet.");
    return;
  }
  const off = document.createElement("canvas");
  off.width = video.videoWidth;
  off.height = video.videoHeight;
  const octx = off.getContext("2d");
  // Capture the raw frame so the still exactly matches the live preview.
  // (Mirroring is purely cosmetic and PD is invariant to it.)
  octx.drawImage(video, 0, 0, off.width, off.height);
  const img = new Image();
  img.onload = () => onImageReady(img);
  img.src = off.toDataURL("image/jpeg", 0.95);
  stopStream();
}

function closeCamera() {
  stopStream();
  $("step-camera").hidden = true;
  $("step-input").hidden = false;
}

function setCamStatus(msg) { camStatus.textContent = msg; }

function layoutCanvas() {
  const img = state.img;
  const maxW = Math.min(img.naturalWidth, document.querySelector("main").clientWidth - 4);
  state.scale = maxW / img.naturalWidth;
  canvas.width = Math.round(img.naturalWidth * state.scale);
  canvas.height = Math.round(img.naturalHeight * state.scale);
}

// --- Pupil detection ---------------------------------------------------------
async function detectPupils() {
  try {
    await initLandmarker();
    setStatus("Detecting pupils…");
    const result = faceLandmarker.detect(state.img);
    const faces = result.faceLandmarks;
    if (!faces || faces.length === 0) {
      setStatus("No face detected — drag the cyan markers onto the pupils manually.");
      return;
    }
    const lm = faces[0];
    const toCanvas = (p) => ({
      x: p.x * canvas.width,
      y: p.y * canvas.height,
    });
    state.handles.leftPupil = toCanvas(lm[LEFT_IRIS_CENTER]);
    state.handles.rightPupil = toCanvas(lm[RIGHT_IRIS_CENTER]);
    setStatus("Pupils detected. Adjust any marker if needed.");
    draw();
  } catch (err) {
    console.error(err);
    setStatus("Detection failed — place the cyan markers manually. (" + err.message + ")");
  }
}

// --- Card detection (OpenCV.js, in a Web Worker) -----------------------------
// All OpenCV work — the 10 MB WASM compile and the CV pipeline — runs in a
// worker. The main thread only downscales the photo to a small RGBA frame and
// awaits a message, so the page can never hang (RESULT_CODE_HUNG) regardless of
// device speed.
const CARD_PROC_MAXDIM = 1024; // cap the working resolution sent to the worker

let cvWorker = null;
let cvMsgId = 0;
const cvPending = new Map();

function getCvWorker() {
  if (!cvWorker) {
    cvWorker = new Worker("js/cv-worker.js?v=8");
    cvWorker.onmessage = (e) => {
      const { id, corners, error } = e.data;
      const p = cvPending.get(id);
      if (!p) return;
      cvPending.delete(id);
      if (error) p.reject(new Error(error));
      else p.resolve(corners);
    };
    cvWorker.onerror = (e) => {
      const err = new Error("Card detector worker failed: " + (e.message || "unknown"));
      for (const [id, p] of cvPending) { p.reject(err); cvPending.delete(id); }
    };
  }
  return cvWorker;
}

function detectCardInWorker(imageData) {
  return new Promise((resolve, reject) => {
    const id = ++cvMsgId;
    cvPending.set(id, { resolve, reject });
    // Transfer the pixel buffer to avoid a copy.
    getCvWorker().postMessage({ id, imageData }, [imageData.data.buffer]);
  });
}

// Draw the photo onto a small offscreen canvas and return its RGBA pixels plus
// the scale factor from that frame back to natural-image coordinates.
function downscaledFrame(img, maxDim) {
  const longest = Math.max(img.naturalWidth, img.naturalHeight);
  const procScale = Math.min(1, maxDim / longest);
  const w = Math.max(1, Math.round(img.naturalWidth * procScale));
  const h = Math.max(1, Math.round(img.naturalHeight * procScale));
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d");
  octx.drawImage(img, 0, 0, w, h);
  return { imageData: octx.getImageData(0, 0, w, h), procScale };
}

async function detectCard() {
  try {
    setStatus("Loading card detector (one-time ~10 MB download)…");
    const { imageData, procScale } = downscaledFrame(state.img, CARD_PROC_MAXDIM);
    const corners = await detectCardInWorker(imageData);
    if (!corners) {
      setStatus("Card not auto-detected — place the orange markers on a card edge manually.");
      return;
    }
    // Worker returns corners in the downscaled frame; map → natural → canvas.
    const quad = corners.map((p) => ({ x: p.x / procScale, y: p.y / procScale }));
    const wTop = dist(quad[0], quad[1]);
    const hL = dist(quad[0], quad[3]);
    let a, b;
    if (wTop >= hL) {
      a = quad[0]; b = quad[1]; // top edge is the long edge
    } else {
      a = quad[0]; b = quad[3]; // left edge is the long edge
    }
    const s = state.scale;
    state.handles.cardA = { x: a.x * s, y: a.y * s };
    state.handles.cardB = { x: b.x * s, y: b.y * s };
    // The auto-picked edge is the long edge → match the reference selector.
    cardEdge.value = "85.6";
    customWrap.hidden = true;
    setStatus("Card detected. Verify the orange markers sit on one card edge.");
    draw();
  } catch (err) {
    console.error(err);
    setStatus("Card detection failed — place the orange markers manually. (" + err.message + ")");
  }
}

// --- Drawing -----------------------------------------------------------------
function draw() {
  if (!state.img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);

  const h = state.handles;
  drawLine(h.leftPupil, h.rightPupil, "#2ad1d1");
  drawLine(h.cardA, h.cardB, "#ff8a3d");
  drawHandle(h.leftPupil, "#2ad1d1", "L");
  drawHandle(h.rightPupil, "#2ad1d1", "R");
  drawHandle(h.cardA, "#ff8a3d", "1");
  drawHandle(h.cardB, "#ff8a3d", "2");

  updateResult();
}

function drawLine(a, b, color) {
  if (!a || !b) return;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawHandle(p, color, label) {
  if (!p) return;
  ctx.beginPath();
  ctx.arc(p.x, p.y, HANDLE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(0,0,0,0.6)";
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
  ctx.fillStyle = "#000";
  ctx.fill();
}

// --- Measurement -------------------------------------------------------------
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function getCardMm() {
  if (cardEdge.value === "custom") {
    const v = parseFloat(customMm.value);
    return isFinite(v) && v > 0 ? v : null;
  }
  return parseFloat(cardEdge.value);
}

function updateResult() {
  const h = state.handles;
  if (!h.leftPupil || !h.rightPupil || !h.cardA || !h.cardB) return;
  const cardMm = getCardMm();
  const cardPx = dist(h.cardA, h.cardB);
  const pupilPx = dist(h.leftPupil, h.rightPupil);
  if (!cardMm || cardPx < 1) {
    $("pdValue").textContent = "—";
    return;
  }
  const mmPerPx = cardMm / cardPx;
  const pdMm = pupilPx * mmPerPx;

  $("pdValue").textContent = pdMm.toFixed(1);
  $("breakdownList").innerHTML = [
    `Reference card edge: <strong>${cardMm} mm</strong>`,
    `Card edge in image: <strong>${cardPx.toFixed(1)} px</strong>`,
    `Scale: <strong>${mmPerPx.toFixed(4)} mm/px</strong>`,
    `Pupil distance in image: <strong>${pupilPx.toFixed(1)} px</strong>`,
    `Computed PD: <strong>${pdMm.toFixed(2)} mm</strong>`,
  ].map((s) => `<li>${s}</li>`).join("");
}

// --- Drag interaction --------------------------------------------------------
function canvasPoint(evt) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * sx,
    y: (evt.clientY - rect.top) * sy,
  };
}

function nearestHandle(pt) {
  let best = null, bestD = HIT_RADIUS;
  for (const key of Object.keys(state.handles)) {
    const hp = state.handles[key];
    if (!hp) continue;
    const d = dist(pt, hp);
    if (d < bestD) { bestD = d; best = key; }
  }
  return best;
}

canvas.addEventListener("pointerdown", (e) => {
  const pt = canvasPoint(e);
  const key = nearestHandle(pt);
  if (key) {
    state.dragging = key;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!state.dragging) return;
  const pt = canvasPoint(e);
  pt.x = Math.max(0, Math.min(canvas.width, pt.x));
  pt.y = Math.max(0, Math.min(canvas.height, pt.y));
  state.handles[state.dragging] = pt;
  draw();
});

function endDrag(e) {
  if (state.dragging) {
    state.dragging = null;
    canvas.style.cursor = "grab";
    if (e?.pointerId != null && canvas.hasPointerCapture?.(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  }
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

// --- Controls ----------------------------------------------------------------
cardEdge.addEventListener("change", () => {
  customWrap.hidden = cardEdge.value !== "custom";
  draw();
});
customMm.addEventListener("input", draw);
$("redetectBtn").addEventListener("click", detectPupils);
$("detectCardBtn").addEventListener("click", detectCard);
$("resetBtn").addEventListener("click", () => {
  fileInput.value = "";
  $("step-measure").hidden = true;
  $("step-result").hidden = true;
  $("step-camera").hidden = true;
  $("step-input").hidden = false;
  state.img = null;
});

// Camera controls
$("cameraBtn").addEventListener("click", openCamera);
$("captureBtn").addEventListener("click", captureFrame);
$("cancelCamBtn").addEventListener("click", closeCamera);
$("switchCamBtn").addEventListener("click", async () => {
  useFrontCamera = !useFrontCamera;
  await startStream();
});

function setStatus(msg) {
  statusEl.textContent = msg;
}

// Re-layout on resize so the canvas stays within the viewport.
let resizeRAF = null;
window.addEventListener("resize", () => {
  if (!state.img) return;
  if (resizeRAF) cancelAnimationFrame(resizeRAF);
  resizeRAF = requestAnimationFrame(() => {
    const prev = { w: canvas.width, h: canvas.height };
    layoutCanvas();
    const fx = canvas.width / prev.w;
    const fy = canvas.height / prev.h;
    for (const key of Object.keys(state.handles)) {
      const hp = state.handles[key];
      if (hp) { hp.x *= fx; hp.y *= fy; }
    }
    draw();
  });
});
