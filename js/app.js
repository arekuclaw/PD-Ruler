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
}

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
$("resetBtn").addEventListener("click", () => {
  fileInput.value = "";
  $("step-measure").hidden = true;
  $("step-result").hidden = true;
  state.img = null;
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
