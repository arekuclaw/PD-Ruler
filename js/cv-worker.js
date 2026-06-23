// PD Ruler — OpenCV card detection, run off the main thread.
// Keeping the 10 MB WASM compile and the CV pipeline in a worker means the
// page can never hang (RESULT_CODE_HUNG) no matter how slow the device is.

const ID1_RATIO = 85.6 / 53.98; // ≈ 1.586, long/short of an ID-1 card

let cvReady = null;
function loadOpenCV() {
  if (cvReady) return cvReady;
  cvReady = new Promise((resolve, reject) => {
    try {
      self.importScripts("https://docs.opencv.org/4.9.0/opencv.js");
    } catch (e) {
      reject(new Error("Failed to load OpenCV.js (network/CDN)"));
      return;
    }
    const started = Date.now();
    const iv = setInterval(() => {
      if (self.cv && typeof self.cv.Mat === "function") {
        clearInterval(iv);
        resolve(self.cv);
      } else if (Date.now() - started > 90000) {
        clearInterval(iv);
        reject(new Error("OpenCV.js init timed out"));
      }
    }, 50);
  });
  cvReady.catch(() => { cvReady = null; }); // allow retry after failure
  return cvReady;
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// Order 4 points as [top-left, top-right, bottom-right, bottom-left].
function orderCorners(pts) {
  const bySum = [...pts].sort((a, b) => (a.x + a.y) - (b.x + b.y));
  const tl = bySum[0], br = bySum[3];
  const rest = bySum.slice(1, 3).sort((a, b) => (a.x - a.y) - (b.x - b.y));
  const bl = rest[0], tr = rest[1];
  return [tl, tr, br, bl];
}

// Find the best card-like convex quad in an already-downscaled RGBA frame.
// Returns 4 corners in the frame's own coordinates, or null.
function findCardQuad(cv, imageData) {
  const src = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  const blur = new cv.Mat();
  const edges = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  let best = null;
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
    cv.Canny(blur, edges, 50, 150);
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.dilate(edges, edges, kernel);
    kernel.delete();
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const imgArea = src.rows * src.cols;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < imgArea * 0.01) { cnt.delete(); continue; }
      const peri = cv.arcLength(cnt, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.02 * peri, true);
      if (approx.rows === 4 && cv.isContourConvex(approx)) {
        const pts = [];
        for (let r = 0; r < 4; r++) {
          pts.push({ x: approx.data32S[r * 2], y: approx.data32S[r * 2 + 1] });
        }
        const ordered = orderCorners(pts);
        const longSide = (dist(ordered[0], ordered[1]) + dist(ordered[3], ordered[2])) / 2;
        const shortSide = (dist(ordered[0], ordered[3]) + dist(ordered[1], ordered[2])) / 2;
        const ratio = Math.max(longSide, shortSide) / Math.min(longSide, shortSide);
        const ratioErr = Math.abs(ratio - ID1_RATIO);
        if (ratioErr < 0.35) {
          const score = area / (1 + ratioErr * imgArea * 0.0005);
          if (!best || score > best.score) best = { corners: ordered, score };
        }
      }
      approx.delete();
      cnt.delete();
    }
  } finally {
    src.delete(); gray.delete(); blur.delete(); edges.delete();
    contours.delete(); hierarchy.delete();
  }
  return best ? best.corners : null;
}

self.onmessage = async (e) => {
  const { id, imageData } = e.data;
  try {
    const cv = await loadOpenCV();
    const corners = findCardQuad(cv, imageData);
    self.postMessage({ id, corners });
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};
