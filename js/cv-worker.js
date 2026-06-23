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

// Find the best card-like rectangle in an already-downscaled RGBA frame.
// Returns 4 corners in the frame's own coordinates, or null.
//
// Real ID cards on a face rarely yield a clean closed outer contour: the
// border is interrupted by hair/skin of similar contrast and the surface
// carries lots of internal detail, so the cleanest contour is usually a
// *fragment* spanning the card. We therefore match on the contour's
// minAreaRect — its aspect ratio and size — rather than requiring the contour
// to fill that rectangle. A tight aspect tolerance (ID-1 ≈ 1.586) keeps this
// from firing on arbitrary rectangles; when nothing matches we return null and
// the caller falls back to manual markers. Verified on real photos: a card
// held flat on the forehead is found, while a finger-occluded card is not.
const AR_TOLERANCE = 0.15;   // accepted aspect-ratio deviation from ID-1
const MIN_RECT_FRAC = 0.02;  // card's bounding box must be ≥2% of the frame
const MAX_RECT_FRAC = 0.6;   // …and not a near-full-frame rectangle

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
    cv.Canny(blur, edges, 30, 90);
    // Close small gaps so card-border fragments join up.
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
    cv.morphologyEx(edges, edges, cv.MORPH_CLOSE, kernel);
    kernel.delete();
    cv.findContours(edges, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const imgArea = src.rows * src.cols;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const rect = cv.minAreaRect(cnt);
      const rw = rect.size.width, rh = rect.size.height;
      if (rw < 1 || rh < 1) { cnt.delete(); continue; }
      const rectFrac = (rw * rh) / imgArea;
      const ratio = Math.max(rw, rh) / Math.min(rw, rh);
      const ratioErr = Math.abs(ratio - ID1_RATIO);
      if (rectFrac > MIN_RECT_FRAC && rectFrac < MAX_RECT_FRAC && ratioErr < AR_TOLERANCE) {
        // Favour larger, better-proportioned rectangles.
        const score = rectFrac / (1 + 5 * ratioErr);
        if (!best || score > best.score) {
          const pts = cv.RotatedRect.points(rect).map((p) => ({ x: p.x, y: p.y }));
          best = { corners: orderCorners(pts), score };
        }
      }
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
