# PD Ruler ⚡

A browser-based web app that estimates **pupillary distance (PD)** — the distance
between the centers of the pupils in millimeters — from a single photo of a face.

It uses a **driver license** (or any ISO/IEC 7810 **ID-1** card, the same size as a
credit card) held in the frame as a real-world scale reference, then converts pixels
to millimeters.

## How it works

1. **Pupil detection** — [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe)
   locates the iris centers (landmarks 468 and 473) and measures the pixel distance
   between them.
2. **Scale reference** — you mark one edge of the card. An ID-1 card is a fixed size:
   - Long edge: **85.60 mm**
   - Short edge: **53.98 mm**
3. **Conversion** — the app computes:

   ```
   mm_per_pixel = card_mm / card_pixels
   PD_mm        = pupil_pixels × mm_per_pixel
   ```

All markers are draggable, so you can fine-tune the detection before reading the result.

## Privacy

Everything runs **client-side in your browser**. The photo is never uploaded to any
server — the face model and WASM runtime are loaded from a CDN, but your image stays
on your device.

## Accuracy & limitations

The scale is only correct when the reference card is in the **same plane as the eyes**:

- Hold the card flat against the forehead or cheek, **parallel to the camera sensor**.
- Keep the card at the **same distance** from the camera as the eyes (perspective
  makes closer objects look larger).
- Face the camera straight-on; head tilt or rotation introduces error.
- Use a sharp, well-lit, reasonably high-resolution photo.

This tool is a convenience estimate. **For prescription eyewear, confirm your PD with
an optician.**

## Running it

It's a static site — no build step.

```bash
# any static file server, e.g.
python3 -m http.server 8000
# then open http://localhost:8000
```

Or deploy the repo to **GitHub Pages** (Settings → Pages → deploy from `main` / root).

## Tech

- Vanilla JS (ES modules), HTML, CSS — no framework, no bundler
- [@mediapipe/tasks-vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) for face/iris landmarks
- HTML Canvas for overlay and interactive markers
