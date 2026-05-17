/**
 * inference.js
 * ------------
 * Loads the TF.js model and class names, then exposes a single `predict`
 * function that accepts an HTMLCanvasElement and returns ranked predictions.
 *
 * Preprocessing (must match training exactly):
 *   1. Resize canvas content to 28×28 px into an offscreen canvas.
 *   2. Read pixel data (RGBA), extract the alpha channel as the stroke mask.
 *   3. Normalise to [0, 1] float32.
 *   4. Reshape to [1, 28, 28, 1].
 */

import * as tf from "@tensorflow/tfjs";

const MODEL_URL = "/model/model.json";
const NAMES_URL = "/model/class_names.json";

let model = null;
let classNames = null;

export async function loadModel() {
  if (model && classNames) return; // already loaded

  [classNames, model] = await Promise.all([
    fetch(NAMES_URL).then((r) => r.json()),
    tf.loadLayersModel(MODEL_URL),
  ]);

  console.log("[SketchSense] model loaded,", classNames.length, "classes");
}

export function getClassNames() {
  return classNames ?? [];
}

/**
 * Converts canvas pixels → 28×28 greyscale tensor.
 *
 * The canvas background is white (#fff) and strokes are black.
 * The training data uses the *opacity* channel (strokes are opaque,
 * background is transparent). We therefore:
 *   - draw the canvas onto a white background
 *   - invert so strokes become bright (high value)
 * to get the same distribution the CNN was trained on.
 */
function canvasToTensor(canvas) {
  const SIZE = 28;
  const off = document.createElement("canvas");
  off.width = SIZE;
  off.height = SIZE;
  const ctx = off.getContext("2d");

  // White background
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(canvas, 0, 0, SIZE, SIZE);

  const { data } = ctx.getImageData(0, 0, SIZE, SIZE); // RGBA, 0–255

  // Build float32 array: invert luminance so strokes → 1.0, bg → 0.0
  const pixels = new Float32Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const luma = (r + g + b) / 3 / 255; // 0 = black stroke, 1 = white bg
    pixels[i] = 1 - luma;               // invert: stroke → 1, bg → 0
  }

  return tf.tensor(pixels, [1, SIZE, SIZE, 1]);
}

/**
 * Returns top-N predictions sorted by confidence (descending).
 * @param {HTMLCanvasElement} canvas
 * @param {number} topN
 * @returns {Array<{label: string, confidence: number}>}
 */
export async function predict(canvas, topN = 5) {
  if (!model) throw new Error("Model not loaded. Call loadModel() first.");

  const tensor = canvasToTensor(canvas);
  const probsTensor = model.predict(tensor);
  const probs = await probsTensor.data();
  tf.dispose([tensor, probsTensor]);

  return classNames
    .map((label, i) => ({ label, confidence: probs[i] }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, topN);
}
