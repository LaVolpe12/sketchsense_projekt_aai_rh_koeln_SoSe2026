# SketchSense

Real-time sketch recognition in the browser using QuickDraw data and TensorFlow.js.

Draw on the canvas — the CNN model predicts what you're drawing every 150 ms, directly in the browser with no backend required.

---

## Project structure

```
quickdraw-sketchsense/
  ml/
    data/               ← downloaded .npy files (git-ignored)
    models/             ← trained model + test split (git-ignored)
    class_names.json    ← list of 20 categories
    download_data.py    ← step 1: download QuickDraw .npy files
    train_model.py      ← step 2: train CNN
    evaluate_model.py   ← step 3: evaluate (accuracy, confusion matrix)
    export_tfjs.py      ← step 4: convert to TF.js format
  web/
    public/model/       ← exported TF.js model (model.json + weights)
    src/
      components/       ← DrawingCanvas, PredictionChart, ConfidenceTimeline, HintBox
      ml/inference.js   ← browser-side model loading & prediction
      App.jsx           ← root component
      styles.css
    package.json
    vite.config.js
  requirements.txt
  README.md
```

---

## Setup

### 1 — Python environment

```bash
cd quickdraw-sketchsense

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> Tested with Python 3.10+. TensorFlow 2.13+ is required.

---

### 2 — Download training data

```bash
cd ml
python download_data.py
```

Downloads 20 `.npy` files (~50–80 MB each) from Google Cloud Storage into `ml/data/`.  
**Expected output:** each file is printed with its size.  
**Typical duration:** 5–15 minutes depending on your connection.

---

### 3 — Train the model

```bash
python train_model.py
```

Trains a small CNN on 10,000 drawings per class (200,000 total).

| Parameter | Value |
|-----------|-------|
| Input     | 28 × 28 greyscale |
| Classes   | 20 |
| Epochs    | up to 30 (early stopping) |
| Batch     | 128 |

**Expected output:** model saved to `ml/models/sketchsense.keras`, test split saved as `X_test.npy` / `y_test.npy`.  
**Typical duration:** 15–45 min on CPU, 5–10 min on GPU.  
**Expected accuracy:** ≥ 85% test accuracy.

> To improve accuracy: increase `SAMPLES_PER_CLASS` in `train_model.py` (max 100,000).

---

### 4 — Evaluate the model

```bash
python evaluate_model.py
```

Prints test accuracy, top-3 accuracy, and a per-class `classification_report`.  
Saves a confusion matrix image to `ml/models/confusion_matrix.png`.

---

### 5 — Export to TensorFlow.js

```bash
python export_tfjs.py
```

Converts `sketchsense.keras` → TF.js graph model and copies the files to `web/public/model/`:

```
web/public/model/
  model.json
  group1-shard1of1.bin
  class_names.json
```

---

### 6 — Run the web app

```bash
cd ../web
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

#### Test on your smartphone (same Wi-Fi)

Vite binds to `0.0.0.0` automatically (see `vite.config.js`).  
Find your computer's local IP:

```bash
# macOS / Linux
ifconfig | grep "inet " | grep -v 127

# Windows
ipconfig
```

Then open `http://<your-ip>:5173` on your phone.

---

## How the browser inference works

```
User draws on <canvas>
        ↓
Offscreen 28×28 canvas (downscale)
        ↓
Read pixel RGB → compute luminance → invert (stroke=1, bg=0)
        ↓
Float32Array [1, 28, 28, 1]  ←— same shape as training
        ↓
tf.loadGraphModel  →  model.predict()
        ↓
Softmax probabilities [20]
        ↓
Top-5 sorted results → React state → chart update
```

The 150 ms interval is driven by `setInterval` in `App.jsx`.

---

## Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: Missing cat.npy` | Data not downloaded | Run `download_data.py` |
| `Failed to load model` in browser | Export not run | Run `export_tfjs.py` |
| `CORS error` in browser | Opening `index.html` directly | Use `npm run dev`, not file:// |
| Very low accuracy (<60%) | Too few samples | Increase `SAMPLES_PER_CLASS` |
| Touch drawing not working on phone | Browser zoom | Disable zoom in browser settings |

---

## Dependencies

**Python (ml/)**
- TensorFlow / Keras — model training
- TensorFlow.js converter — model export
- scikit-learn — evaluation metrics
- matplotlib — confusion matrix plot
- NumPy — array operations

**JavaScript (web/)**
- React 18 + Vite — UI framework
- @tensorflow/tfjs — in-browser inference
- Recharts — bar chart and line chart
