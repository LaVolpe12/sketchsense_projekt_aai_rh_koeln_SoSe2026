"""
export_tfjs.py
--------------
Exports the trained Keras 3 model to TensorFlow.js format.

Approach:
  1. Load trained model (Keras 3), extract weights via get_weights().
  2. Rebuild identical architecture with tf_keras (Keras 2 compat layer).
  3. Transfer weights, verify outputs are identical.
  4. Use tensorflowjs.converters.keras_h5_conversion.save_keras_model()
     directly (Python API, no CLI needed).
  5. Copy class_names.json to web/public/model/.
"""

import json
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import keras           # Keras 3 — for loading the trained .keras file
import tf_keras        # Keras 2 compat — used for TF.js export

BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "ml", "models", "sketchsense.keras")
NAMES_SRC   = os.path.join(BASE_DIR, "ml", "class_names.json")
OUT_DIR     = os.path.join(BASE_DIR, "web", "public", "model")
IMG_SIZE    = 28

with open(NAMES_SRC) as f:
    NUM_CLASSES = len(json.load(f))


def build_tf_keras_model() -> tf_keras.Model:
    """Mirror of train_model.py's build_model() — must match exactly."""
    inputs = tf_keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Block 1
    x = tf_keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Dropout(0.2)(x)

    # Block 2
    x = tf_keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.MaxPooling2D()(x)
    x = tf_keras.layers.Dropout(0.2)(x)

    # Block 3 (no pooling)
    x = tf_keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.Dropout(0.2)(x)

    # Block 4 (no pooling)
    x = tf_keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf_keras.layers.BatchNormalization()(x)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)

    # Head
    x = tf_keras.layers.Dense(512, activation="relu")(x)
    x = tf_keras.layers.Dropout(0.4)(x)
    outputs = tf_keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return tf_keras.Model(inputs, outputs)


def export() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train_model.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load Keras 3 model, extract weights ────────────────────────────────
    print("Loading Keras 3 model …")
    trained = keras.models.load_model(MODEL_PATH)
    weights = trained.get_weights()
    print(f"  {len(weights)} weight tensors, {NUM_CLASSES} classes")

    dummy = np.random.rand(2, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    ref_preds = trained.predict(dummy, verbose=0)

    # ── 2. Rebuild with tf_keras, transfer weights ────────────────────────────
    print("Building tf_keras model and transferring weights …")
    export_model = build_tf_keras_model()
    export_model(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32"))
    export_model.set_weights(weights)

    export_preds = export_model.predict(dummy, verbose=0)
    diff = float(np.max(np.abs(ref_preds - export_preds)))
    assert diff < 1e-5, f"Weight mismatch: max_diff={diff}"
    print(f"  Weight transfer verified (max diff={diff:.2e})")

    # ── 3. Export to TF.js format ─────────────────────────────────────────────
    print("Exporting to TensorFlow.js …")
    from tensorflowjs.converters.keras_h5_conversion import save_keras_model
    save_keras_model(export_model, OUT_DIR)

    # ── 4. Copy class names ───────────────────────────────────────────────────
    shutil.copy(NAMES_SRC, os.path.join(OUT_DIR, "class_names.json"))

    print(f"\nExport complete → {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith("."):
            continue
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:40s}  {size/1024:.1f} KB")


if __name__ == "__main__":
    export()
