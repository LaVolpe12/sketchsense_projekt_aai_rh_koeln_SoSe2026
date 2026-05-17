"""
train_model.py
--------------
Loads QuickDraw .npy files, trains a CNN, saves the best model.

Memory strategy (important for 8 GB laptops):
  Raw data is kept as uint8 (1 byte/pixel) in RAM.
  Normalisation to float32 happens inside the tf.data pipeline,
  so only one batch is ever in float32 memory at a time.

  RAM usage:  100 classes × 25k × 784 bytes ≈ 2.0 GB  (uint8)
  vs. naive:  100 classes × 25k × 784 × 4  ≈ 7.8 GB  (float32)

Architecture (bigger than the 20-class model):
  Conv(32)→BN→Pool  Conv(64)→BN→Pool  Conv(128)→BN  Conv(256)→BN→GAP
  Dense(512)→Dropout→Dense(N_classes, softmax)
  ~390k parameters — fast in browser, strong enough for 100 classes.

Transfer-learning note:
  To later extend to 345 classes, freeze the Conv blocks and replace
  only the Dense head. The conv layers learn generic sketch features
  that transfer well across QuickDraw categories.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sketchsense.keras")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

SAMPLES_PER_CLASS = 25_000
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.10
BATCH_SIZE        = 128
EPOCHS            = 40       # more epochs for harder task; early stopping handles it
IMG_SIZE          = 28

with open(NAMES_PATH) as f:
    CLASS_NAMES: list[str] = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
print(f"Training on {NUM_CLASSES} classes.")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X (uint8, shape N×28×28×1) and y (int32).
    Keeping X as uint8 saves 4× RAM vs float32.
    Normalisation happens in the tf.data pipeline.
    """
    X_parts, y_parts = [], []

    for idx, name in enumerate(CLASS_NAMES):
        path = os.path.join(DATA_DIR, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run download_data.py first."
            )
        data = np.load(path)[:SAMPLES_PER_CLASS]          # shape (N, 784) uint8
        data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)     # keep as uint8 — no /255 here
        labels = np.full(len(data), idx, dtype="int32")
        X_parts.append(data)
        y_parts.append(labels)
        print(f"  loaded {name:20s}  {len(data):>6,} samples")

    X = np.concatenate(X_parts, axis=0)   # uint8 → ~2.0 GB for 100 classes × 25k
    y = np.concatenate(y_parts, axis=0)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def split_dataset(X, y):
    n      = len(X)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    return (
        (X[n_test+n_val:],      y[n_test+n_val:]),
        (X[n_test:n_test+n_val], y[n_test:n_test+n_val]),
        (X[:n_test],             y[:n_test]),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model() -> keras.Model:
    """
    Deeper CNN than the 20-class model.
    4 conv blocks (3 with pooling, 1 without) give richer features
    for 150 visually diverse classes.

    Spatial flow:  28→14→7→7 (no pool on last two conv)→GlobalAvgPool
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")

    # Block 1 — edge detectors
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)          # 28→14
    x = keras.layers.Dropout(0.2)(x)

    # Block 2 — shapes
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D()(x)          # 14→7
    x = keras.layers.Dropout(0.2)(x)

    # Block 3 — object parts (no pooling, keep 7×7)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    # Block 4 — high-level features (no pooling)
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)   # 7×7×256 → 256

    # Head
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="sketchsense_cnn_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def make_dataset(X: np.ndarray, y: np.ndarray,
                 shuffle: bool = False,
                 augment: bool = False) -> tf.data.Dataset:
    """
    Builds a tf.data pipeline.
    Normalisation (uint8→float32 /255) happens here, so only
    one batch is in float32 memory at a time.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=50_000, seed=42)
    ds = ds.batch(BATCH_SIZE)
    # Normalise inside the pipeline
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if augment:
        aug = keras.Sequential([
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomTranslation(0.05, 0.05),
        ])
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def train() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nLoading dataset …")
    X, y = load_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    ram_gb = X.nbytes / 1e9
    print(f"\nRAM used by dataset: {ram_gb:.2f} GB (uint8)")
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}\n")

    model = build_model()
    model.summary()

    train_ds = make_dataset(X_train, y_train, shuffle=True,  augment=True)
    val_ds   = make_dataset(X_val,   y_val,   shuffle=False, augment=False)
    test_ds  = make_dataset(X_test,  y_test,  shuffle=False, augment=False)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest accuracy: {acc:.4f}  |  Loss: {loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")

    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
    print("Test split saved to ml/models/")


if __name__ == "__main__":
    train()
