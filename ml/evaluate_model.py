"""
evaluate_model.py
-----------------
Loads the trained model and the saved test split, then prints:
  - Test accuracy
  - Top-3 accuracy
  - classification_report (per-class precision / recall / F1)
  - Confusion matrix saved as ml/models/confusion_matrix.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sketchsense.keras")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
CM_PATH    = os.path.join(MODEL_DIR, "confusion_matrix.png")

with open(NAMES_PATH) as f:
    CLASS_NAMES: list[str] = json.load(f)


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix — SketchSense",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    print(f"Confusion matrix saved to: {CM_PATH}")


def evaluate() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy"))

    print(f"Evaluating on {len(X_test):,} test samples …\n")

    probs = model.predict(X_test, batch_size=256, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    acc    = np.mean(y_pred == y_test)
    top3   = top_k_accuracy(y_test, probs, k=3)

    print(f"\nTest Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Top-3 Accuracy: {top3:.4f}  ({top3*100:.2f}%)")

    print("\n── Classification Report ────────────────────────────────")
    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=3
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)


if __name__ == "__main__":
    evaluate()
