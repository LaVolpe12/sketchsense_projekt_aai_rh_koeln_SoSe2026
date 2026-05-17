/**
 * App.jsx
 * -------
 * Root component. Wires together:
 *   - Model loading (once, on mount)
 *   - 150 ms prediction interval while the user is drawing
 *   - DrawingCanvas, PredictionChart, ConfidenceTimeline, HintBox
 */

import React, { useRef, useState, useEffect, useCallback } from "react";
import DrawingCanvas from "./components/DrawingCanvas.jsx";
import PredictionChart from "./components/PredictionChart.jsx";
import ConfidenceTimeline from "./components/ConfidenceTimeline.jsx";
import HintBox from "./components/HintBox.jsx";
import { loadModel, predict, getClassNames } from "./ml/inference.js";

const PREDICT_INTERVAL_MS = 150;

export default function App() {
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const tickRef = useRef(0);

  const [modelReady, setModelReady] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [history, setHistory] = useState([]);
  const [hasStrokes, setHasStrokes] = useState(false);
  const [classNames, setClassNames] = useState([]);

  // Load model on mount
  useEffect(() => {
    loadModel()
      .then(() => {
        setModelReady(true);
        setClassNames(getClassNames());
      })
      .catch((err) => {
        console.error(err);
        setLoadError(err.message);
      });
  }, []);

  // Run prediction
  const runPredict = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas || !modelReady || !hasStrokes) return;

    try {
      const preds = await predict(canvas, 5);
      setPredictions(preds);
      setHistory((prev) => [
        ...prev,
        { t: tickRef.current++, label: preds[0]?.label, confidence: preds[0]?.confidence ?? 0 },
      ]);
    } catch (e) {
      console.warn("Prediction error:", e);
    }
  }, [modelReady, hasStrokes]);

  // Start/stop interval when hasStrokes changes
  useEffect(() => {
    if (hasStrokes && modelReady) {
      intervalRef.current = setInterval(runPredict, PREDICT_INTERVAL_MS);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [hasStrokes, modelReady, runPredict]);

  function handleStroke() {
    setHasStrokes(true);
  }

  function handleClear() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, rect.width, rect.height);
    setPredictions([]);
    setHistory([]);
    setHasStrokes(false);
    tickRef.current = 0;
  }

  const topLabel = predictions[0]?.label ?? null;
  const topConf  = predictions[0]?.confidence ?? 0;

  return (
    <div className="app">
      {/* Top institution bar */}
      <div className="institution-bar">
        <span className="institution-text">
          Rheinische Hochschule Köln &nbsp;·&nbsp; Master of Engineering – Technical Management &nbsp;·&nbsp; Applied Artificial Intelligence
        </span>
      </div>

      <header className="app-header">
        <h1>SketchSense</h1>
        <div className="app-subtitle-row">
          <span className="subtitle">Echtzeit-Skizzenerkennung mit Convolutional Neural Networks</span>
          {modelReady && (
            <span className="model-badge">100 Classes · CNN · 82.3% Val-Accuracy</span>
          )}
        </div>
      </header>

      {/* Banners */}
      {loadError && <div className="error-banner">Modell konnte nicht geladen werden: {loadError}</div>}
      {!modelReady && !loadError && (
        <div className="loading-banner">Neuronales Netzwerk wird geladen…</div>
      )}

      <main className="main-grid">
        <section className="canvas-section">
          <DrawingCanvas canvasRef={canvasRef} onStroke={handleStroke} disabled={!modelReady} />
          <div className="canvas-controls">
            <button className="btn-clear" onClick={handleClear} disabled={!hasStrokes}>
              Löschen
            </button>
            {modelReady && (
              <>
                <span className="status-dot ready" />
                <span className="status-label">Modell bereit</span>
              </>
            )}
          </div>
        </section>

        <section className="results-section">
          <PredictionChart predictions={predictions} />
          <HintBox topLabel={topLabel} confidence={topConf} />
          <ConfidenceTimeline history={history} />
        </section>
      </main>

      <footer className="app-footer">
        <div className="footer-credits">
          <span>Nico Drescher &amp; Adriano Volpe</span>
          <span className="footer-sep">·</span>
          <span>RH Köln 2026</span>
        </div>
        <div className="footer-categories">
          <span className="footer-label">
            {classNames.length > 0 ? `${classNames.length} Kategorien:` : "Kategorien:"}
          </span>{" "}
          {classNames.length > 0 ? classNames.join(" · ") : "wird geladen…"}
        </div>
      </footer>
    </div>
  );
}
