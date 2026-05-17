/**
 * PredictionChart
 * ---------------
 * Horizontal bar chart showing the top-5 class probabilities.
 * Built with Recharts for clean rendering without heavy dependencies.
 *
 * Props:
 *   predictions: Array<{label: string, confidence: number}>
 */

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const COLORS = ["#5036B8", "#736EC6", "#9DD539", "#97D094", "#c5e8a6"];

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function PredictionChart({ predictions }) {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="chart-placeholder">
        <span>Zeichne etwas, um Vorhersagen zu sehen</span>
      </div>
    );
  }

  const data = predictions.map((p) => ({
    name: p.label,
    value: p.confidence,
  }));

  return (
    <div className="prediction-chart">
      <h3 className="chart-title">Top-5 Klassen-Wahrscheinlichkeiten</h3>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 48, bottom: 4, left: 64 }}
        >
          <XAxis type="number" domain={[0, 1]} tickFormatter={pct} tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 13, fontWeight: 600 }} width={60} />
          <Tooltip formatter={(v) => pct(v)} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i] || "#c5e8a6"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="top-guess">
        {predictions[0]?.label && (
          <>
            <span className="guess-label">{predictions[0].label}</span>
            <span className="guess-conf">{pct(predictions[0].confidence)}</span>
          </>
        )}
      </div>
    </div>
  );
}
