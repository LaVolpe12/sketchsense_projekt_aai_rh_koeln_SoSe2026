/**
 * ConfidenceTimeline
 * ------------------
 * Line chart that shows the confidence of the top prediction over time.
 * Each entry is added every time a new prediction arrives.
 *
 * Props:
 *   history: Array<{t: number, label: string, confidence: number}>
 */

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const MAX_POINTS = 30;

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="timeline-tooltip">
      <strong>{d.label}</strong> {(d.confidence * 100).toFixed(1)}%
    </div>
  );
}

export default function ConfidenceTimeline({ history }) {
  const data = history.slice(-MAX_POINTS);

  return (
    <div className="confidence-timeline">
      <h3 className="chart-title">Konfidenz der Vorhersage im Zeitverlauf</h3>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart data={data} margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <XAxis dataKey="t" hide />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} width={36} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.5} stroke="#b8b0e8" strokeDasharray="4 2" />
          <Line
            type="monotone"
            dataKey="confidence"
            stroke="#5036B8"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
