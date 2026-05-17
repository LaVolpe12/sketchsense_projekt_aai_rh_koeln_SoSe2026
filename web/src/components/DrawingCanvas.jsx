/**
 * DrawingCanvas
 * -------------
 * A responsive drawing surface that works with mouse and touch via
 * Pointer Events (one unified API for both input types).
 *
 * Props:
 *   onStroke: () => void   called after each pointer-up (stroke finished)
 *   canvasRef: ref         parent gets direct access to <canvas> element
 *   disabled: bool         when model is still loading
 */

import React, { useEffect, useRef } from "react";

export default function DrawingCanvas({ onStroke, canvasRef, disabled }) {
  const isDrawing = useRef(false);
  const lastPos = useRef(null);

  // Resize canvas to fill its CSS size (device-pixel-ratio aware)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, rect.width, rect.height);
    };

    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [canvasRef]);

  function getPos(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  function getLineWidth() {
    // Target ~2.5px stroke at 28×28 — scale up proportionally to canvas CSS width.
    // Example: 350px canvas → 350/14 = 25px lineWidth → 25*(28/350) = 2px at 28×28.
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect ? Math.max(12, rect.width / 14) : 20;
  }

  function startDraw(e) {
    if (disabled) return;
    e.preventDefault();
    isDrawing.current = true;
    lastPos.current = getPos(e);

    // Draw a filled dot for a single tap/click
    const lw = getLineWidth();
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.arc(lastPos.current.x, lastPos.current.y, lw / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }

  function draw(e) {
    if (!isDrawing.current || disabled) return;
    e.preventDefault();

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const pos = getPos(e);

    ctx.beginPath();
    ctx.moveTo(lastPos.current.x, lastPos.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = getLineWidth();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();

    lastPos.current = pos;
  }

  function endDraw(e) {
    if (!isDrawing.current) return;
    e.preventDefault();
    isDrawing.current = false;
    lastPos.current = null;
    onStroke?.();
  }

  return (
    <canvas
      ref={canvasRef}
      className="drawing-canvas"
      style={{ touchAction: "none", cursor: disabled ? "not-allowed" : "crosshair" }}
      onPointerDown={startDraw}
      onPointerMove={draw}
      onPointerUp={endDraw}
      onPointerLeave={endDraw}
    />
  );
}
