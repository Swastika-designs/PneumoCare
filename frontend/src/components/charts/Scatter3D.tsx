import { useMemo, useState } from "react";
import { Legend, seriesColor } from "./index";

interface Point3 {
  pc1: number;
  pc2: number;
  pc3: number;
  group: string;
}

/**
 * Lightweight rotatable 3-D scatter (orthographic projection in SVG). Recharts has no 3-D
 * charts, and a full WebGL library would be overkill for a few hundred PCA scores.
 */
export function Scatter3D({ points, height = 360 }: { points: Point3[]; height?: number }) {
  const [yaw, setYaw] = useState(35);
  const [pitch, setPitch] = useState(20);
  const groups = useMemo(() => Array.from(new Set(points.map((p) => p.group))).slice(0, 8), [points]);
  const projected = useMemo(() => {
    const scale = Math.max(...points.flatMap((p) => [Math.abs(p.pc1), Math.abs(p.pc2), Math.abs(p.pc3)]), 1e-9);
    const a = (yaw * Math.PI) / 180;
    const b = (pitch * Math.PI) / 180;
    return points
      .map((p) => {
        const x = p.pc1 / scale, y = p.pc2 / scale, z = p.pc3 / scale;
        const x1 = x * Math.cos(a) + z * Math.sin(a);
        const z1 = -x * Math.sin(a) + z * Math.cos(a);
        const y1 = y * Math.cos(b) - z1 * Math.sin(b);
        const z2 = y * Math.sin(b) + z1 * Math.cos(b);
        return { sx: x1, sy: y1, depth: z2, group: p.group };
      })
      .sort((p, q) => p.depth - q.depth);
  }, [points, yaw, pitch]);
  const axes = useMemo(() => {
    const a = (yaw * Math.PI) / 180;
    const b = (pitch * Math.PI) / 180;
    const proj = (x: number, y: number, z: number) => {
      const x1 = x * Math.cos(a) + z * Math.sin(a);
      const z1 = -x * Math.sin(a) + z * Math.cos(a);
      return { x: x1, y: y * Math.cos(b) - z1 * Math.sin(b) };
    };
    return [
      { label: "PC1", ...proj(1, 0, 0) },
      { label: "PC2", ...proj(0, 1, 0) },
      { label: "PC3", ...proj(0, 0, 1) },
    ];
  }, [yaw, pitch]);
  const size = 300;
  const half = size / 2;
  const s = half * 0.8;
  return (
    <div>
      <Legend items={groups.map((g, i) => ({ label: g, color: seriesColor(i) }))} />
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", height }} role="img" aria-label="3D PCA scatter plot">
        {axes.map((ax) => (
          <g key={ax.label}>
            <line x1={half} y1={half} x2={half + ax.x * s} y2={half - ax.y * s} stroke="#94a3b8" strokeWidth={1} />
            <text x={half + ax.x * s * 1.08} y={half - ax.y * s * 1.08} fontSize={9} fill="#475569" textAnchor="middle">{ax.label}</text>
          </g>
        ))}
        {projected.map((p, i) => (
          <circle
            key={i}
            cx={half + p.sx * s}
            cy={half - p.sy * s}
            r={2.4}
            fill={seriesColor(Math.max(0, groups.indexOf(p.group)))}
            fillOpacity={0.55 + 0.35 * ((p.depth + 1) / 2)}
            stroke="#fff"
            strokeWidth={0.4}
          />
        ))}
      </svg>
      <div className="mt-2 grid grid-cols-2 gap-4 text-xs text-slate-600">
        <label>Rotate horizontally ({yaw}°)
          <input type="range" min={-180} max={180} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} className="w-full accent-brand-600" />
        </label>
        <label>Tilt ({pitch}°)
          <input type="range" min={-90} max={90} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} className="w-full accent-brand-600" />
        </label>
      </div>
    </div>
  );
}
