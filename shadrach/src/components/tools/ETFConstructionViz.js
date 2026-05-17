"use client";
import { useState, useEffect, useRef, useCallback } from "react";

// ─── Math ─────────────────────────────────────────────────────────────────────

function seededRNG(seed) {
  let s = seed >>> 0;
  return () => { s = Math.imul(1664525, s) + 1013904223; return (s >>> 0) / 4294967296; };
}

function projSimplex(v) {
  const n = v.length;
  const u = [...v].sort((a, b) => b - a);
  const cssv = [];
  let cs = 0;
  for (let i = 0; i < n; i++) { cs += u[i]; cssv.push(cs); }
  let rho = 0;
  for (let i = n - 1; i >= 0; i--) { if (u[i] * (i + 1) > cssv[i] - 1) { rho = i; break; } }
  const theta = (cssv[rho] - 1) / (rho + 1);
  return v.map(x => Math.max(x - theta, 0));
}

function matVec(M, v) { return M.map(row => row.reduce((s, x, j) => s + x * v[j], 0)); }
function vecNorm(v) { return Math.sqrt(v.reduce((s, x) => s + x * x, 0)); }

function buildGraph(m, density, seed) {
  const r = seededRNG(seed);
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = i + 1; j < m; j++)
      if (r() < density) { const w = 0.15 + r() * 0.65; A[i][j] = A[j][i] = w; }
  // Guarantee connectivity via spanning chain
  const perm = [...Array(m).keys()].sort(() => r() - 0.5);
  for (let k = 0; k < m - 1; k++) {
    const i = perm[k], j = perm[k + 1];
    if (!A[i][j]) { const w = 0.2 + r() * 0.3; A[i][j] = A[j][i] = w; }
  }
  const L = A.map((row, i) => {
    const d = row.reduce((s, x) => s + x, 0);
    return row.map((a, j) => i === j ? d : -a);
  });
  const maxDeg = Math.max(...A.map(row => row.reduce((s, x) => s + x, 0)));
  const gamma = 0.9 / (2 * maxDeg + 1e-9);
  return { A, L, gamma };
}

function banachStep(w, L, mu, g) {
  const Lw = matVec(L, w);
  return projSimplex(w.map((wi, i) => wi - g * Lw[i] + g * mu[i]));
}

function computeWstar(L, mu, g, m) {
  let w = new Array(m).fill(1 / m);
  for (let k = 0; k < 3000; k++) w = banachStep(w, L, mu, g);
  return w;
}

// ─── Palette ──────────────────────────────────────────────────────────────────

const PAL = [
  "#6366f1","#f59e0b","#10b981","#ec4899",
  "#3b82f6","#ef4444","#8b5cf6","#14b8a6",
  "#f97316","#84cc16","#06b6d4","#a855f7",
];

// ─── Graph SVG ────────────────────────────────────────────────────────────────

function GraphViz({ A, m, weights }) {
  if (!A || !weights) return null;
  const S = 240, cx = S / 2, cy = S / 2;
  const R = S / 2 - 34;
  const maxA = Math.max(...A.flat(), 0.01);
  const nodes = Array.from({ length: m }, (_, i) => {
    const a = (2 * Math.PI * i / m) - Math.PI / 2;
    return { x: cx + R * Math.cos(a), y: cy + R * Math.sin(a), w: weights[i] };
  });
  return (
    <svg width={S} height={S} className="overflow-visible">
      {A.map((row, i) => row.map((aij, j) => {
        if (j <= i || !aij) return null;
        const op = 0.1 + 0.75 * (aij / maxA);
        return (
          <line key={`${i}-${j}`}
            x1={nodes[i].x} y1={nodes[i].y}
            x2={nodes[j].x} y2={nodes[j].y}
            stroke={`rgba(139,92,246,${op.toFixed(2)})`}
            strokeWidth={0.5 + 2.5 * (aij / maxA)} />
        );
      }))}
      {nodes.map((n, i) => {
        const nr = 9 + 14 * n.w;
        const col = PAL[i % PAL.length];
        return (
          <g key={i}>
            <circle cx={n.x} cy={n.y} r={nr}
              fill={`${col}44`} stroke={col} strokeWidth={1.5} />
            <text x={n.x} y={n.y + 4} textAnchor="middle"
              fontSize={8} fill="white" fontWeight="bold">{i + 1}</text>
            <text x={n.x} y={n.y + nr + 10} textAnchor="middle"
              fontSize={7} fill="rgba(255,255,255,0.5)">
              {(n.w * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Weight Bars ──────────────────────────────────────────────────────────────

function WeightBars({ weights, target, m }) {
  if (!weights) return null;
  const H = 110, pad = 5;
  const bw = (260 - pad * (m + 1)) / m;
  return (
    <svg width={260} height={H + 18}>
      {weights.map((w, i) => {
        const tgt = target?.[i] ?? 0;
        const col = PAL[i % PAL.length];
        const hB = Math.max(w * H * 3.2, 1);
        const hT = tgt * H * 3.2;
        const x = pad + i * (bw + pad);
        return (
          <g key={i}>
            <line x1={x} y1={H - hT} x2={x + bw} y2={H - hT}
              stroke="rgba(255,255,255,0.28)" strokeWidth={1.5} strokeDasharray="3 2" />
            <rect x={x} y={H - hB} width={bw} height={hB}
              fill={`${col}cc`} rx={2} />
            <text x={x + bw / 2} y={H + 13} textAnchor="middle"
              fontSize={7} fill="rgba(255,255,255,0.38)">A{i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Convergence Sparkline ────────────────────────────────────────────────────

function Sparkline({ errors }) {
  if (!errors?.length) return (
    <div className="h-14 flex items-center justify-center text-light/25 text-xs">
      press Run to see convergence
    </div>
  );
  const W = 260, H = 54;
  const le = errors.map(e => Math.log10(Math.max(e, 1e-15)));
  const mn = Math.min(...le) - 0.3, mx = Math.max(...le) + 0.3;
  const n = le.length;
  const pts = le.map((v, i) =>
    `${(i / Math.max(n - 1, 1)) * W},${H - ((v - mn) / (mx - mn)) * H}`
  ).join(" ");
  return (
    <svg width={W} height={H}>
      <polyline points={pts} fill="none"
        stroke="#6366f1" strokeWidth={1.8} strokeLinejoin="round" />
      <text x={2} y={11} fontSize={8} fill="rgba(255,255,255,0.32)">
        log&#x2081;&#x2080; &#x2016;w&#x207F; &#x2212; w*&#x2016;
      </text>
    </svg>
  );
}

// ─── Metric Chip ──────────────────────────────────────────────────────────────

function Chip({ label, value, sub }) {
  return (
    <div className="flex flex-col items-center bg-surface/50 rounded-lg px-3 py-2 min-w-[82px]">
      <span className="text-primary font-mono text-sm font-bold">{value}</span>
      <span className="text-light/55 text-[11px] mt-0.5">{label}</span>
      {sub && <span className="text-light/30 text-[10px]">{sub}</span>}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ETFConstructionViz() {
  const [m, setM] = useState(6);
  const [density, setDensity] = useState(0.55);
  const [running, setRunning] = useState(false);
  const [iter, setIter] = useState(0);
  const [weights, setWeights] = useState(null);
  const [target, setTarget] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const [kappa, setKappa] = useState(null);
  const [convErr, setConvErr] = useState(null);
  const [logErrs, setLogErrs] = useState([]);

  const stRef = useRef(null);
  const timerRef = useRef(null);
  const SEED = 2025;

  const init = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const { A, L, gamma } = buildGraph(m, density, SEED);
    const r = seededRNG(SEED + 99);
    const mu = Array.from({ length: m }, () => 0.01 + r() * 0.07);
    const w_star = computeWstar(L, mu, gamma, m);
    const w0 = projSimplex(Array.from({ length: m }, () => r() - 0.5));
    stRef.current = { L, A, mu, gamma, w: w0, w_star, errors: [], iter: 0 };
    setGraphData({ A, m });
    setTarget(w_star);
    setWeights([...w0]);
    setIter(0); setKappa(null); setConvErr(null); setLogErrs([]);
    setRunning(false);
  }, [m, density]);

  useEffect(() => { init(); }, [init]);

  useEffect(() => {
    if (!running || !stRef.current) return;
    const tick = () => {
      const s = stRef.current;
      for (let i = 0; i < 2; i++) {
        const wn = banachStep(s.w, s.L, s.mu, s.gamma);
        const err = vecNorm(wn.map((wi, j) => wi - s.w_star[j]));
        s.errors.push(err); s.w = wn; s.iter++;
        if (err < 1e-9) break;
      }
      const err = s.errors[s.errors.length - 1];
      let kappaEst = null;
      if (s.errors.length >= 8) {
        const rec = s.errors.slice(-12);
        let sum = 0, cnt = 0;
        for (let i = 1; i < rec.length; i++)
          if (rec[i - 1] > 1e-10) { sum += rec[i] / rec[i - 1]; cnt++; }
        if (cnt > 0) kappaEst = Math.min(Math.max(sum / cnt, 0.001), 0.9999);
      }
      setWeights([...s.w]); setIter(s.iter); setConvErr(err);
      if (kappaEst != null) setKappa(kappaEst);
      setLogErrs(prev => [...prev.slice(-60), err]);
      if (err < 1e-9 || s.iter >= 600) { setRunning(false); return; }
      timerRef.current = setTimeout(tick, 65);
    };
    timerRef.current = setTimeout(tick, 65);
    return () => clearTimeout(timerRef.current);
  }, [running]);

  const gamma = stRef.current?.gamma ?? 0.01;
  const lam2Est = kappa != null ? (1 - kappa) / gamma : null;
  const condEst = lam2Est != null && lam2Est > 0.001 ? 1 / (gamma * lam2Est) : null;
  const speedup = condEst != null
    ? (m * m * condEst * Math.log(1e6) / 8).toExponential(1)
    : "—";

  return (
    <div className="bg-surface/30 rounded-2xl border border-primary/20 p-5">
      {/* Controls */}
      <div className="flex flex-wrap gap-3 items-center mb-5">
        <div className="flex items-center gap-1.5">
          <span className="text-light/50 text-sm">Assets m:</span>
          {[4, 6, 8, 10, 12].map(v => (
            <button key={v} onClick={() => setM(v)}
              className={`px-2 py-0.5 rounded text-xs font-mono transition-colors
                ${m === v
                  ? "bg-primary text-dark font-bold"
                  : "bg-surface/60 text-light/50 hover:text-light"}`}>
              {v}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-light/50 text-sm">Density &#961;:</span>
          <input type="range" min={30} max={85} value={Math.round(density * 100)}
            onChange={e => setDensity(+e.target.value / 100)}
            className="w-20 accent-primary" />
          <span className="text-primary font-mono text-xs">{Math.round(density * 100)}%</span>
        </div>
        <button onClick={init}
          className="px-2.5 py-1 text-xs rounded bg-surface/50 text-light/50 hover:text-light
            border border-light/10 hover:border-light/30 transition-colors">
          Reset
        </button>
        <button onClick={() => setRunning(r => !r)}
          className={`px-4 py-1 text-sm rounded font-medium transition-colors
            ${running
              ? "bg-rose-500/80 text-white hover:bg-rose-500"
              : "bg-primary/90 text-dark hover:bg-primary"}`}>
          {running ? "⏸ Pause" : iter === 0 ? "▶ Run" : "▶ Continue"}
        </button>
        <span className="text-light/35 text-xs font-mono ml-auto">n = {iter}</span>
      </div>

      {/* Visuals */}
      <div className="grid grid-cols-2 gap-5 sm:grid-cols-1">
        <div className="flex flex-col items-center">
          <p className="text-light/40 text-xs mb-1.5">
            Asset correlation graph G &mdash; node size &#8733; w<sub>i</sub><sup>(n)</sup>
          </p>
          <GraphViz A={graphData?.A} m={m} weights={weights} />
        </div>
        <div className="flex flex-col gap-3">
          <div>
            <p className="text-light/40 text-xs mb-1.5">
              Weights w<sup>(n)</sup> &#8594; w* &nbsp;(dashed lines mark fixed point)
            </p>
            <WeightBars weights={weights} target={target} m={m} />
          </div>
          <Sparkline errors={logErrs} />
        </div>
      </div>

      {/* Metrics */}
      <div className="flex flex-wrap gap-2.5 mt-5 justify-center">
        <Chip label="&#954; (measured)"
          value={kappa != null ? kappa.toFixed(4) : "—"} sub="contraction rate" />
        <Chip label="&#955;&#8322; (est.)"
          value={lam2Est != null ? lam2Est.toFixed(3) : "—"} sub="Fiedler value" />
        <Chip label="&#x2016;error&#x2016;"
          value={convErr != null ? convErr.toExponential(1) : "—"} sub={`n = ${iter}`} />
        <Chip label="&#119983;(8,3)"
          value="49,152" sub="pre-comp. states" />
        <Chip label="Speedup S"
          value={speedup !== "—" ? `${speedup}&#215;` : "—"} sub="vs. online Banach" />
      </div>

      <p className="text-center text-light/25 text-[11px] mt-3">
        w* computed via 3,000 Banach iterations. &#954; estimated from consecutive error ratios
        &#x2016;w<sup>n</sup>&#x2212;w*&#x2016;/&#x2016;w<sup>n&#x2212;1</sup>&#x2212;w*&#x2016;.
      </p>
    </div>
  );
}
