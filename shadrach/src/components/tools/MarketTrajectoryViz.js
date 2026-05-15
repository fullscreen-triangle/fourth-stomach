import { useEffect, useRef, useState } from 'react';

// ── Shaders ───────────────────────────────────────────────────────────────────

// Trail segments rendered as GL_LINE_STRIP with fading alpha
const TRAIL_VERT = `#version 300 es
in vec2 a_pos;
in float a_alpha;
out float v_alpha;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  v_alpha = a_alpha;
}`;

const TRAIL_FRAG = `#version 300 es
precision mediump float;
in float v_alpha;
out vec4 outColor;
uniform vec3 u_color;
void main() {
  outColor = vec4(u_color, v_alpha * v_alpha);
}`;

// Current position: glowing dot
const DOT_VERT = `#version 300 es
in vec2 a_pos;
in float a_sz;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  gl_PointSize = a_sz;
}`;

const DOT_FRAG = `#version 300 es
precision mediump float;
uniform vec3 u_color;
out vec4 outColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float core = 1.0 - smoothstep(0.05, 0.35, r);
  float halo = (1.0 - smoothstep(0.2, 0.5, r)) * 0.4;
  outColor = vec4(u_color, core + halo);
}`;

function mkShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); return null; }
  return s;
}

function mkProg(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, mkShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mkShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(p)); return null; }
  return p;
}

// ── Ornstein-Uhlenbeck oscillator ─────────────────────────────────────────────
// dx = v dt
// dv = -κ x dt − γ v dt + σ dW
// This is a damped harmonic oscillator + noise — an explicit oscillator in phase space.
// The stationary distribution is Gaussian; the trajectory is ergodic → Poincaré recurrent.

function makeAsset(sigma) {
  return { x: (Math.random() - 0.5) * sigma * 3, v: (Math.random() - 0.5) * sigma * 3 };
}

function stepOU(asset, kappa, gamma, sigma, dt) {
  const noise = () => {
    // Box-Muller
    const u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
  };
  asset.v += (-kappa * asset.x - gamma * asset.v) * dt + sigma * Math.sqrt(dt) * noise();
  asset.x += asset.v * dt;
  // soft boundary (reflect at ±1.5 in normalised coords)
  const bound = 1.5;
  if (Math.abs(asset.x) > bound) { asset.x = Math.sign(asset.x) * bound; asset.v *= -0.6; }
  if (Math.abs(asset.v) > bound) { asset.v = Math.sign(asset.v) * bound; }
}

// ── Asset colors ──────────────────────────────────────────────────────────────

const ASSET_COLORS = [
  [0.17, 0.66, 0.60],  // teal    (primary)
  [0.91, 0.45, 0.29],  // coral
  [0.83, 0.66, 0.26],  // gold
  [0.10, 0.45, 0.72],  // blue
];

const TRAIL_LEN = 220; // history points per asset

// ── Component ─────────────────────────────────────────────────────────────────

export default function MarketTrajectoryViz() {
  const canvasRef = useRef(null);
  const liveRef   = useRef({ kappa: 1.2, sigma: 0.8, gamma: 0.4, n: 2, running: true });
  const stateRef  = useRef({ assets: [], trails: [] });
  const rafRef    = useRef(null);

  const [kappa, setKappa] = useState(1.2);
  const [sigma, setSigma] = useState(0.8);
  const [gamma, setGamma] = useState(0.4);
  const [nAssets, setNAssets] = useState(2);
  const [info, setInfo] = useState({ recurrences: 0, maxRad: 0 });

  // Sync sliders into liveRef and reset assets
  useEffect(() => {
    liveRef.current = { kappa, sigma, gamma, n: nAssets, running: liveRef.current.running };
    const assets = Array.from({ length: nAssets }, () => makeAsset(sigma));
    const trails = assets.map(() => []);
    stateRef.current = { assets, trails };
  }, [kappa, sigma, gamma, nAssets]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: true });
    if (!gl) { console.warn('WebGL2 not available'); return; }

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);

    const trailProg = mkProg(gl, TRAIL_VERT, TRAIL_FRAG);
    const dotProg   = mkProg(gl, DOT_VERT, DOT_FRAG);
    if (!trailProg || !dotProg) return;

    const trailBuf  = gl.createBuffer();
    const alphaBuf  = gl.createBuffer(); // reused every frame, avoids per-frame alloc
    const dotBuf    = gl.createBuffer();

    liveRef.current.running = true;
    const initialSigma = liveRef.current.sigma;
    const initialN     = liveRef.current.n;
    const initAssets   = Array.from({ length: initialN }, () => makeAsset(initialSigma));
    stateRef.current   = { assets: initAssets, trails: initAssets.map(() => []) };

    let prev = 0;
    let recurrences = 0;
    const NORM = 1.8; // normalise OU coords to screen space

    function frame(now) {
      if (!liveRef.current.running) return;
      const dt = Math.min((now - prev) / 1000, 0.04);
      prev = now;

      const { kappa: k, sigma: s, gamma: g } = liveRef.current;
      const { assets, trails } = stateRef.current;

      // Step dynamics
      for (let i = 0; i < assets.length; i++) {
        const prevX = assets[i].x;
        stepOU(assets[i], k, g, s, dt);

        // Poincaré recurrence: count zero-crossings (trajectory returning through origin region)
        if (prevX * assets[i].x < 0 && Math.abs(assets[i].v) < 0.3) recurrences++;

        // Append to ring-buffer trail
        trails[i].push({ x: assets[i].x / NORM, y: assets[i].v / NORM });
        if (trails[i].length > TRAIL_LEN) trails[i].shift();
      }

      const maxRad = Math.max(...assets.map(a => Math.sqrt(a.x*a.x + a.v*a.v)));
      setInfo({ recurrences: recurrences % 10000, maxRad: maxRad.toFixed(2) });

      // Render
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.039, 0.055, 0.090, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Draw trails
      gl.useProgram(trailProg);
      const uTrailColor = gl.getUniformLocation(trailProg, 'u_color');
      const aTrailPos   = gl.getAttribLocation(trailProg, 'a_pos');
      const aTrailAlpha = gl.getAttribLocation(trailProg, 'a_alpha');

      for (let i = 0; i < assets.length; i++) {
        const trail = trails[i];
        if (trail.length < 2) continue;
        const posArr   = new Float32Array(trail.length * 2);
        const alphaArr = new Float32Array(trail.length);
        for (let j = 0; j < trail.length; j++) {
          posArr[j * 2]     = trail[j].x;
          posArr[j * 2 + 1] = trail[j].y;
          alphaArr[j] = j / trail.length; // fade from 0 (oldest) to 1 (newest)
        }
        const col = ASSET_COLORS[i % ASSET_COLORS.length];
        gl.uniform3fv(uTrailColor, col);

        gl.bindBuffer(gl.ARRAY_BUFFER, trailBuf);
        gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(aTrailPos);
        gl.vertexAttribPointer(aTrailPos, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, alphaBuf);
        gl.bufferData(gl.ARRAY_BUFFER, alphaArr, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(aTrailAlpha);
        gl.vertexAttribPointer(aTrailAlpha, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINE_STRIP, 0, trail.length);
      }

      // Draw current positions as glowing dots
      gl.useProgram(dotProg);
      const uDotColor = gl.getUniformLocation(dotProg, 'u_color');
      const aDotPos   = gl.getAttribLocation(dotProg, 'a_pos');
      const aDotSz    = gl.getAttribLocation(dotProg, 'a_sz');

      const dotPos = new Float32Array(assets.length * 2);
      const dotSz  = new Float32Array(assets.length);
      for (let i = 0; i < assets.length; i++) {
        dotPos[i * 2]     = assets[i].x / NORM;
        dotPos[i * 2 + 1] = assets[i].v / NORM;
        dotSz[i] = 14;

        gl.uniform3fv(uDotColor, ASSET_COLORS[i % ASSET_COLORS.length]);
        gl.bindBuffer(gl.ARRAY_BUFFER, dotBuf);
        gl.bufferData(gl.ARRAY_BUFFER, dotPos.slice(i*2, i*2+2), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(aDotPos);
        gl.vertexAttribPointer(aDotPos, 2, gl.FLOAT, false, 0, 0);
        gl.vertexAttrib1f(aDotSz, 14);
        gl.drawArrays(gl.POINTS, 0, 1);
      }

      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      liveRef.current.running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      gl.deleteProgram(trailProg);
      gl.deleteProgram(dotProg);
      gl.deleteBuffer(trailBuf);
      gl.deleteBuffer(alphaBuf);
      gl.deleteBuffer(dotBuf);
    };
  }, []);

  return (
    <div className="rounded-2xl border border-primary/20 bg-dark overflow-hidden">
      <div className="px-6 pt-6 pb-4 border-b border-primary/10">
        <h3 className="text-lg font-bold text-primary mb-1">Market Phase-Space Trajectory</h3>
        <p className="text-xs text-light/40 leading-relaxed">
          Each asset is an Ornstein-Uhlenbeck oscillator in phase space (price x, velocity v).
          The trajectory is ergodic and bounded — it always returns (Poincaré recurrence).
          Computation = traversal of this closed loop.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row">
        <div className="relative flex-1">
          <canvas ref={canvasRef} width={560} height={380}
            className="block w-full" style={{ imageRendering: 'auto' }} />
          {/* Axis labels */}
          <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 pointer-events-none">
            <span className="text-xs text-light/20">x = −price</span>
            <span className="text-xs text-light/20 font-mono">Phase Space (x, v)</span>
            <span className="text-xs text-light/20">x = +price</span>
          </div>
          <div className="absolute top-2 left-2 text-xs text-light/20 rotate-90 origin-left"
            style={{ writingMode: 'vertical-lr' }}>v = velocity</div>
        </div>

        <div className="flex flex-col gap-4 p-5 lg:w-56 shrink-0 border-t lg:border-t-0 lg:border-l border-primary/10">
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-surface rounded-lg p-2">
              <div className="text-primary font-mono text-sm font-bold">{info.recurrences}</div>
              <div className="text-light/30 text-xs mt-0.5">recurrences</div>
            </div>
            <div className="bg-surface rounded-lg p-2">
              <div className="text-primary font-mono text-sm font-bold">{info.maxRad}</div>
              <div className="text-light/30 text-xs mt-0.5">max radius</div>
            </div>
          </div>

          <label className="text-xs text-light/50">
            Assets N = <span className="text-primary font-mono">{nAssets}</span>
            <input type="range" min={1} max={4} step={1} value={nAssets}
              onChange={e => setNAssets(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <label className="text-xs text-light/50">
            Mean reversion κ = <span className="text-primary font-mono">{kappa.toFixed(2)}</span>
            <input type="range" min={0.1} max={3.0} step={0.1} value={kappa}
              onChange={e => setKappa(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <label className="text-xs text-light/50">
            Volatility σ = <span className="text-primary font-mono">{sigma.toFixed(2)}</span>
            <input type="range" min={0.1} max={2.5} step={0.1} value={sigma}
              onChange={e => setSigma(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <label className="text-xs text-light/50">
            Damping γ = <span className="text-primary font-mono">{gamma.toFixed(2)}</span>
            <input type="range" min={0.0} max={2.0} step={0.1} value={gamma}
              onChange={e => setGamma(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <div className="space-y-1 text-xs text-light/25 leading-relaxed">
            {ASSET_COLORS.slice(0, nAssets).map((col, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: `rgb(${col.map(c=>Math.round(c*255)).join(',')})` }} />
                <span>Asset {i + 1}</span>
              </div>
            ))}
          </div>

          <p className="text-xs text-light/25 leading-relaxed">
            <span className="text-primary/60 font-semibold">Poincaré Recurrence</span> — every bounded
            trajectory eventually returns. Markets are gases; trajectories are closed loops.
          </p>
        </div>
      </div>
    </div>
  );
}
