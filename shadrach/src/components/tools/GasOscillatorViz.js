import { useEffect, useRef, useState } from 'react';

const VERT = `#version 300 es
in vec2 a_pos;
in float a_sz;
in vec4 a_col;
out vec4 v_col;
void main() {
  gl_Position = vec4(a_pos, 0.0, 1.0);
  gl_PointSize = a_sz;
  v_col = a_col;
}`;

const FRAG = `#version 300 es
precision mediump float;
in vec4 v_col;
out vec4 outColor;
void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  float r = length(c);
  if (r > 0.5) discard;
  float core = 1.0 - smoothstep(0.1, 0.4, r);
  float halo = (1.0 - smoothstep(0.3, 0.5, r)) * 0.35;
  outColor = vec4(v_col.rgb, v_col.a * (core + halo));
}`;

function mkShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader error:', gl.getShaderInfoLog(s));
    return null;
  }
  return s;
}

function mkProg(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, mkShader(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mkShader(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error('Program error:', gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

function spawnParticles(n, T) {
  return Array.from({ length: n }, () => ({
    sk: Math.random(),
    st: Math.random(),
    se: Math.random(),
    omega: (0.4 + Math.random() * 1.6) * T,
    phi: Math.random() * Math.PI * 2,
    amp: 0.03 + Math.random() * 0.07,
    sx: 0,
    sy: 0,
  }));
}

// Teal → gold → coral gradient by kinetic entropy value t ∈ [0,1]
function tempColor(t) {
  const c = Math.min(1, t);
  const r = c < 0.5 ? 0.17 + c * 1.66 : 0.83 + (c - 0.5) * 0.34;
  const g = c < 0.5 ? 0.66 - c * 0.32 : 0.50 - (c - 0.5) * 0.40;
  const b = c < 0.5 ? 0.60 - c * 1.20 : 0.0;
  return [Math.min(1, r), Math.max(0, g), Math.max(0, b)];
}

export default function GasOscillatorViz() {
  const canvasRef = useRef(null);
  const liveRef = useRef({ n: 120, T: 1.0, kappa: 0.4, particles: [], running: true });
  const rafRef = useRef(null);

  const [n, setN] = useState(120);
  const [T, setT] = useState(1.0);
  const [kappa, setKappa] = useState(0.4);
  const [stats, setStats] = useState({ z: '—', avgT: '—', avgS: '—', avgP: '—' });

  // Sync slider state into liveRef for the RAF loop to read without stale closures
  useEffect(() => {
    liveRef.current.n = n;
    liveRef.current.T = T;
    liveRef.current.kappa = kappa;
    liveRef.current.particles = spawnParticles(n, T);
  }, [n, T, kappa]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
    if (!gl) { console.warn('WebGL2 not supported'); return; }

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive for glow

    const prog = mkProg(gl, VERT, FRAG);
    if (!prog) return;

    const bufPos = gl.createBuffer();
    const bufSz  = gl.createBuffer();
    const bufCol = gl.createBuffer();

    const aPos = gl.getAttribLocation(prog, 'a_pos');
    const aSz  = gl.getAttribLocation(prog, 'a_sz');
    const aCol = gl.getAttribLocation(prog, 'a_col');

    liveRef.current.particles = spawnParticles(liveRef.current.n, liveRef.current.T);
    liveRef.current.running = true;

    let prev = 0;

    function frame(now) {
      if (!liveRef.current.running) return;
      const dt = Math.min((now - prev) / 1000, 0.05);
      prev = now;

      const { particles: ps, T: temp, kappa: coup } = liveRef.current;
      if (!ps.length) { rafRef.current = requestAnimationFrame(frame); return; }

      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        p.phi += p.omega * dt;
        // Kuramoto nearest-neighbour coupling
        if (coup > 0 && i > 0) {
          p.phi += coup * Math.sin(ps[i - 1].phi - p.phi) * dt;
        }
        // Project 3-D S-entropy coords onto 2-D screen (xy = sk,st; se → size)
        p.sx = (p.sk * 1.8 - 0.9) + Math.sin(p.phi) * p.amp;
        p.sy = (p.st * 1.8 - 0.9) + Math.cos(p.phi * 0.618) * p.amp * 0.85;
        // Soft boundary reflection
        if (p.sx >  0.95) { p.sx =  0.95; p.sk = (p.sx + 0.9) / 1.8; }
        if (p.sx < -0.95) { p.sx = -0.95; p.sk = (p.sx + 0.9) / 1.8; }
        if (p.sy >  0.95) { p.sy =  0.95; p.st = (p.sy + 0.9) / 1.8; }
        if (p.sy < -0.95) { p.sy = -0.95; p.st = (p.sy + 0.9) / 1.8; }
      }

      const posArr = new Float32Array(ps.length * 2);
      const szArr  = new Float32Array(ps.length);
      const colArr = new Float32Array(ps.length * 4);
      let sumZ = 0, sumT = 0, sumS = 0, sumP = 0;

      for (let i = 0; i < ps.length; i++) {
        const p = ps[i];
        posArr[i * 2]     = p.sx;
        posArr[i * 2 + 1] = p.sy;
        szArr[i] = 4 + p.se * 10;
        const [r, g, b] = tempColor(p.sk * temp);
        colArr[i * 4]     = r;
        colArr[i * 4 + 1] = g;
        colArr[i * 4 + 2] = b;
        colArr[i * 4 + 3] = 0.80;
        sumZ += Math.exp(-p.omega / Math.max(0.01, temp));
        sumT += p.sk * temp;
        sumS += p.st;
        sumP += p.se;
      }

      const inv = 1 / ps.length;
      setStats({
        z:    sumZ.toFixed(2),
        avgT: (sumT * inv).toFixed(3),
        avgS: (sumS * inv).toFixed(3),
        avgP: (sumP * inv).toFixed(3),
      });

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.039, 0.055, 0.090, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);

      gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
      gl.bufferData(gl.ARRAY_BUFFER, posArr, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, bufSz);
      gl.bufferData(gl.ARRAY_BUFFER, szArr, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aSz);
      gl.vertexAttribPointer(aSz, 1, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, bufCol);
      gl.bufferData(gl.ARRAY_BUFFER, colArr, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aCol);
      gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, ps.length);
      rafRef.current = requestAnimationFrame(frame);
    }

    rafRef.current = requestAnimationFrame(frame);

    return () => {
      liveRef.current.running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      gl.deleteProgram(prog);
      gl.deleteBuffer(bufPos);
      gl.deleteBuffer(bufSz);
      gl.deleteBuffer(bufCol);
    };
  }, []); // mount once; live state communicated via liveRef

  return (
    <div className="rounded-2xl border border-primary/20 bg-dark overflow-hidden">
      <div className="px-6 pt-6 pb-4 border-b border-primary/10">
        <h3 className="text-lg font-bold text-primary mb-1">Live Gas Oscillator Field</h3>
        <p className="text-xs text-light/40 leading-relaxed">
          Each particle is an oscillator in S-entropy space [0,1]³ — kinetic Sk (position x), temporal St (position y),
          energetic Se (size). Colour: teal = cold, gold = warm, coral = hot.
          Kuramoto coupling drives phase synchronisation.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row">
        <canvas
          ref={canvasRef}
          width={580}
          height={380}
          className="block shrink-0 w-full lg:w-auto"
          style={{ imageRendering: 'pixelated' }}
        />

        <div className="flex flex-col gap-4 p-5 lg:w-56 shrink-0 border-t lg:border-t-0 lg:border-l border-primary/10">
          <div className="grid grid-cols-2 gap-2 text-center">
            {[
              { label: 'Z (partition fn)', val: stats.z },
              { label: '⟨T⟩ temperature',  val: stats.avgT },
              { label: '⟨S⟩ complexity',   val: stats.avgS },
              { label: '⟨P⟩ density',      val: stats.avgP },
            ].map(({ label, val }) => (
              <div key={label} className="bg-surface rounded-lg p-2">
                <div className="text-primary font-mono text-sm font-bold">{val}</div>
                <div className="text-light/30 text-xs mt-0.5 leading-tight">{label}</div>
              </div>
            ))}
          </div>

          <label className="text-xs text-light/50">
            Molecules N = <span className="text-primary font-mono">{n}</span>
            <input type="range" min={30} max={280} value={n}
              onChange={e => setN(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <label className="text-xs text-light/50">
            Temperature T = <span className="text-primary font-mono">{T.toFixed(2)}</span>
            <input type="range" min={0.1} max={3.0} step={0.05} value={T}
              onChange={e => setT(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <label className="text-xs text-light/50">
            Coupling κ = <span className="text-primary font-mono">{kappa.toFixed(2)}</span>
            <input type="range" min={0} max={2.0} step={0.05} value={kappa}
              onChange={e => setKappa(+e.target.value)}
              className="w-full mt-1.5 accent-primary" />
          </label>

          <p className="text-xs text-light/25 leading-relaxed">
            <span className="text-primary/60 font-semibold">Processor-Oscillator Duality</span> — every oscillator
            IS a processor. T = processing rate, S = categorical complexity, P = computational density.
            Interference = computation.
          </p>
        </div>
      </div>
    </div>
  );
}
