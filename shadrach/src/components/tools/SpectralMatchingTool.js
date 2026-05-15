import { useEffect, useRef, useState, useCallback } from 'react';

// ── WebGL2 shaders ────────────────────────────────────────────────────────────

const QUAD_VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Renders a single spectral texture as a coloured image
const SPEC_FRAG = `#version 300 es
precision mediump float;
uniform sampler2D u_tex;
uniform vec3 u_color;
in vec2 v_uv;
out vec4 outColor;
void main() {
  float amp = texture(u_tex, v_uv).r;
  outColor = vec4(u_color * amp * 1.4, amp);
}`;

// Computes pixel-wise interference between two spectra
const INTERF_FRAG = `#version 300 es
precision mediump float;
uniform sampler2D u_texA;
uniform sampler2D u_texB;
in vec2 v_uv;
out vec4 outColor;
void main() {
  float a = texture(u_texA, v_uv).r;
  float b = texture(u_texB, v_uv).r;
  // Constructive: both bright at same (freq, phase)
  float con = a * b * 4.0;
  // Destructive: one bright, other dark
  float desA = a * (1.0 - b) * 2.5;
  float desB = b * (1.0 - a) * 2.5;
  vec3 teal   = vec3(0.17, 0.66, 0.60);
  vec3 coral  = vec3(0.91, 0.45, 0.29);
  vec3 navy   = vec3(0.10, 0.23, 0.36);
  vec3 col = teal * con + coral * desA + navy * desB;
  float brightness = min(1.0, con + desA + desB);
  outColor = vec4(col, brightness);
}`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function mkShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
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
    console.error(gl.getProgramInfoLog(p));
    return null;
  }
  return p;
}

// Full-screen quad geometry (two triangles)
function makeQuad(gl) {
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]),
    gl.STATIC_DRAW);
  return buf;
}

// ── Portfolio generators ──────────────────────────────────────────────────────

const SEEDS = {
  technology: { drift: 0.0008, vols: [0.030, 0.012], freqs: [1/8, 1/21] },
  energy:     { drift: 0.0004, vols: [0.040, 0.018], freqs: [1/16, 1/5]  },
  finance:    { drift: 0.0006, vols: [0.022, 0.009], freqs: [1/12, 1/30] },
  bonds:      { drift: 0.0002, vols: [0.008, 0.004], freqs: [1/32, 1/7]  },
  random:     null,
};

function generateSeries(type, n = 64, seed = 42) {
  let rng = seed;
  const lcg = () => { rng = (rng * 1664525 + 1013904223) & 0xffffffff; return (rng >>> 0) / 4294967296; };

  const g = type === 'random' ? null : SEEDS[type];
  const series = new Array(n);
  let price = 100;

  for (let i = 0; i < n; i++) {
    const noise = (lcg() - 0.5) * (g ? g.vols[0] : 0.035);
    const regime = g ? g.freqs.reduce((s, f, j) => s + Math.sin(2*Math.PI*f*i + j) * g.vols[1], 0) : (lcg() - 0.5) * 0.02;
    price *= (1 + (g ? g.drift : 0.0005) + noise + regime);
    series[i] = price;
  }

  const mean = series.reduce((a, b) => a + b) / n;
  const std  = Math.sqrt(series.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1;
  return series.map(x => (x - mean) / std);
}

// O(N²) DFT — N=64 is fast enough
function dft(signal) {
  const N = signal.length;
  const out = [];
  for (let k = 0; k < N / 2; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(angle);
      im -= signal[n] * Math.sin(angle);
    }
    out.push({ amp: Math.sqrt(re*re + im*im) / N, phase: Math.atan2(im, re) });
  }
  return out;
}

// Build NFREQ×NPHASE spectral image as Float32 amplitude map
// freq → x column, phase → y (Gaussian blob), amplitude → brightness
function buildSpecImage(spectrum, nFreq, nPhase) {
  const img = new Float32Array(nFreq * nPhase);
  const sigma = 0.05; // phase-axis Gaussian width (fraction of full height)
  const maxAmp = Math.max(...spectrum.map(c => c.amp)) || 1;

  for (let fx = 0; fx < nFreq; fx++) {
    const { amp, phase } = spectrum[fx] || { amp: 0, phase: 0 };
    const a = amp / maxAmp;
    const py = ((phase / (Math.PI * 2)) + 0.5 + 1) % 1; // normalise phase to [0,1]

    for (let y = 0; y < nPhase; y++) {
      const fy = y / nPhase;
      let dy = Math.abs(fy - py);
      if (dy > 0.5) dy = 1 - dy; // wrap-around distance
      img[y * nFreq + fx] = a * Math.exp(-(dy * dy) / (2 * sigma * sigma));
    }
  }
  return img;
}

// Spectral cosine similarity (JS-side, precise)
function spectralSimilarity(specA, specB) {
  let dot = 0, na = 0, nb = 0;
  for (let k = 0; k < specA.length; k++) {
    dot += specA[k].amp * specB[k].amp * Math.cos(specA[k].phase - specB[k].phase);
    na  += specA[k].amp ** 2;
    nb  += specB[k].amp ** 2;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

// ── WebGL texture upload ──────────────────────────────────────────────────────

function uploadR32(gl, data, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // R32F requires EXT_color_buffer_float for rendering, but fine for sampling
  const ext = gl.getExtension('OES_texture_float_linear') || gl.getExtension('EXT_color_buffer_float');
  try {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
  } catch {
    // fallback: convert to R8
    const u8 = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i++) u8[i] = Math.round(Math.min(1, data[i]) * 255);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  }
  return tex;
}

// ── Component ─────────────────────────────────────────────────────────────────

const NFREQ  = 32;  // spectral image width (= N/2 for N=64 time points)
const NPHASE = 96;  // spectral image height (phase axis)
const CANVAS_W = 600;
const CANVAS_H = 220;
const PANEL_W  = Math.floor(CANVAS_W / 3); // each third: specA | interf | specB

const PORTFOLIO_OPTIONS = ['technology', 'energy', 'finance', 'bonds', 'random'];

export default function SpectralMatchingTool() {
  const canvasRef = useRef(null);
  const glRef     = useRef(null);
  const progsRef  = useRef({});
  const bufsRef   = useRef({});
  const texRef    = useRef({ a: null, b: null });

  const [portA, setPortA] = useState('technology');
  const [portB, setPortB] = useState('finance');
  const [similarity, setSimilarity] = useState(null);
  const [seedA, setSeedA]   = useState(42);
  const [seedB, setSeedB]   = useState(137);

  const recompute = useCallback((pA, pB, sA, sB) => {
    const gl = glRef.current;
    if (!gl) return;

    const seriesA = generateSeries(pA, 64, sA);
    const seriesB = generateSeries(pB, 64, sB);
    const specA   = dft(seriesA);
    const specB   = dft(seriesB);
    const sim     = spectralSimilarity(specA, specB);
    setSimilarity(sim);

    const imgA = buildSpecImage(specA, NFREQ, NPHASE);
    const imgB = buildSpecImage(specB, NFREQ, NPHASE);

    // Delete old textures
    if (texRef.current.a) gl.deleteTexture(texRef.current.a);
    if (texRef.current.b) gl.deleteTexture(texRef.current.b);
    texRef.current.a = uploadR32(gl, imgA, NFREQ, NPHASE);
    texRef.current.b = uploadR32(gl, imgB, NFREQ, NPHASE);

    renderScene(gl, progsRef.current, bufsRef.current, texRef.current);
  }, []);

  function renderScene(gl, progs, bufs, texs) {
    if (!progs.spec || !progs.interf || !texs.a || !texs.b) return;

    gl.clearColor(0.039, 0.055, 0.090, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const W = CANVAS_W, H = CANVAS_H;

    function drawQuad(prog, tex0, tex1, vpX, vpW, color) {
      gl.useProgram(prog);
      gl.viewport(vpX, 0, vpW, H);
      gl.scissor(vpX, 0, vpW, H);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex0);
      gl.uniform1i(gl.getUniformLocation(prog, 'u_texA'), 0);

      if (tex1) {
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, tex1);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_texB'), 1);
      }
      if (color) {
        gl.uniform3fv(gl.getUniformLocation(prog, 'u_color'), color);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_tex'), 0);
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, bufs.quad);
      const aPos = gl.getAttribLocation(prog, 'a_pos');
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    gl.enable(gl.SCISSOR_TEST);
    // Left panel: spectrum A (teal)
    drawQuad(progs.spec, texs.a, null, 0, PANEL_W, [0.17, 0.66, 0.60]);
    // Middle panel: interference
    drawQuad(progs.interf, texs.a, texs.b, PANEL_W, PANEL_W, null);
    // Right panel: spectrum B (coral)
    drawQuad(progs.spec, texs.b, null, PANEL_W * 2, PANEL_W, [0.91, 0.45, 0.29]);
    gl.disable(gl.SCISSOR_TEST);
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = canvas.getContext('webgl2');
    if (!gl) { console.warn('WebGL2 not supported'); return; }
    glRef.current = gl;

    progsRef.current = {
      spec:   mkProg(gl, QUAD_VERT, SPEC_FRAG),
      interf: mkProg(gl, QUAD_VERT, INTERF_FRAG),
    };
    bufsRef.current = { quad: makeQuad(gl) };

    recompute(portA, portB, seedA, seedB);

    return () => {
      if (texRef.current.a) gl.deleteTexture(texRef.current.a);
      if (texRef.current.b) gl.deleteTexture(texRef.current.b);
      Object.values(progsRef.current).forEach(p => gl.deleteProgram(p));
      gl.deleteBuffer(bufsRef.current.quad);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-render whenever portfolio or seed changes
  useEffect(() => {
    recompute(portA, portB, seedA, seedB);
  }, [portA, portB, seedA, seedB, recompute]);

  const simPct  = similarity != null ? (similarity * 100).toFixed(1) : '—';
  const simColor = similarity == null ? 'text-light/40'
                 : similarity > 0.6   ? 'text-primary'
                 : similarity > 0.2   ? 'text-gold'
                 : 'text-coral';

  return (
    <div className="rounded-2xl border border-primary/20 bg-dark overflow-hidden">
      <div className="px-6 pt-6 pb-4 border-b border-primary/10">
        <h3 className="text-lg font-bold text-primary mb-1">Universal Spectral Matching</h3>
        <p className="text-xs text-light/40 leading-relaxed">
          Portfolios are converted to oscillators via DFT. The spectrum (frequency, phase, amplitude)
          becomes a 2D image. A WebGL2 fragment shader computes pixel-wise interference — no database
          required; comparison IS live shader computation.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 px-6 py-4 border-b border-primary/10 items-end">
        <div>
          <label className="text-xs text-light/50 block mb-1">Portfolio A (teal)</label>
          <select value={portA} onChange={e => setPortA(e.target.value)}
            className="bg-surface border border-primary/20 text-light text-xs rounded px-2 py-1 outline-none">
            {PORTFOLIO_OPTIONS.map(p => <option key={p} value={p}>{p.charAt(0).toUpperCase()+p.slice(1)}</option>)}
          </select>
          {portA === 'random' && (
            <button onClick={() => setSeedA(s => s + 1)}
              className="ml-2 text-xs text-primary/70 hover:text-primary underline">resample</button>
          )}
        </div>

        <div>
          <label className="text-xs text-light/50 block mb-1">Portfolio B (coral)</label>
          <select value={portB} onChange={e => setPortB(e.target.value)}
            className="bg-surface border border-primary/20 text-light text-xs rounded px-2 py-1 outline-none">
            {PORTFOLIO_OPTIONS.map(p => <option key={p} value={p}>{p.charAt(0).toUpperCase()+p.slice(1)}</option>)}
          </select>
          {portB === 'random' && (
            <button onClick={() => setSeedB(s => s + 1)}
              className="ml-2 text-xs text-primary/70 hover:text-primary underline">resample</button>
          )}
        </div>

        <div className="ml-auto text-center">
          <div className={`font-mono text-3xl font-bold ${simColor}`}>{simPct}%</div>
          <div className="text-xs text-light/30">spectral similarity</div>
          <div className="text-xs text-light/20 mt-0.5">
            {similarity != null && (similarity > 0.6 ? 'constructive (correlated)' : similarity > 0.2 ? 'mixed' : 'destructive (independent)')}
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H}
          className="block w-full" style={{ imageRendering: 'auto' }} />
        {/* Column labels */}
        <div className="absolute bottom-0 left-0 right-0 flex pointer-events-none">
          {['Spectrum A', 'Interference', 'Spectrum B'].map((lbl, i) => (
            <div key={lbl} className="flex-1 text-center pb-1.5 text-xs text-light/20">{lbl}</div>
          ))}
        </div>
      </div>

      <p className="px-6 py-3 text-xs text-light/25 leading-relaxed border-t border-primary/10">
        <span className="text-primary/60">Spectral Image Theorem:</span> frequency → x-axis, phase → y-axis (Gaussian blob), amplitude → brightness.
        Teal pixels = both portfolios oscillate at the same frequency and phase (constructive).
        Coral = only A. Navy = only B. Similarity = normalised dot-product of spectral images.
      </p>
    </div>
  );
}
