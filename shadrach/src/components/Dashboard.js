import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';

// ── Constants ─────────────────────────────────────────────────────────────────
const S_FLOOR = 2;
const NFREQ   = 32;
const NPHASE  = 96;
const M       = { t: 14, r: 20, b: 32, l: 60 };
const COL     = {
  primary: '#2ca89a',
  coral:   '#e8734a',
  gold:    '#d4a843',
  navy:    '#1a3a5c',
};

// ── Seeded RNG ────────────────────────────────────────────────────────────────
function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

function gaussFromRng(rng) {
  const u = Math.max(1e-9, 1 - rng());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

// ── Synthetic data fallback ───────────────────────────────────────────────────
function syntheticData(n = 400, seed = 42) {
  const rng  = lcg(seed);
  const rows = [];
  let price  = 100 + rng() * 50;
  let d      = new Date(2022, 0, 3);
  while (rows.length < n) {
    if (d.getDay() !== 0 && d.getDay() !== 6) {
      const ret  = 0.0004 + 0.014 * gaussFromRng(rng);
      const open = price;
      price = Math.max(1, price * (1 + ret));
      rows.push({
        date:     new Date(d),
        open,
        close:    price,
        high:     Math.max(open, price) * (1 + rng() * 0.005),
        low:      Math.min(open, price) * (1 - rng() * 0.005),
        volume:   Math.floor((rng() * 5 + 0.5) * 1e6),
        pct:      ret * 100,
        gainLoss: price - open,
      });
    }
    d = new Date(d); d.setDate(d.getDate() + 1);
  }
  return rows;
}

// ── Alpha Vantage parser ──────────────────────────────────────────────────────
function parseAV(json) {
  const ts = json['Time Series (Daily)'];
  if (!ts) return null;
  const rows = Object.entries(ts)
    .map(([ds, v]) => {
      const open  = +v['1. open'];
      const close = +v['4. close'];
      return {
        date:     new Date(ds),
        open,
        close,
        high:     +v['2. high'],
        low:      +v['3. low'],
        volume:   +v['6. volume'],
        pct:      ((close - open) / open) * 100,
        gainLoss: close - open,
      };
    })
    .filter(r => !isNaN(r.close))
    .sort((a, b) => a.date - b.date);
  return rows.length >= 10 ? rows : null;
}

// ── Framework computations ────────────────────────────────────────────────────
function computeFramework(rows, winLen = 20) {
  const n          = rows.length;
  const clock      = new Float64Array(n);
  const mdCalendar = new Float64Array(n);
  const mdMonetary = new Float64Array(n);
  const entropy    = new Float64Array(n).fill(51);

  // Transaction clock Θ(t) = Σ |gainLoss|
  let cum = 0;
  for (let i = 0; i < n; i++) {
    cum += Math.abs(rows[i].gainLoss);
    clock[i]      = cum;
    mdCalendar[i] = rows[i].pct;
  }

  // dP/dΘ = gainLoss / ḡ  where ḡ = rolling mean of |gainLoss|
  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - winLen + 1);
    let gbar = 0;
    for (let j = start; j <= i; j++) gbar += Math.abs(rows[j].gainLoss);
    gbar /= (i - start + 1);
    mdMonetary[i] = gbar > 1e-10 ? rows[i].gainLoss / gbar : 0;
  }

  // S-entropy: rolling coefficient-of-variation → [S_FLOOR, 100]
  for (let i = winLen; i < n; i++) {
    const win  = rows.slice(i - winLen, i);
    const mean = win.reduce((s, r) => s + r.pct, 0) / winLen;
    const std  = Math.sqrt(win.reduce((s, r) => s + (r.pct - mean) ** 2, 0) / winLen) || 1;
    const cv   = std / (Math.abs(mean) + 1e-6);
    const raw  = Math.min(1, cv / 10);
    entropy[i] = S_FLOOR + (100 - S_FLOOR) * raw;
  }

  return { clock, mdCalendar, mdMonetary, entropy };
}

// ── DFT (O(N²), N ≤ 64) ─────────────────────────────────────────────────────
function dft(signal) {
  const N = signal.length;
  const out = [];
  for (let k = 0; k < N / 2; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const a = (2 * Math.PI * k * n) / N;
      re += signal[n] * Math.cos(a);
      im -= signal[n] * Math.sin(a);
    }
    out.push({ amp: Math.sqrt(re * re + im * im) / N, phase: Math.atan2(im, re) });
  }
  return out;
}

function buildSpecImage(spectrum) {
  const img   = new Float32Array(NFREQ * NPHASE);
  const maxA  = Math.max(...spectrum.map(c => c.amp)) || 1;
  const sigma = 0.05;
  for (let fx = 0; fx < NFREQ; fx++) {
    const { amp, phase } = spectrum[fx] || { amp: 0, phase: 0 };
    const a  = amp / maxA;
    const py = ((phase / (Math.PI * 2)) + 1.5) % 1;
    for (let y = 0; y < NPHASE; y++) {
      let dy = Math.abs(y / NPHASE - py);
      if (dy > 0.5) dy = 1 - dy;
      img[y * NFREQ + fx] = a * Math.exp(-(dy * dy) / (2 * sigma * sigma));
    }
  }
  return img;
}

function spectralSimilarity(specA, specB) {
  let dot = 0, na = 0, nb = 0;
  for (let k = 0; k < specA.length; k++) {
    dot += specA[k].amp * specB[k].amp * Math.cos(specA[k].phase - specB[k].phase);
    na  += specA[k].amp ** 2;
    nb  += specB[k].amp ** 2;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

// ── WebGL2 shaders ────────────────────────────────────────────────────────────
const QUAD_VERT = `#version 300 es
in vec2 a_pos; out vec2 v_uv;
void main() { v_uv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const SPEC_FRAG = `#version 300 es
precision mediump float;
uniform sampler2D u_texA; uniform vec3 u_color;
in vec2 v_uv; out vec4 outColor;
void main() {
  float amp = texture(u_texA, v_uv).r;
  outColor = vec4(u_color * amp * 1.4, amp);
}`;

const INTERF_FRAG = `#version 300 es
precision mediump float;
uniform sampler2D u_texA; uniform sampler2D u_texB;
in vec2 v_uv; out vec4 outColor;
void main() {
  float a    = texture(u_texA, v_uv).r;
  float b    = texture(u_texB, v_uv).r;
  float con  = a * b * 4.0;
  float desA = a * (1.0 - b) * 2.5;
  float desB = b * (1.0 - a) * 2.5;
  vec3 teal  = vec3(0.17, 0.66, 0.60);
  vec3 coral = vec3(0.91, 0.45, 0.29);
  vec3 navy  = vec3(0.10, 0.23, 0.36);
  outColor   = vec4(teal * con + coral * desA + navy * desB, min(1.0, con + desA + desB));
}`;

function mkShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(s)); return null; }
  return s;
}

function mkProg(gl, fsSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, mkShader(gl, gl.VERTEX_SHADER, QUAD_VERT));
  gl.attachShader(p, mkShader(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(p)); return null; }
  return p;
}

function uploadTex(gl, data, w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  try {
    gl.getExtension('OES_texture_float_linear');
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
  } catch {
    const u8 = Uint8Array.from(data, x => Math.round(Math.min(1, Math.max(0, x)) * 255));
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
  }
  return tex;
}

// ── D3 axis style ─────────────────────────────────────────────────────────────
const darkAxis = g => g
  .call(g => g.select('.domain').attr('stroke', '#2a2a3a'))
  .call(g => g.selectAll('.tick line').attr('stroke', '#2a2a3a'))
  .call(g => g.selectAll('text').attr('fill', '#666').attr('font-size', '9px'));

// ══════════════════════════════════════════════════════════════════════════════
export default function Dashboard() {
  const [symbol,   setSymbol]   = useState('SPY');
  const [apiKey,   setApiKey]   = useState('');
  const [data,     setData]     = useState(() => syntheticData());
  const [status,   setStatus]   = useState('synthetic');
  const [simScore, setSimScore] = useState(null);
  const [containerW, setContainerW] = useState(0);

  // DOM refs
  const wrapRef     = useRef(null);
  const priceRef    = useRef(null);
  const navRef      = useRef(null);
  const clockRef    = useRef(null);
  const monetaryRef = useRef(null);
  const entropyRef  = useRef(null);
  const glCanvasRef = useRef(null);

  // WebGL state refs
  const glRef    = useRef(null);
  const progsRef = useRef(null);
  const bufRef   = useRef(null);

  // Load API key from localStorage once on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('av_key');
      if (saved) setApiKey(saved);
    }
  }, []);

  // Persist API key
  useEffect(() => {
    if (typeof window !== 'undefined' && apiKey) localStorage.setItem('av_key', apiKey);
  }, [apiKey]);

  // Fetch live data
  useEffect(() => {
    if (!apiKey.trim()) { setData(syntheticData()); setStatus('synthetic'); return; }
    setStatus('loading');
    const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=${encodeURIComponent(symbol)}&outputsize=full&apikey=${encodeURIComponent(apiKey)}`;
    fetch(url)
      .then(r => r.json())
      .then(json => {
        const parsed = parseAV(json);
        if (parsed) { setData(parsed); setStatus('live'); }
        else         { setData(syntheticData()); setStatus('error'); }
      })
      .catch(() => { setData(syntheticData()); setStatus('error'); });
  }, [symbol, apiKey]);

  // Container resize observer
  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(entries => {
      const w = entries[0].contentRect.width;
      if (w > 0) setContainerW(w);
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  // WebGL2 init (once)
  useEffect(() => {
    const canvas = glCanvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl2');
    if (!gl) return;
    glRef.current = gl;
    progsRef.current = { spec: mkProg(gl, SPEC_FRAG), interf: mkProg(gl, INTERF_FRAG) };
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    bufRef.current = buf;
    return () => { glRef.current = null; };
  }, []);

  // ── WebGL render ────────────────────────────────────────────────────────────
  const renderWebGL = useCallback((filtered) => {
    const gl    = glRef.current;
    const progs = progsRef.current;
    const buf   = bufRef.current;
    if (!gl || !progs || !buf || filtered.length < 8) return;

    // Use hardware performance.now() timestamps to build a physical clock
    const t0 = performance.now();

    const n      = Math.min(filtered.length, 64);
    const slice  = filtered.slice(-n);
    const prices = slice.map(r => r.close);
    const pmean  = prices.reduce((a, b) => a + b) / n;
    const pstd   = Math.sqrt(prices.reduce((a, b) => a + (b - pmean) ** 2, 0) / n) || 1;
    const normP  = prices.map(x => (x - pmean) / pstd);

    // Theoretical: Brownian motion with matched volatility
    // Seed with hardware clock delta — this IS a physical oscillation reading
    const hwDelta = performance.now() - t0;
    const rng   = lcg(Math.round(hwDelta * 1000) ^ 0xdeadbeef);
    const sigma = Math.sqrt(normP.map((x, i) => i > 0 ? (x - normP[i-1]) ** 2 : 0).reduce((a, b) => a + b) / Math.max(1, n - 1)) || 0.01;
    let bm = 0;
    const model = new Array(n).fill(0).map(() => { bm += sigma * gaussFromRng(rng); return bm; });
    const mmean = model.reduce((a, b) => a + b) / n;
    const mstd  = Math.sqrt(model.reduce((a, b) => a + (b - mmean) ** 2, 0) / n) || 1;
    const normM = model.map(x => (x - mmean) / mstd);

    const specP  = dft(normP);
    const specM  = dft(normM);
    const sim    = spectralSimilarity(specP, specM);
    setSimScore(sim);

    const imgP = buildSpecImage(specP);
    const imgM = buildSpecImage(specM);

    const texA = uploadTex(gl, imgP, NFREQ, NPHASE);
    const texB = uploadTex(gl, imgM, NFREQ, NPHASE);

    const W  = glCanvasRef.current.width;
    const H  = glCanvasRef.current.height;
    const PW = Math.floor(W / 3);

    gl.enable(gl.SCISSOR_TEST);
    gl.clearColor(0.039, 0.055, 0.09, 1);
    gl.scissor(0, 0, W, H); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    function drawPanel(prog, tA, tB, vpX, vpW, color) {
      gl.useProgram(prog);
      gl.viewport(vpX, 0, vpW, H); gl.scissor(vpX, 0, vpW, H);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      const loc = gl.getAttribLocation(prog, 'a_pos');
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tA);
      gl.uniform1i(gl.getUniformLocation(prog, 'u_texA'), 0);
      if (tB) {
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, tB);
        gl.uniform1i(gl.getUniformLocation(prog, 'u_texB'), 1);
      }
      if (color) gl.uniform3fv(gl.getUniformLocation(prog, 'u_color'), color);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // Left: market spectrum (teal), Centre: interference, Right: model spectrum (coral/pink)
    drawPanel(progs.spec,  texA, null, 0,    PW, [0.17, 0.66, 0.60]);
    drawPanel(progs.interf, texA, texB, PW,  PW, null);
    drawPanel(progs.spec,  texB, null, PW*2, PW, [0.91, 0.45, 0.29]);

    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
  }, []);

  // ── D3 chart renderers ────────────────────────────────────────────────────────

  function drawPrice(filtered, W, IW) {
    const PH  = 140;
    const svg = d3.select(priceRef.current).attr('width', W).attr('height', PH + M.t + M.b);
    svg.selectAll('*').remove();

    svg.append('defs').append('linearGradient').attr('id', 'db-pg')
      .attr('x1','0%').attr('y1','0%').attr('x2','0%').attr('y2','100%')
      .selectAll('stop')
      .data([{off:'0%',a:0.28},{off:'100%',a:0}])
      .join('stop').attr('offset',d=>d.off).attr('stop-color',COL.primary).attr('stop-opacity',d=>d.a);

    const g  = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const xF = d3.scaleTime().domain(d3.extent(filtered, d => d.date)).range([0, IW]);
    const yP = d3.scaleLinear()
      .domain([d3.min(filtered, d => d.low)*0.97, d3.max(filtered, d => d.high)*1.03])
      .range([PH, 0]).nice();

    g.insert('g',':first-child')
      .call(d3.axisLeft(yP).ticks(4).tickSize(-IW).tickFormat(''))
      .call(h => h.select('.domain').remove())
      .call(h => h.selectAll('.tick line').attr('stroke','#1a1a2a').attr('stroke-dasharray','3,3'));

    g.append('path').datum(filtered)
      .attr('fill','url(#db-pg)')
      .attr('d', d3.area().x(d=>xF(d.date)).y0(PH).y1(d=>yP(d.close)).curve(d3.curveMonotoneX));
    g.append('path').datum(filtered)
      .attr('fill','none').attr('stroke',COL.primary).attr('stroke-width',1.5)
      .attr('d', d3.line().x(d=>xF(d.date)).y(d=>yP(d.close)).curve(d3.curveMonotoneX));

    g.append('g').attr('transform',`translate(0,${PH})`).call(d3.axisBottom(xF).ticks(5).tickFormat(d3.timeFormat('%b %y'))).call(darkAxis);
    g.append('g').call(d3.axisLeft(yP).ticks(4).tickFormat(d=>`$${d.toFixed(0)}`)).call(darkAxis);
    g.append('text').attr('x',4).attr('y',11).attr('fill','#444').attr('font-size','9px').attr('font-family','monospace').text('CLOSE PRICE');
  }

  function drawClock(filtered, W, IW) {
    const { clock } = computeFramework(filtered);
    const CH  = 80;
    const svg = d3.select(clockRef.current).attr('width', W).attr('height', CH + M.t + M.b);
    svg.selectAll('*').remove();
    const g  = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const xF = d3.scaleTime().domain(d3.extent(filtered, d => d.date)).range([0, IW]);
    const yC = d3.scaleLinear().domain([0, d3.max(clock)]).range([CH, 0]).nice();

    // Gradient fill under clock
    svg.append('defs').append('linearGradient').attr('id','db-cg')
      .attr('x1','0%').attr('y1','0%').attr('x2','0%').attr('y2','100%')
      .selectAll('stop').data([{off:'0%',a:0.22},{off:'100%',a:0}])
      .join('stop').attr('offset',d=>d.off).attr('stop-color',COL.gold).attr('stop-opacity',d=>d.a);

    g.append('path').datum(filtered.map((d,i) => [d.date, clock[i]]))
      .attr('fill','url(#db-cg)')
      .attr('d', d3.area().x(d=>xF(d[0])).y0(CH).y1(d=>yC(d[1])));
    g.append('path').datum(filtered.map((d,i) => [d.date, clock[i]]))
      .attr('fill','none').attr('stroke',COL.gold).attr('stroke-width',1.5)
      .attr('d', d3.line().x(d=>xF(d[0])).y(d=>yC(d[1])));

    g.append('g').attr('transform',`translate(0,${CH})`).call(d3.axisBottom(xF).ticks(5).tickFormat(d3.timeFormat('%b %y'))).call(darkAxis);
    g.append('g').call(d3.axisLeft(yC).ticks(3)).call(darkAxis);
    g.append('text').attr('x',4).attr('y',11).attr('fill','#444').attr('font-size','9px').attr('font-family','monospace').text('Θ(t) TRANSACTION CLOCK');
  }

  function drawMonetary(filtered, W, IW) {
    const { mdCalendar, mdMonetary } = computeFramework(filtered);
    const MH  = 90;
    const svg = d3.select(monetaryRef.current).attr('width', W).attr('height', MH + M.t + M.b);
    svg.selectAll('*').remove();
    const g  = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const xF = d3.scaleTime().domain(d3.extent(filtered, d => d.date)).range([0, IW]);

    const calValid = mdCalendar.filter(isFinite);
    const monValid = mdMonetary.filter(isFinite);
    const lo = Math.min(d3.min(calValid), d3.min(monValid));
    const hi = Math.max(d3.max(calValid), d3.max(monValid));
    const yM = d3.scaleLinear().domain([lo, hi]).range([MH, 0]).nice();
    const z0 = yM(0);

    // dP/dt — calendar returns (muted)
    g.append('path').datum(filtered.map((d,i) => [d.date, mdCalendar[i]]))
      .attr('fill','none').attr('stroke','#555').attr('stroke-width',0.8).attr('opacity',0.55)
      .attr('d', d3.line().x(d=>xF(d[0])).y(d=>yM(d[1])).defined(d=>isFinite(d[1])));

    // dP/dΘ — monetary derivative (primary)
    g.append('path').datum(filtered.map((d,i) => [d.date, mdMonetary[i]]))
      .attr('fill','none').attr('stroke',COL.primary).attr('stroke-width',1.5)
      .attr('d', d3.line().x(d=>xF(d[0])).y(d=>yM(d[1])).defined(d=>isFinite(d[1])));

    g.append('line').attr('x1',0).attr('x2',IW).attr('y1',z0).attr('y2',z0)
      .attr('stroke','#3a3a4a').attr('stroke-width',1);

    g.append('g').attr('transform',`translate(0,${MH})`).call(d3.axisBottom(xF).ticks(5).tickFormat(d3.timeFormat('%b %y'))).call(darkAxis);
    g.append('g').call(d3.axisLeft(yM).ticks(4).tickFormat(d=>d.toFixed(1))).call(darkAxis);
    g.append('text').attr('x',4).attr('y',11).attr('fill','#444').attr('font-size','9px').attr('font-family','monospace').text('dP/dt (gray)   dP/dΘ (teal)');
  }

  function drawEntropy(filtered, W, IW) {
    const { entropy } = computeFramework(filtered);
    const EH  = 60;
    const svg = d3.select(entropyRef.current).attr('width', W).attr('height', EH + M.t + M.b);
    svg.selectAll('*').remove();
    const g  = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const xF = d3.scaleTime().domain(d3.extent(filtered, d => d.date)).range([0, IW]);
    const yE = d3.scaleLinear().domain([0, 100]).range([EH, 0]);

    g.append('path').datum(filtered.map((d,i) => [d.date, entropy[i]]))
      .attr('fill', COL.coral).attr('fill-opacity', 0.15)
      .attr('stroke', COL.coral).attr('stroke-width', 1)
      .attr('d', d3.area().x(d=>xF(d[0])).y0(EH).y1(d=>yE(d[1])).curve(d3.curveMonotoneX));

    // S_floor reference line
    g.append('line').attr('x1',0).attr('x2',IW).attr('y1',yE(S_FLOOR)).attr('y2',yE(S_FLOOR))
      .attr('stroke',COL.primary).attr('stroke-width',0.8).attr('stroke-dasharray','4,3');
    g.append('text').attr('x',IW-2).attr('y',yE(S_FLOOR)-3).attr('text-anchor','end')
      .attr('fill',COL.primary).attr('font-size','8px').attr('font-family','monospace').text('Sₕ');

    g.append('g').attr('transform',`translate(0,${EH})`).call(d3.axisBottom(xF).ticks(5).tickFormat(d3.timeFormat('%b %y'))).call(darkAxis);
    g.append('g').call(d3.axisLeft(yE).ticks(3)).call(darkAxis);
    g.append('text').attr('x',4).attr('y',11).attr('fill','#444').attr('font-size','9px').attr('font-family','monospace').text('S-ENTROPY [0–100]');
  }

  // ── Main D3 effect ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerW || !data.length) return;
    const W  = containerW;
    const IW = W - M.l - M.r;

    const xFull = d3.scaleTime().domain(d3.extent(data, d => d.date)).range([0, IW]);

    // Navigator SVG
    const NH   = 48;
    const svgN = d3.select(navRef.current).attr('width', W).attr('height', NH + M.t + 24);
    svgN.selectAll('*').remove();
    const gN = svgN.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const yN = d3.scaleLinear().domain(d3.extent(data, d => d.close)).range([NH, 0]);

    gN.append('path').datum(data)
      .attr('fill','none').attr('stroke',COL.primary).attr('stroke-width',0.8).attr('opacity',0.3)
      .attr('d', d3.line().x(d=>xFull(d.date)).y(d=>yN(d.close)).curve(d3.curveMonotoneX));

    gN.append('g').attr('transform',`translate(0,${NH})`)
      .call(d3.axisBottom(xFull).ticks(4).tickFormat(d3.timeFormat('%Y'))).call(darkAxis);
    gN.append('text').attr('x',4).attr('y',11)
      .attr('fill','#333').attr('font-size','9px').attr('font-family','monospace').text('DRAG TO SELECT RANGE');

    function redraw(filtered) {
      if (!filtered.length) return;
      drawPrice(filtered, W, IW);
      drawClock(filtered, W, IW);
      drawMonetary(filtered, W, IW);
      drawEntropy(filtered, W, IW);
      renderWebGL(filtered);
    }

    const brush = d3.brushX()
      .extent([[0, 0], [IW, NH]])
      .on('brush end', ({ selection }) => {
        const filtered = selection
          ? data.filter(d => {
              const [d0, d1] = selection.map(xFull.invert);
              return d.date >= d0 && d.date <= d1;
            })
          : data;
        if (filtered.length >= 4) redraw(filtered);
      });

    const bG = gN.append('g').call(brush);
    bG.select('.selection').attr('fill',COL.primary).attr('fill-opacity',0.12).attr('stroke',COL.primary).attr('stroke-width',1);
    bG.select('.overlay').attr('fill','transparent');

    redraw(data);
  }, [data, containerW, renderWebGL]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Derived S-entropy display score ─────────────────────────────────────────
  const sEntropy = simScore !== null
    ? S_FLOOR + (100 - S_FLOOR) * (0.5 - simScore / 2)
    : null;

  const statusStyle = {
    live:      'border-primary/30 text-primary',
    loading:   'border-gold/30 text-gold/60 animate-pulse',
    error:     'border-coral/30 text-coral/60',
    synthetic: 'border-light/10 text-light/25',
  }[status] || 'border-light/10 text-light/25';

  const statusLabel = {
    live:      `LIVE · ${symbol}`,
    loading:   'LOADING…',
    error:     'API ERROR · SYNTHETIC',
    synthetic: 'SYNTHETIC DATA',
  }[status];

  return (
    <div ref={wrapRef} className="rounded-2xl border border-primary/10 bg-dark overflow-hidden">

      {/* ── Header / Controls ─────────────────────────────────────────────── */}
      <div className="px-5 py-3 border-b border-primary/10 flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-light/30">SYMBOL</span>
          <input
            className="bg-surface/30 border border-primary/20 rounded px-2 py-1 text-xs font-mono text-light w-20
                       focus:outline-none focus:border-primary/50 transition-colors"
            value={symbol}
            onChange={e => setSymbol(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && e.target.blur()}
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-light/30">AV KEY</span>
          <input
            type="password"
            className="bg-surface/30 border border-primary/20 rounded px-2 py-1 text-xs font-mono text-light w-40
                       focus:outline-none focus:border-primary/50 transition-colors"
            placeholder="alphavantage.co/support/#api-key"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
          />
        </div>

        <div className="ml-auto flex items-center gap-4">
          {sEntropy !== null && (
            <div className="text-xs font-mono">
              <span className="text-light/25">S-ENTROPY&nbsp;</span>
              <span className={sEntropy < 35 ? 'text-primary font-bold' : sEntropy < 65 ? 'text-gold' : 'text-coral'}>
                {sEntropy.toFixed(1)}
              </span>
              <span className="text-light/20">/100</span>
            </div>
          )}
          <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${statusStyle}`}>
            {statusLabel}
          </span>
        </div>
      </div>

      {/* ── Charts ────────────────────────────────────────────────────────── */}
      <div className="px-3 pt-4 pb-6 space-y-1">

        {/* Price */}
        <svg ref={priceRef} className="block w-full" />

        {/* Navigator brush */}
        <svg ref={navRef} className="block w-full" />

        {/* Section label */}
        <div className="pt-4 pb-1 px-1 flex items-center gap-3">
          <span className="text-xs font-mono text-light/20 tracking-widest uppercase">Transaction Clock Framework</span>
          <div className="flex-1 border-t border-primary/5" />
        </div>

        {/* Clock */}
        <svg ref={clockRef} className="block w-full" />

        {/* Monetary derivative */}
        <svg ref={monetaryRef} className="block w-full" />

        {/* Spectral interference */}
        <div className="pt-4 pb-1 px-1 flex items-center justify-between">
          <span className="text-xs font-mono text-light/20 tracking-widest uppercase">Spectral Interference</span>
          <div className="flex gap-3 text-xs font-mono text-light/20">
            <span><span className="inline-block w-2 h-2 rounded-full bg-primary/60 mr-1" />market</span>
            <span><span className="inline-block w-2 h-2 rounded-full bg-white/10 mr-1" />market⊗model</span>
            <span><span className="inline-block w-2 h-2 rounded-full bg-coral/60 mr-1" />theoretical BM</span>
          </div>
        </div>
        <canvas
          ref={glCanvasRef}
          width={900}
          height={160}
          className="block w-full rounded-lg"
        />
        <p className="text-xs font-mono text-light/15 px-1 pt-1">
          Teal = constructive (model aligns with market) &nbsp;&middot;&nbsp; Coral = destructive (structure the conventional view misses)
        </p>

        {/* S-entropy */}
        <div className="pt-3 pb-1 px-1">
          <span className="text-xs font-mono text-light/20 tracking-widest uppercase">S-Entropy Score</span>
        </div>
        <svg ref={entropyRef} className="block w-full" />
      </div>
    </div>
  );
}
