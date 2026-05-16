import { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';

// ── Seeded RNG ────────────────────────────────────────────────────────────────

function lcg(seed) {
  let s = (seed >>> 0) || 1;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

// ── Synthetic OHLCV data (weekdays only, geometric Brownian motion) ────────────

function generateData(seed = 42, n = 400) {
  const rng  = lcg(seed);
  const gauss = () => {
    const u = Math.max(1e-9, 1 - rng());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  };

  const rows = [];
  let price  = 100 + rng() * 50;
  let d      = new Date(2021, 0, 4); // 2021-01-04

  while (rows.length < n) {
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) {
      const ret = 0.0004 + (0.010 + rng() * 0.012) * gauss();
      const open = price;
      price = Math.max(1, price * (1 + ret));
      rows.push({
        date:   new Date(d),
        open,
        close:  price,
        high:   Math.max(open, price) * (1 + rng() * 0.006),
        low:    Math.min(open, price) * (1 - rng() * 0.006),
        volume: Math.floor((rng() * 4.5 + 0.5) * 1e6),
        pct:    ret * 100,
      });
    }
    d = new Date(d); d.setDate(d.getDate() + 1);
  }
  return rows;
}

// Monthly aggregation
function toMonthly(rows) {
  return [...d3.rollup(
    rows,
    v => ({ date: d3.timeMonth.floor(v[0].date), vol: d3.sum(v, r => r.volume), ret: d3.mean(v, r => r.pct) }),
    r => +d3.timeMonth.floor(r.date)
  ).values()].sort((a, b) => a.date - b.date);
}

// ── Layout ────────────────────────────────────────────────────────────────────

const M    = { t: 14, r: 20, b: 32, l: 56 };
const PH   = 160;  // price chart inner height
const NH   = 52;   // navigator inner height
const BH   = 104;  // bar chart inner height

// ── D3 axis helpers ───────────────────────────────────────────────────────────

const darkAxis = g => g
  .call(g => g.select('.domain').attr('stroke', '#2a2a3a'))
  .call(g => g.selectAll('.tick line').attr('stroke', '#2a2a3a'))
  .call(g => g.selectAll('text').attr('fill', '#777').attr('font-size', '9px'));

// ── Component ─────────────────────────────────────────────────────────────────

export default function StockDashboard({
  name         = 'MARKET',
  primaryColor = '#2ca89a',
  negColor     = '#e8734a',
  seed         = 42,
}) {
  const wrapRef  = useRef(null);
  const priceRef = useRef(null);
  const navRef   = useRef(null);
  const volRef   = useRef(null);
  const retRef   = useRef(null);

  const [info, setInfo] = useState({ shown: 0, total: 0, rows: [] });
  const data = useMemo(() => generateData(seed), [seed]);

  useEffect(() => {
    if (!wrapRef.current) return;

    const W  = wrapRef.current.clientWidth || 800;
    const IW = W - M.l - M.r;

    // ── Shared x scale for navigator ─────────────────────────────────────────
    const xFull = d3.scaleTime()
      .domain(d3.extent(data, d => d.date))
      .range([0, IW]);

    // ── Price chart setup ─────────────────────────────────────────────────────
    const svgP = d3.select(priceRef.current).attr('width', W).attr('height', PH + M.t + M.b);
    svgP.selectAll('*').remove();

    const defsP = svgP.append('defs');
    defsP.append('linearGradient').attr('id', `area-grad-${seed}`)
      .attr('x1', '0%').attr('y1', '0%').attr('x2', '0%').attr('y2', '100%')
      .selectAll('stop').data([
        { off: '0%',   c: primaryColor, a: 0.30 },
        { off: '100%', c: primaryColor, a: 0.00 },
      ]).join('stop')
      .attr('offset', d => d.off)
      .attr('stop-color', d => d.c)
      .attr('stop-opacity', d => d.a);

    defsP.append('clipPath').attr('id', `clip-p-${seed}`)
      .append('rect').attr('width', IW).attr('height', PH);

    const gP   = svgP.append('g').attr('transform', `translate(${M.l},${M.t})`);
    const clipG = gP.append('g').attr('clip-path', `url(#clip-p-${seed})`);

    const areaPath = clipG.append('path').attr('fill', `url(#area-grad-${seed})`);
    const linePath = clipG.append('path')
      .attr('fill', 'none').attr('stroke', primaryColor).attr('stroke-width', 1.5);

    const gPxAxis = gP.append('g').attr('transform', `translate(0,${PH})`);
    const gPyAxis = gP.append('g');

    // Grid
    const gPgrid = gP.insert('g', ':first-child').attr('class', 'grid');

    gP.append('text').attr('x', 4).attr('y', 11)
      .attr('fill', '#555').attr('font-size', '9px').attr('font-family', 'monospace')
      .text('CLOSE PRICE');

    // ── Navigator setup ───────────────────────────────────────────────────────
    const svgN = d3.select(navRef.current).attr('width', W).attr('height', NH + M.t + 24);
    svgN.selectAll('*').remove();
    const gN = svgN.append('g').attr('transform', `translate(${M.l},${M.t})`);

    const yN     = d3.scaleLinear().domain(d3.extent(data, d => d.close)).range([NH, 0]);
    const navFn  = d3.line().x(d => xFull(d.date)).y(d => yN(d.close)).curve(d3.curveMonotoneX);

    gN.append('path').datum(data)
      .attr('fill', 'none').attr('stroke', primaryColor).attr('stroke-width', 0.8).attr('opacity', 0.35)
      .attr('d', navFn);

    gN.append('g').attr('transform', `translate(0,${NH})`)
      .call(d3.axisBottom(xFull).ticks(4).tickFormat(d3.timeFormat('%Y')))
      .call(darkAxis);

    gN.append('text').attr('x', 4).attr('y', 11)
      .attr('fill', '#444').attr('font-size', '9px').attr('font-family', 'monospace')
      .text('DRAG TO SELECT RANGE');

    // Brush
    const brush = d3.brushX()
      .extent([[0, 0], [IW, NH]])
      .on('brush end', ({ selection }) => {
        const filtered = selection
          ? data.filter(d => { const [d0, d1] = selection.map(xFull.invert); return d.date >= d0 && d.date <= d1; })
          : data;
        redraw(filtered);
        setInfo({ shown: filtered.length, total: data.length, rows: [...filtered].reverse().slice(0, 10) });
      });

    const brushG = gN.append('g').call(brush);
    brushG.select('.selection')
      .attr('fill', primaryColor).attr('fill-opacity', 0.12)
      .attr('stroke', primaryColor).attr('stroke-width', 1);
    brushG.select('.overlay').attr('fill', 'transparent');

    // ── Volume bar chart setup ────────────────────────────────────────────────
    const svgV = d3.select(volRef.current).attr('width', W).attr('height', BH + M.t + M.b);
    svgV.selectAll('*').remove();
    const gV = svgV.append('g').attr('transform', `translate(${M.l},${M.t})`);
    gV.append('text').attr('x', 4).attr('y', 11)
      .attr('fill', '#555').attr('font-size', '9px').attr('font-family', 'monospace')
      .text('VOLUME / MONTH');

    // ── Monthly return bar chart setup ────────────────────────────────────────
    const svgR = d3.select(retRef.current).attr('width', W).attr('height', BH + M.t + M.b);
    svgR.selectAll('*').remove();
    const gR = svgR.append('g').attr('transform', `translate(${M.l},${M.t})`);
    gR.append('text').attr('x', 4).attr('y', 11)
      .attr('fill', '#555').attr('font-size', '9px').attr('font-family', 'monospace')
      .text('MONTHLY RETURN %');

    // ── Redraw (called by brush + on init) ────────────────────────────────────
    function redraw(filtered) {
      if (!filtered.length) return;

      // 1. Price line + area
      const xF = d3.scaleTime().domain(d3.extent(filtered, d => d.date)).range([0, IW]);
      const yP = d3.scaleLinear()
        .domain([d3.min(filtered, d => d.low) * 0.97, d3.max(filtered, d => d.high) * 1.03])
        .range([PH, 0]).nice();

      const areaCurve = d3.area()
        .x(d => xF(d.date)).y0(PH).y1(d => yP(d.close)).curve(d3.curveMonotoneX);
      const lineCurve = d3.line()
        .x(d => xF(d.date)).y(d => yP(d.close)).curve(d3.curveMonotoneX);

      areaPath.datum(filtered).transition().duration(120).attr('d', areaCurve);
      linePath.datum(filtered).transition().duration(120).attr('d', lineCurve);

      gPxAxis.transition().duration(120)
        .call(d3.axisBottom(xF).ticks(Math.min(filtered.length, 6)).tickFormat(d3.timeFormat('%b %Y')))
        .call(darkAxis);

      gPyAxis.transition().duration(120)
        .call(d3.axisLeft(yP).ticks(4).tickFormat(d => `$${d.toFixed(0)}`))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('text').attr('fill', '#777').attr('font-size', '9px'))
        .call(g => g.selectAll('.tick line').attr('stroke', 'none'));

      gPgrid.transition().duration(120)
        .call(d3.axisLeft(yP).ticks(4).tickSize(-IW).tickFormat(''))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('.tick line')
          .attr('stroke', '#1a1a2a').attr('stroke-dasharray', '3,3'));

      // 2–3. Monthly aggregates for the selected range
      const monthly = toMonthly(filtered);
      if (!monthly.length) return;

      const nTicks = Math.max(1, Math.floor(monthly.length / 7));
      const xM = d3.scaleBand().domain(monthly.map(d => +d.date)).range([0, IW]).padding(0.2);

      // Volume
      const yV = d3.scaleLinear().domain([0, d3.max(monthly, d => d.vol)]).range([BH, 0]).nice();

      gV.selectAll('.vb').data(monthly, d => +d.date).join(
        e => e.append('rect').attr('class', 'vb').attr('opacity', 0),
        u => u,
        x => x.remove()
      ).transition().duration(120)
        .attr('x', d => xM(+d.date))
        .attr('y', d => yV(d.vol))
        .attr('width', xM.bandwidth())
        .attr('height', d => BH - yV(d.vol))
        .attr('fill', d => d.ret >= 0 ? primaryColor : negColor)
        .attr('opacity', 0.72);

      gV.selectAll('.vx').data([null]).join('g').attr('class', 'vx')
        .attr('transform', `translate(0,${BH})`)
        .call(d3.axisBottom(xM)
          .tickValues(xM.domain().filter((_, i) => i % nTicks === 0))
          .tickFormat(d => d3.timeFormat('%b %y')(new Date(d))))
        .call(darkAxis);

      gV.selectAll('.vy').data([null]).join('g').attr('class', 'vy')
        .call(d3.axisLeft(yV).ticks(3).tickFormat(d => `${(d / 1e6).toFixed(1)}M`))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('text').attr('fill', '#777').attr('font-size', '9px'));

      // Monthly returns
      const absMax = Math.max(0.01, d3.max(monthly, d => Math.abs(d.ret)));
      const yR = d3.scaleLinear().domain([-absMax, absMax]).range([BH, 0]).nice();
      const z0 = yR(0);

      gR.selectAll('.rb').data(monthly, d => +d.date).join(
        e => e.append('rect').attr('class', 'rb').attr('opacity', 0),
        u => u,
        x => x.remove()
      ).transition().duration(120)
        .attr('x', d => xM(+d.date))
        .attr('y', d => d.ret >= 0 ? yR(d.ret) : z0)
        .attr('width', xM.bandwidth())
        .attr('height', d => Math.abs(yR(d.ret) - z0))
        .attr('fill', d => d.ret >= 0 ? primaryColor : negColor)
        .attr('opacity', 0.82);

      gR.selectAll('.rzero').data([null]).join('line').attr('class', 'rzero')
        .attr('x1', 0).attr('x2', IW).attr('y1', z0).attr('y2', z0)
        .attr('stroke', '#3a3a4a').attr('stroke-width', 1);

      gR.selectAll('.rx').data([null]).join('g').attr('class', 'rx')
        .attr('transform', `translate(0,${BH})`)
        .call(d3.axisBottom(xM)
          .tickValues(xM.domain().filter((_, i) => i % nTicks === 0))
          .tickFormat(d => d3.timeFormat('%b %y')(new Date(d))))
        .call(darkAxis);

      gR.selectAll('.ry').data([null]).join('g').attr('class', 'ry')
        .call(d3.axisLeft(yR).ticks(4).tickFormat(d => `${d.toFixed(2)}%`))
        .call(g => g.select('.domain').remove())
        .call(g => g.selectAll('text').attr('fill', '#777').attr('font-size', '9px'));
    }

    // Initial render — full dataset
    redraw(data);
    setInfo({ shown: data.length, total: data.length, rows: [...data].reverse().slice(0, 10) });

    return () => {
      [priceRef, navRef, volRef, retRef].forEach(r => {
        if (r.current) d3.select(r.current).selectAll('*').remove();
      });
    };
  }, [data, seed, primaryColor, negColor]); // eslint-disable-line react-hooks/exhaustive-deps

  const fmt = n => `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;

  return (
    <div ref={wrapRef} className="rounded-2xl border border-primary/10 bg-dark overflow-hidden mt-12">
      {/* Header */}
      <div className="px-5 py-3 border-b border-primary/10 flex items-center justify-between flex-wrap gap-2">
        <div>
          <span className="text-sm font-bold text-primary font-mono">{name}</span>
          <span className="text-xs text-light/25 ml-3 hidden sm:inline">
            Drag the range selector · all charts filter together
          </span>
        </div>
        <span className="text-xs font-mono text-light/40">
          <span className="text-primary/70">{info.shown}</span>
          <span className="text-light/20"> / </span>
          {info.total} sessions
        </span>
      </div>

      {/* Charts */}
      <div className="px-3 pt-4 space-y-1">
        {/* Price line */}
        <svg ref={priceRef} className="block w-full" />
        {/* Navigator brush */}
        <svg ref={navRef}  className="block w-full" />
        <svg ref={volRef} className="block w-full mt-3" />
        <svg ref={retRef} className="block w-full" />
      </div>

      {/* Scrollable data table */}
      <div className="border-t border-primary/10 overflow-x-auto" style={{ maxHeight: '240px', overflowY: 'auto' }}>
        <table className="w-full text-xs font-mono min-w-[580px]">
          <thead className="sticky top-0 bg-dark z-10">
            <tr className="border-b border-primary/10">
              {['Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Volume'].map(h => (
                <th key={h} className="text-left px-3 py-2 text-light/25 font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {info.rows.map((row, i) => (
              <tr key={i} className="border-b border-primary/5 hover:bg-surface/20 transition-colors">
                <td className="px-3 py-1.5 text-light/40">{d3.timeFormat('%Y-%m-%d')(row.date)}</td>
                <td className="px-3 py-1.5 text-light/55">${row.open.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-light/55">${row.high.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-light/55">${row.low.toFixed(2)}</td>
                <td className="px-3 py-1.5 text-light font-semibold">${row.close.toFixed(2)}</td>
                <td className={`px-3 py-1.5 ${row.pct >= 0 ? 'text-primary' : 'text-coral'}`}>{fmt(row.pct)}</td>
                <td className="px-3 py-1.5 text-light/35">{(row.volume / 1e6).toFixed(2)}M</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
