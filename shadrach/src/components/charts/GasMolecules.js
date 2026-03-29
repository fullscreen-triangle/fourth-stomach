import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

/**
 * Gas molecular dynamics visualization.
 * Molecules bounce in a bounded chamber, collide, and form harmonic coincidences.
 */
export default function GasMolecules({ width = 500, height = 400, nMolecules = 30 }) {
  const svgRef = useRef()
  const animRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const margin = 20
    const teal = '#2ca89a'
    const coral = '#e8734a'
    const gold = '#d4a843'

    // Chamber boundary
    svg.append('rect')
      .attr('x', margin).attr('y', margin)
      .attr('width', width - 2 * margin).attr('height', height - 2 * margin)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(44,168,154,0.3)')
      .attr('stroke-width', 1)
      .attr('rx', 4)

    // Initialize molecules
    const molecules = Array.from({ length: nMolecules }, (_, i) => ({
      id: i,
      x: margin + 20 + Math.random() * (width - 2 * margin - 40),
      y: margin + 20 + Math.random() * (height - 2 * margin - 40),
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      r: 4 + Math.random() * 4,
      omega: 0.5 + Math.random() * 3, // characteristic frequency
      phase: Math.random() * Math.PI * 2,
    }))

    // Harmonic coincidence edges (pre-compute)
    const coincidences = []
    for (let i = 0; i < nMolecules; i++) {
      for (let j = i + 1; j < nMolecules; j++) {
        for (let n = 1; n <= 3; n++) {
          for (let m = 1; m <= 3; m++) {
            if (Math.abs(n * molecules[i].omega - m * molecules[j].omega) < 0.3) {
              coincidences.push({ i, j, strength: 1 - Math.abs(n * molecules[i].omega - m * molecules[j].omega) / 0.3 })
              break
            }
          }
        }
      }
    }

    // Draw coincidence edges
    const edgeG = svg.append('g')

    // Draw molecules
    const circles = svg.append('g')
      .selectAll('circle')
      .data(molecules)
      .enter()
      .append('circle')
      .attr('r', d => d.r)
      .attr('fill', (d, i) => i < nMolecules / 3 ? teal : i < 2 * nMolecules / 3 ? coral : gold)
      .attr('fill-opacity', 0.6)
      .attr('stroke', (d, i) => i < nMolecules / 3 ? teal : i < 2 * nMolecules / 3 ? coral : gold)
      .attr('stroke-opacity', 0.8)

    // Wavefunction rings
    const waves = svg.append('g')
      .selectAll('circle')
      .data(molecules)
      .enter()
      .append('circle')
      .attr('r', d => d.r + 8)
      .attr('fill', 'none')
      .attr('stroke', (d, i) => i < nMolecules / 3 ? teal : i < 2 * nMolecules / 3 ? coral : gold)
      .attr('stroke-opacity', 0.15)
      .attr('stroke-width', 1)

    let t = 0
    function tick() {
      t += 0.016

      // Update positions
      molecules.forEach(m => {
        m.x += m.vx
        m.y += m.vy

        // Bounce off walls
        if (m.x - m.r < margin) { m.x = margin + m.r; m.vx *= -1 }
        if (m.x + m.r > width - margin) { m.x = width - margin - m.r; m.vx *= -1 }
        if (m.y - m.r < margin) { m.y = margin + m.r; m.vy *= -1 }
        if (m.y + m.r > height - margin) { m.y = height - margin - m.r; m.vy *= -1 }
      })

      // Update circles
      circles.attr('cx', d => d.x).attr('cy', d => d.y)

      // Pulsing wavefunction rings
      waves
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => d.r + 6 + Math.sin(t * d.omega + d.phase) * 4)
        .attr('stroke-opacity', d => 0.1 + Math.abs(Math.sin(t * d.omega + d.phase)) * 0.15)

      // Update coincidence edges
      edgeG.selectAll('line').remove()
      coincidences.forEach(c => {
        const a = molecules[c.i]
        const b = molecules[c.j]
        const dist = Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        if (dist < 150) {
          edgeG.append('line')
            .attr('x1', a.x).attr('y1', a.y)
            .attr('x2', b.x).attr('y2', b.y)
            .attr('stroke', teal)
            .attr('stroke-opacity', c.strength * 0.3 * (1 - dist / 150))
            .attr('stroke-width', 1)
        }
      })

      animRef.current = requestAnimationFrame(tick)
    }

    animRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animRef.current)
  }, [width, height, nMolecules])

  return <svg ref={svgRef} width={width} height={height} style={{ background: 'transparent' }} />
}
