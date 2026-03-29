import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

/**
 * Multi-modal transformation cycle visualization.
 * Shows Circuit ↔ Sequence ↔ Gas transformations with information preservation.
 */
export default function TransformCycle({ width = 500, height = 400 }) {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const cx = width / 2
    const cy = height / 2
    const R = Math.min(width, height) * 0.3

    const teal = '#2ca89a'
    const coral = '#e8734a'
    const gold = '#d4a843'

    const modes = [
      { name: 'Circuit', symbol: '\u26A1', angle: -Math.PI / 2, color: teal, desc: 'Kirchhoff Laws' },
      { name: 'Sequence', symbol: '\u2630', angle: Math.PI / 6, color: coral, desc: 'Directional Encoding' },
      { name: 'Gas', symbol: '\u2B24', angle: 5 * Math.PI / 6, color: gold, desc: 'Molecular Dynamics' },
    ]

    // Compute positions
    modes.forEach(m => {
      m.x = cx + R * Math.cos(m.angle)
      m.y = cy + R * Math.sin(m.angle)
    })

    // Transformation arrows (curved)
    const transforms = [
      { from: 0, to: 1, label: 'T_{C\u2192S}' },
      { from: 1, to: 2, label: 'T_{S\u2192G}' },
      { from: 2, to: 0, label: 'T_{G\u2192C}' },
    ]

    // Draw curved arrows
    transforms.forEach((t, i) => {
      const s = modes[t.from]
      const e = modes[t.to]

      // Control point for curve (offset toward center)
      const mx = (s.x + e.x) / 2 + (cy - (s.y + e.y) / 2) * 0.3
      const my = (s.y + e.y) / 2 - (cx - (s.x + e.x) / 2) * 0.3

      const path = svg.append('path')
        .attr('d', `M ${s.x} ${s.y} Q ${mx} ${my} ${e.x} ${e.y}`)
        .attr('fill', 'none')
        .attr('stroke', 'rgba(255,255,255,0.2)')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrow)')

      // Animated dot along path
      const dot = svg.append('circle')
        .attr('r', 4)
        .attr('fill', modes[t.from].color)
        .attr('opacity', 0.8)

      const pathNode = path.node()
      const totalLength = pathNode.getTotalLength()

      function animateDot() {
        dot.attr('opacity', 0.8)
        const start = performance.now()
        const duration = 3000

        function step(now) {
          const elapsed = (now - start) % (duration + 1000)
          if (elapsed > duration) {
            dot.attr('opacity', 0)
            requestAnimationFrame(step)
            return
          }
          const progress = elapsed / duration
          const point = pathNode.getPointAtLength(progress * totalLength)
          dot.attr('cx', point.x).attr('cy', point.y).attr('opacity', 0.8)
          requestAnimationFrame(step)
        }
        requestAnimationFrame(step)
      }
      setTimeout(animateDot, i * 1000)

      // Label
      const labelX = (s.x + e.x) / 2 + (cy - (s.y + e.y) / 2) * 0.15
      const labelY = (s.y + e.y) / 2 - (cx - (s.x + e.x) / 2) * 0.15
      svg.append('text')
        .attr('x', labelX).attr('y', labelY)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.4)')
        .attr('font-size', 9)
        .attr('font-family', 'monospace')
        .text(t.label)
    })

    // Arrow marker
    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', 'rgba(255,255,255,0.3)')

    // Mode nodes
    modes.forEach(m => {
      // Outer glow
      svg.append('circle')
        .attr('cx', m.x).attr('cy', m.y).attr('r', 40)
        .attr('fill', m.color).attr('fill-opacity', 0.08)
        .attr('stroke', m.color).attr('stroke-opacity', 0.3)

      // Inner circle
      svg.append('circle')
        .attr('cx', m.x).attr('cy', m.y).attr('r', 28)
        .attr('fill', 'rgba(10,14,23,0.9)')
        .attr('stroke', m.color).attr('stroke-width', 2)

      // Symbol
      svg.append('text')
        .attr('x', m.x).attr('y', m.y + 4)
        .attr('text-anchor', 'middle')
        .attr('fill', m.color)
        .attr('font-size', 18)
        .text(m.symbol)

      // Name
      svg.append('text')
        .attr('x', m.x).attr('y', m.y + 50)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('font-size', 11)
        .attr('font-weight', 'bold')
        .text(m.name)

      // Description
      svg.append('text')
        .attr('x', m.x).attr('y', m.y + 64)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.4)')
        .attr('font-size', 8)
        .text(m.desc)
    })

    // Center info preservation badge
    svg.append('circle')
      .attr('cx', cx).attr('cy', cy).attr('r', 32)
      .attr('fill', 'rgba(44,168,154,0.1)')
      .attr('stroke', teal).attr('stroke-opacity', 0.4)
      .attr('stroke-dasharray', '4,4')

    svg.append('text')
      .attr('x', cx).attr('y', cy - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', teal)
      .attr('font-size', 14)
      .attr('font-weight', 'bold')
      .text('>95%')

    svg.append('text')
      .attr('x', cx).attr('y', cy + 10)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.5)')
      .attr('font-size', 8)
      .text('info preserved')

  }, [width, height])

  return <svg ref={svgRef} width={width} height={height} style={{ background: 'transparent' }} />
}
