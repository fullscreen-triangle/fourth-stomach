import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

/**
 * Interactive circuit graph visualization.
 * Nodes = entities with potentials, Edges = transaction flows with R/L/C properties.
 */
export default function CircuitGraph({ width = 500, height = 400 }) {
  const svgRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const nodes = [
      { id: 'A', label: 'Alice', V: 8.2, x: 100, y: 200 },
      { id: 'B', label: 'Bob', V: 5.7, x: 250, y: 80 },
      { id: 'C', label: 'Charlie', V: 6.9, x: 400, y: 200 },
      { id: 'D', label: 'Diana', V: 4.3, x: 250, y: 320 },
      { id: 'E', label: 'Eve', V: 7.1, x: 180, y: 140 },
      { id: 'F', label: 'Frank', V: 3.8, x: 320, y: 140 },
    ]

    const links = [
      { source: 'A', target: 'B', I: 2.5, R: 1.0, type: 'R' },
      { source: 'B', target: 'C', I: 1.8, R: 1.4, type: 'R' },
      { source: 'C', target: 'D', I: 3.1, R: 0.7, type: 'C' },
      { source: 'D', target: 'A', I: 2.0, R: 1.2, type: 'L' },
      { source: 'A', target: 'E', I: 1.2, R: 0.9, type: 'R' },
      { source: 'E', target: 'B', I: 0.8, R: 1.1, type: 'C' },
      { source: 'B', target: 'F', I: 1.5, R: 0.6, type: 'R' },
      { source: 'F', target: 'C', I: 1.3, R: 0.8, type: 'L' },
      { source: 'E', target: 'F', I: 0.7, R: 1.3, type: 'R' },
    ]

    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]))
    const teal = '#2ca89a'
    const coral = '#e8734a'
    const gold = '#d4a843'

    const typeColor = { R: teal, C: coral, L: gold }
    const typeLabel = { R: 'R', C: 'C', L: 'L' }

    // Edges
    const linkG = svg.append('g')
    links.forEach(link => {
      const s = nodeMap[link.source]
      const t = nodeMap[link.target]
      const strokeW = Math.max(1, link.I * 1.5)

      linkG.append('line')
        .attr('x1', s.x).attr('y1', s.y)
        .attr('x2', t.x).attr('y2', t.y)
        .attr('stroke', typeColor[link.type])
        .attr('stroke-width', strokeW)
        .attr('stroke-opacity', 0.5)

      // Current flow indicator (animated dot)
      const mx = (s.x + t.x) / 2
      const my = (s.y + t.y) / 2
      linkG.append('text')
        .attr('x', mx).attr('y', my - 6)
        .attr('text-anchor', 'middle')
        .attr('fill', typeColor[link.type])
        .attr('font-size', 9)
        .attr('font-family', 'monospace')
        .text(`${typeLabel[link.type]}=${link.R.toFixed(1)}`)

      linkG.append('text')
        .attr('x', mx).attr('y', my + 10)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.4)')
        .attr('font-size', 8)
        .attr('font-family', 'monospace')
        .text(`I=${link.I.toFixed(1)}`)
    })

    // Animated current pulses
    links.forEach((link, i) => {
      const s = nodeMap[link.source]
      const t = nodeMap[link.target]
      const dot = linkG.append('circle')
        .attr('r', 3)
        .attr('fill', typeColor[link.type])
        .attr('opacity', 0.8)

      function animate() {
        dot.attr('cx', s.x).attr('cy', s.y)
          .transition()
          .duration(2000 + i * 300)
          .ease(d3.easeLinear)
          .attr('cx', t.x).attr('cy', t.y)
          .on('end', animate)
      }
      setTimeout(animate, i * 200)
    })

    // Nodes
    const potentialScale = d3.scaleLinear().domain([3, 9]).range([12, 24])

    const nodeG = svg.append('g')
    nodes.forEach(node => {
      const r = potentialScale(node.V)
      nodeG.append('circle')
        .attr('cx', node.x).attr('cy', node.y)
        .attr('r', r)
        .attr('fill', 'rgba(44,168,154,0.15)')
        .attr('stroke', teal)
        .attr('stroke-width', 2)

      nodeG.append('text')
        .attr('x', node.x).attr('y', node.y + 3)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('font-size', 10)
        .attr('font-weight', 'bold')
        .text(node.id)

      nodeG.append('text')
        .attr('x', node.x).attr('y', node.y + r + 14)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.5)')
        .attr('font-size', 8)
        .attr('font-family', 'monospace')
        .text(`V=${node.V}`)
    })

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${width - 100}, 20)`)
    ;[['R (Resistor)', teal], ['C (Capacitor)', coral], ['L (Inductor)', gold]].forEach(([label, color], i) => {
      legend.append('line').attr('x1', 0).attr('y1', i * 16).attr('x2', 16).attr('y2', i * 16)
        .attr('stroke', color).attr('stroke-width', 2)
      legend.append('text').attr('x', 22).attr('y', i * 16 + 4)
        .attr('fill', 'rgba(255,255,255,0.6)').attr('font-size', 9).text(label)
    })

  }, [width, height])

  return <svg ref={svgRef} width={width} height={height} style={{ background: 'transparent' }} />
}
