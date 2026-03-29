import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

/**
 * Directional sequence encoding visualization.
 * Shows transactions encoded as cardinal directions scrolling in real-time.
 */
export default function SequenceStream({ width = 500, height = 400 }) {
  const svgRef = useRef()
  const animRef = useRef()

  useEffect(() => {
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const teal = '#2ca89a'
    const coral = '#e8734a'
    const gold = '#d4a843'
    const navy = '#4a90d9'
    const green = '#22c55e'
    const purple = '#a855f7'

    const directions = [
      { name: 'N', symbol: '\u2191', color: green, label: 'Large' },
      { name: 'S', symbol: '\u2193', color: coral, label: 'Small' },
      { name: 'E', symbol: '\u2192', color: teal, label: 'Frequent' },
      { name: 'W', symbol: '\u2190', color: gold, label: 'Rare' },
      { name: 'U', symbol: '\u2299', color: navy, label: 'High Profit' },
      { name: 'D', symbol: '\u2297', color: purple, label: 'Low Profit' },
    ]

    // Generate random sequence
    const seqLength = 60
    const sequence = Array.from({ length: seqLength }, () =>
      directions[Math.floor(Math.random() * directions.length)]
    )

    // Sequence display area
    const seqY = 60
    const cellSize = 14
    const cols = Math.floor((width - 40) / cellSize)

    // Title
    svg.append('text')
      .attr('x', width / 2).attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.5)')
      .attr('font-size', 10)
      .attr('font-family', 'monospace')
      .text('DIRECTIONAL SEQUENCE ENCODING')

    // Sequence grid
    const seqG = svg.append('g').attr('transform', `translate(20, ${seqY})`)

    // Semantic distance bars
    const barY = seqY + Math.ceil(seqLength / cols) * cellSize + 40
    const layers = [
      { name: 'Directional', gamma: 3.7, color: teal },
      { name: 'Positional', gamma: 4.2, color: coral },
      { name: 'Frequency', gamma: 5.8, color: gold },
      { name: 'Compressed', gamma: 7.3, color: navy },
    ]

    svg.append('text')
      .attr('x', width / 2).attr('y', barY - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.4)')
      .attr('font-size', 9)
      .attr('font-family', 'monospace')
      .text('SEMANTIC AMPLIFICATION LAYERS')

    const barScale = d3.scaleLog().domain([1, 700]).range([0, width - 80])
    let cumGamma = 1

    layers.forEach((layer, i) => {
      cumGamma *= layer.gamma
      const barW = barScale(cumGamma)
      const y = barY + i * 28

      svg.append('rect')
        .attr('x', 40).attr('y', y)
        .attr('width', 0).attr('height', 18)
        .attr('fill', layer.color)
        .attr('fill-opacity', 0.3)
        .attr('stroke', layer.color)
        .attr('stroke-opacity', 0.6)
        .attr('rx', 3)
        .transition()
        .duration(800)
        .delay(i * 200)
        .attr('width', barW)

      svg.append('text')
        .attr('x', 36).attr('y', y + 12)
        .attr('text-anchor', 'end')
        .attr('fill', 'rgba(255,255,255,0.5)')
        .attr('font-size', 8)
        .attr('font-family', 'monospace')
        .text(layer.name)

      svg.append('text')
        .attr('x', 44 + barW).attr('y', y + 12)
        .attr('fill', layer.color)
        .attr('font-size', 9)
        .attr('font-weight', 'bold')
        .attr('font-family', 'monospace')
        .attr('opacity', 0)
        .transition()
        .delay(i * 200 + 600)
        .attr('opacity', 1)
        .text(`\u00d7${layer.gamma}`)
    })

    // Total amplification
    svg.append('text')
      .attr('x', width / 2).attr('y', barY + layers.length * 28 + 20)
      .attr('text-anchor', 'middle')
      .attr('fill', teal)
      .attr('font-size', 12)
      .attr('font-weight', 'bold')
      .attr('font-family', 'monospace')
      .attr('opacity', 0)
      .transition()
      .delay(1200)
      .attr('opacity', 1)
      .text(`Total amplification: \u0393 \u2248 658\u00d7`)

    // Legend
    const legendG = svg.append('g').attr('transform', `translate(20, ${height - 30})`)
    directions.forEach((d, i) => {
      const lx = i * ((width - 40) / 6)
      legendG.append('text')
        .attr('x', lx).attr('y', 0)
        .attr('fill', d.color)
        .attr('font-size', 14)
        .text(d.symbol)
      legendG.append('text')
        .attr('x', lx + 16).attr('y', 0)
        .attr('fill', 'rgba(255,255,255,0.4)')
        .attr('font-size', 8)
        .text(d.label)
    })

    // Animate sequence appearing
    let idx = 0
    function addSymbol() {
      if (idx >= sequence.length) return
      const d = sequence[idx]
      const col = idx % cols
      const row = Math.floor(idx / cols)

      seqG.append('text')
        .attr('x', col * cellSize + cellSize / 2)
        .attr('y', row * cellSize + cellSize / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('fill', d.color)
        .attr('font-size', 12)
        .attr('opacity', 0)
        .transition()
        .duration(150)
        .attr('opacity', 0.8)
        .text(d.symbol)

      idx++
      if (idx < sequence.length) {
        animRef.current = setTimeout(addSymbol, 50)
      }
    }
    addSymbol()

    return () => clearTimeout(animRef.current)
  }, [width, height])

  return <svg ref={svgRef} width={width} height={height} style={{ background: 'transparent' }} />
}
