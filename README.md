<p align="center">
  <img src="assets/images/SR-71-crews.png" alt="Logo" width="300"/>
</p>

A computational framework implementing circulation-based transaction networks, harmonic pattern analysis, and multi-modal financial representation. The system integrates graph-theoretic optimization, spectral analysis, and S-entropy navigation to provide efficient economic coordination mechanisms.

## Overview

This framework implements several interconnected systems:

1. **Circulation Transaction Networks (CTN)**: Batch settlement systems that reduce transaction verification complexity through deferred processing and graph reduction algorithms.

2. **Shadow Transaction Networks (STN)**: Harmonic coincidence detection for identifying latent correlations in transaction patterns through spectral decomposition and frequency analysis.

3. **Graph Completion Finance (GCF)**: Topology-based lending mechanisms that leverage network flow patterns to optimize capital allocation.

4. **Multi-Modal Representation**: Transformation system enabling circuit, sequence, and gas molecular representations of financial networks with proven information preservation properties.

5. **Temporal Arbitrage Framework**: Intraday capital optimization leveraging circulation certainty and shadow network intelligence.

## Theoretical Foundation

The framework is grounded in formal mathematical economics and network theory, with publications documenting:

- Complexity reduction from $O(N)$ to $O(n \log n)$ for settlement operations
- Harmonic coincidence detection with computational complexity $O(NH)$
- Information preservation across representational transformations (>95%)
- Graph completion lending with provable repayment bounds

Detailed theoretical foundations are available in `docs/publication/`.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fourth-stomach.git
cd fourth-stomach

# Install the package
pip install -e .
```

### Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

### Using Make (Unix/Linux/macOS)

```bash
# Install core dependencies
make install

# Install with development tools
make install-dev
```

## Quick Start

### Example 1: Circulation Transaction Network

```python
from ctn import CirculationTransactionNetwork, Transaction

# Initialize network
ctn = CirculationTransactionNetwork()

# Add entities
ctn.add_node("Alice")
ctn.add_node("Bob")
ctn.add_node("Charlie")

# Process transactions
ctn.process_transaction(Transaction("Alice", "Bob", 1500.0, timestamp=1.0))
ctn.process_transaction(Transaction("Bob", "Charlie", 800.0, timestamp=2.0))
ctn.process_transaction(Transaction("Charlie", "Alice", 2000.0, timestamp=3.0))

# End-of-day settlement
settlement = ctn.settle_end_of_day()
print(f"Settlements required: {len(settlement)}")
```

### Example 2: Shadow Network Analysis

```python
from ctn import ShadowTransactionNetwork

# Initialize shadow network
shadow = ShadowTransactionNetwork()

# Extract transaction patterns (requires historical data)
patterns = shadow.extract_patterns(transactions, window_size=30)

# Detect harmonic coincidences
coincidences = shadow.detect_harmonics(patterns, epsilon_tol=0.05)

# Build correlation network
correlation_graph = shadow.build_shadow_graph(coincidences)
```

### Example 3: Multi-Modal Representation

```python
from representation import FinancialCircuit, RepresentationTransformer

# Create circuit representation
circuit = FinancialCircuit()
circuit.add_node("Alice", net_worth=10000, credit_capacity=5000)
circuit.add_resistor("Alice", "Bob", resistance=0.05)

# Transform to other representations
transformer = RepresentationTransformer()
results = transformer.full_cycle_transform(transactions)

print(f"Information preservation: {results['information_preservation']:.1%}")
```

## Running Demonstrations

The framework includes several demonstration scripts:

```bash
# Circulation network demonstration
python src/ctn/demo_circulation.py

# Shadow network analysis
python src/ctn/demo_shadow_network.py

# Graph completion finance
python src/ctn/demo_graph_completion_finance.py

# Multi-modal representations
python src/representation/demo_representations.py
```

Or using Make:

```bash
make demo-ctn      # Circulation network
make demo-shadow   # Shadow network
make demo-gcf      # Graph completion
make demo-rep      # Representations
```

## Project Structure

```
fourth-stomach/
├── src/
│   ├── ctn/                    # Circulation transaction networks
│   │   ├── transaction_graph.py
│   │   ├── shadow_network.py
│   │   ├── graph_completion_finance.py
│   │   └── visualization.py
│   ├── representation/         # Multi-modal representations
│   │   ├── circuit.py
│   │   ├── sequence.py
│   │   ├── gas_molecules.py
│   │   ├── semantic.py
│   │   ├── shadow.py
│   │   └── moon_landing.py
│   ├── harmonic/              # Harmonic analysis (planned)
│   ├── jangara/               # Remittance optimization (planned)
│   ├── laboratory/            # Financial simulation (planned)
│   └── reality/               # Reality-state currency (planned)
├── tests/                     # Test suite
├── docs/
│   ├── publication/           # Academic papers (LaTeX)
│   ├── philosophy/            # Theoretical foundations
│   ├── economics/             # Economic theory
│   └── algorithms/            # Algorithm specifications
├── pyproject.toml             # Package configuration
├── requirements.txt           # Dependencies
└── Makefile                   # Development commands
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Or using Make
make test
make test-cov
```

## Documentation

### Core Documentation

- **Installation & Setup**: This README
- **Theoretical Foundation**: `docs/publication/` (LaTeX sources)
- **API Documentation**: Inline docstrings (PEP 257 compliant)
- **Implementation Details**: `docs/PROJECT_OVERVIEW.md`

### Key Publications

The framework is based on peer-review-ready research:

1. **Harmonic Coincidence Networks** (`docs/publication/harmonic-network-graph.tex`)

   - Spectral decomposition of transaction time series
   - Correlation network construction via harmonic coincidence
   - Computational complexity: $O(NH)$

2. **Graph Completion Lending** (`docs/publication/credit-graph-network.tex`)

   - Topology-based credit allocation
   - Flow gap identification and completion
   - Provable repayment bounds

3. **Circulation Transaction Networks** (`docs/publication/circulation-transactions-network.tex`)

   - Batch verification and deferred settlement
   - Complexity reduction to $O(n \log n)$
   - Kirchhoff's law interpretation

4. **Temporal Arbitrage** (`docs/publication/temporal-arbitrage-in-circulation-networks.tex`)

   - Intraday liquidity optimization
   - Settlement certainty quantification
   - Risk-adjusted return analysis

5. **Multi-Modal Representation** (`docs/publication/financial-representation.tex`)

   - Circuit, sequence, and gas molecular models
   - Information-preserving transformations
   - Semantic distance amplification

6. **Fourth Stomach Framework** (`docs/publication/fourth-stomach.tex`)

   - Unified circulatory processing system
   - Flux-based equilibrium convergence
   - Four-chamber architecture

7. **Validation Framework** (`docs/publication/validation-framework.tex`)
   - Experimental protocols for all publications
   - Statistical validation methods
   - Progressive deployment roadmap

## Performance Characteristics

Current implementation performance (on standard hardware):

- **Transaction processing**: 10,000+ transactions/second
- **Settlement complexity**: $O(n \log n)$ where $n \ll N$
- **Pattern extraction**: $O(NH)$ via FFT
- **Memory footprint**: $O(n + m)$ for $n$ nodes, $m$ edges

Scalability targets:

| Phase | Nodes       | Daily Transactions | Throughput    |
| ----- | ----------- | ------------------ | ------------- |
| 1     | 1,000       | 10,000             | 100 TPS       |
| 2     | 100,000     | 10,000,000         | 10,000 TPS    |
| 3     | 10,000,000+ | 1,000,000,000+     | 1,000,000 TPS |

## Development

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking with mypy
mypy src/
```

### Contributing

Contributions should:

1. Include tests for new functionality
2. Maintain >90% code coverage
3. Follow PEP 8 style guidelines (enforced by Black)
4. Include type hints (PEP 484)
5. Update documentation as needed

## Validation Framework

Comprehensive experimental validation protocols are documented in `docs/publication/validation-framework.tex`, including:

- Synthetic data generation
- Historical backtesting
- Statistical validation (p < 0.001, Cohen's d > 2.0)
- Progressive deployment roadmap

## License

MIT License - See LICENSE file for details.

## Citation

If you use this framework in academic work, please cite:

```bibtex
@software{fourth_stomach,
  title = {Fourth Stomach: Unified Economic Coordination Framework},
  author = {Sachikonye, Kundai Farai},
  year = {2024},
  url = {https://github.com/yourusername/fourth-stomach}
}
```

Individual papers in `docs/publication/` have their own citation formats documented within.

## Contact

For questions, issues, or collaboration inquiries:

- GitHub Issues: [github.com/yourusername/fourth-stomach/issues](https://github.com/yourusername/fourth-stomach/issues)
- Email: [your.email@example.com](mailto:your.email@example.com)

## Acknowledgments

This work builds on theoretical foundations documented in `docs/philosophy/`, `docs/economics/`, and `docs/time/`, integrating concepts from:

- Network theory and graph algorithms
- Spectral analysis and harmonic decomposition
- Thermodynamic optimization principles
- S-entropy navigation frameworks

## References

See individual publications in `docs/publication/` for detailed bibliographies and mathematical derivations.
