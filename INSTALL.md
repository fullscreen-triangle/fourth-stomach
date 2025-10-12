# Installation Guide

## System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (8GB recommended for large-scale simulations)
- **Disk Space**: ~500MB for installation and dependencies

## Installation Methods

### Method 1: Basic Installation (Recommended)

For users who want to run the framework:

```bash
# Clone the repository
git clone https://github.com/yourusername/fourth-stomach.git
cd fourth-stomach

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the package
pip install -e .
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/yourusername/fourth-stomach.git
cd fourth-stomach

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Using Make (Unix-like systems)

If you have `make` installed:

```bash
# Clone repository
git clone https://github.com/yourusername/fourth-stomach.git
cd fourth-stomach

# Install core package
make install

# Or install with development tools
make install-dev
```

## Verification

Verify the installation:

```bash
# Check Python version
python --version  # Should be 3.9 or higher

# Run tests
pytest tests/ -v

# Or using Make
make test
```

## Running Demos

Test the installation by running demonstration scripts:

```bash
# Circulation Transaction Network
python src/ctn/demo_circulation.py

# Shadow Network Analysis
python src/ctn/demo_shadow_network.py

# Multi-Modal Representations
python src/representation/demo_representations.py

# Or using Make
make demo-ctn
make demo-shadow
make demo-rep
```

## Troubleshooting

### ImportError: No module named 'numpy'

The dependencies weren't installed correctly. Try:

```bash
pip install -r requirements.txt
```

### ModuleNotFoundError: No module named 'ctn'

The package wasn't installed in editable mode. Ensure you're in the project root and run:

```bash
pip install -e .
```

### Tests Failing

Ensure you have the development dependencies:

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

### Windows-Specific Issues

On Windows, you may need to:

1. Install Visual C++ Build Tools if scipy installation fails
2. Use `python -m pip` instead of `pip` directly
3. Run PowerShell or Command Prompt as Administrator

### Python Version Issues

Check your Python version:

```bash
python --version
```

If you have multiple Python versions, you may need to specify:

```bash
python3.9 -m venv venv
# or
python3.10 -m venv venv
```

## Optional Dependencies

### Database Support (for future persistence)

```bash
pip install sqlalchemy redis
```

### Web API Support (for future REST API)

```bash
pip install fastapi uvicorn websockets
```

### Documentation Building

```bash
pip install sphinx sphinx-rtd-theme
```

### Cryptography (for MDTEC integration)

```bash
pip install cryptography pynacl
```

## Docker Installation (Alternative)

A Dockerfile will be provided in future releases for containerized deployment.

## Updating

To update to the latest version:

```bash
cd fourth-stomach
git pull origin main
pip install -e . --upgrade
```

## Uninstallation

To remove the package:

```bash
pip uninstall fourth-stomach
```

To completely remove the environment:

```bash
deactivate  # Exit virtual environment
rm -rf venv/  # Remove virtual environment directory
```

## Next Steps

After installation:

1. **Read the documentation**: Start with `README.md`
2. **Run demos**: Try `demo_circulation.py` and other demos
3. **Read theory**: Review papers in `docs/publication/`
4. **Run tests**: Verify everything works with `pytest`
5. **Explore code**: Check `src/ctn/` and `src/representation/`

## Support

For installation issues:

- Check existing issues: https://github.com/yourusername/fourth-stomach/issues
- Open a new issue with:
  - Your Python version (`python --version`)
  - Your OS and version
  - Full error message
  - Steps to reproduce

## Development Setup

For developers contributing to the project:

```bash
# Clone and install in dev mode
git clone https://github.com/yourusername/fourth-stomach.git
cd fourth-stomach
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Set up pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Format code
make format

# Run linters
make lint

# Run tests with coverage
make test-cov
```

## Platform-Specific Notes

### Linux

All features fully supported. Install system dependencies if needed:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev

# Fedora/RHEL
sudo dnf install python3-devel
```

### macOS

Fully supported. You may need to install Xcode Command Line Tools:

```bash
xcode-select --install
```

### Windows

Supported with minor limitations:

- Make commands unavailable (use Python commands directly)
- Some path separators differ (handled automatically by pathlib)
- Recommended: Use Windows Terminal or PowerShell

## Performance Optimization

For better performance on large datasets:

```bash
# Install optimized BLAS/LAPACK
pip install numpy scipy --upgrade --force-reinstall

# Consider installing Intel MKL (if using Intel CPU)
pip install mkl
```

## Citation

If using this software in academic work, see the citation format in `README.md`.
