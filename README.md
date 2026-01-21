# Reza Filter (C++-accelerated) — Python package

**Goal:** users do:

```python
import reza
y = reza.bandpass(x, fs=100.0, fc_low=0.5, fc_high=5.0)
```

## What is accelerated in C++
- Gain template generation (low/high/band)
- Auto-`d` selection via edge-sharpness convergence
- rFFT-domain complex multiply (X * gain)

FFT/iFFT uses NumPy (already optimized).

## Install (after you publish to PyPI)
```bash
pip install reza-filter
```

## Local dev install
```bash
python -m pip install -U pip
python -m pip install -e .
python -c "import reza; print('has_cpp=', reza.has_cpp()); print(reza.__version__)"
```

## Build wheels (recommended for public release)
Use GitHub Actions + cibuildwheel: see `.github/workflows/wheels.yml`.

## Publish to PyPI (after wheels exist)
```bash
python -m pip install -U build twine
python -m build
python -m twine upload dist/*
```
