# Reza Filter (Exponential Frequency-Domain Filters)

This repository provides three lightweight, copy/paste-ready Python functions for filtering 1D or multi-channel time-series data using the **Reza exponential gain** approach in the frequency domain:

- `reza_low_pass_filter`
- `reza_high_pass_filter`
- `reza_band_pass_filter`

Unlike classical IIR/FIR designs, these functions shape the spectrum using an exponential magnitude template. The filters are **zero-phase (acausal)** because they operate in the frequency domain (FFT → magnitude shaping → inverse FFT).

## Install
No installation required. Copy a function into your project.

Dependencies:
- NumPy

## Quick usage

```python
import numpy as np

# x: shape (n_samples,) or (n_channels, n_samples)
y_lp = reza_low_pass_filter(x, fs=250, fc=30)
y_hp = reza_high_pass_filter(x, fs=250, fc=1)
y_bp = reza_band_pass_filter(x, fs=250, fc1=5, fc2=8)

# If your data is (n_samples, n_channels), set axis=0 or axis=1 appropriately
y_bp = reza_band_pass_filter(x, fs=250, fc1=5, fc2=8, axis=0)


