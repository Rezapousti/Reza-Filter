# Reza Filter

## Quick start
```python
import reza
```

## Install
```bash
python -m pip install -U pip
python -m pip install reza-filter
```

## Band-pass
```bash
y = reza.bandpass(x, fs, fc_low, fc_high)
```

## Low-pass / High-pass
```bash
y_lp = reza.lowpass(x, fs, fc)
y_hp = reza.highpass(x, fs, fc)
```
