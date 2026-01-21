from __future__ import annotations

import numpy as np
from . import _fallback

try:
    from . import _reza_cpp as _cpp  # compiled extension
    _HAS_CPP = True
except Exception:
    _cpp = None
    _HAS_CPP = False

__all__ = ["lowpass", "highpass", "bandpass", "filter_signal", "has_cpp", "__version__"]
__version__ = "0.2.0"

def has_cpp() -> bool:
    return _HAS_CPP

def _move_axis_to_last(x: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(x, axis, -1) if axis != -1 else x

def _move_axis_back(x: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(x, -1, axis) if axis != -1 else x

def lowpass(data, fs, fc, *, axis=-1, c=0.9, offset=1.0, d=None,
            initial_d=10.0, d_increment=5.0, threshold=1e-4, max_iter=200, max_d=1e6):
    if not _HAS_CPP:
        return _fallback.lowpass(data, fs, fc, axis=axis, c=c, offset=offset, d=d,
                                 initial_d=initial_d, d_increment=d_increment, threshold=threshold,
                                 max_iter=max_iter, max_d=max_d)

    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = x_m.shape[-1]

    if d is None:
        d = _cpp.auto_d_lowpass(fs, n, fc, c, offset,
                                float(initial_d), float(d_increment), float(threshold),
                                int(max_iter), float(max_d))

    gain = np.ascontiguousarray(_cpp.gain_lowpass(fs, n, fc, c, offset, float(d)), dtype=np.float64)
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _cpp.apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)

def highpass(data, fs, fc, *, axis=-1, c=0.9, offset=1.0, d=None,
             initial_d=10.0, d_increment=5.0, threshold=1e-4, max_iter=200, max_d=1e6):
    if not _HAS_CPP:
        return _fallback.highpass(data, fs, fc, axis=axis, c=c, offset=offset, d=d,
                                  initial_d=initial_d, d_increment=d_increment, threshold=threshold,
                                  max_iter=max_iter, max_d=max_d)

    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = x_m.shape[-1]

    if d is None:
        d = _cpp.auto_d_highpass(fs, n, fc, c, offset,
                                 float(initial_d), float(d_increment), float(threshold),
                                 int(max_iter), float(max_d))

    gain = np.ascontiguousarray(_cpp.gain_highpass(fs, n, fc, c, offset, float(d)), dtype=np.float64)
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _cpp.apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)

def bandpass(data, fs, fc_low, fc_high, *, axis=-1, c=0.9, offset=1.0, d=None,
             initial_d=10.0, d_increment=5.0, threshold=1e-4, max_iter=200, max_d=1e6):
    if not _HAS_CPP:
        return _fallback.bandpass(data, fs, fc_low, fc_high, axis=axis, c=c, offset=offset, d=d,
                                  initial_d=initial_d, d_increment=d_increment, threshold=threshold,
                                  max_iter=max_iter, max_d=max_d)

    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = x_m.shape[-1]

    if d is None:
        d = _cpp.auto_d_bandpass(fs, n, fc_low, fc_high, c, offset,
                                 float(initial_d), float(d_increment), float(threshold),
                                 int(max_iter), float(max_d))

    gain = np.ascontiguousarray(_cpp.gain_bandpass(fs, n, fc_low, fc_high, c, offset, float(d)), dtype=np.float64)
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _cpp.apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)

def filter_signal(data, fs, *, lowcut=None, highcut=None, axis=-1, **kwargs):
    if lowcut is None and highcut is None:
        raise ValueError("Provide at least one of lowcut or highcut")
    if lowcut is not None and highcut is not None:
        return bandpass(data, fs, lowcut, highcut, axis=axis, **kwargs)
    if highcut is not None:
        return lowpass(data, fs, highcut, axis=axis, **kwargs)
    return highpass(data, fs, lowcut, axis=axis, **kwargs)
