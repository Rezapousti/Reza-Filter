from __future__ import annotations

"""
Reza Filter (minimal user-facing API)
------------------------------------
Goal: users should only need:
    import reza
    y = reza.lp(x, fs=200, fc=5)          # low-pass 5 Hz
    y = reza.hp(x, fs=200, fc=10)         # high-pass 10 Hz
    y = reza.bp(x, fs=200, f1=5, f2=10)   # band-pass 5–10 Hz

All shape parameters are internal. The dynamic-decay exponent d is auto-selected
(auto_d) and cached so users never need to tune it.
"""

import numpy as np
from functools import lru_cache

from . import _fallback

try:
    from . import _reza_cpp as _cpp  # compiled extension
    _HAS_CPP = True
except Exception:
    _cpp = None
    _HAS_CPP = False

__version__ = "0.2.1"

__all__ = [
    "lp", "hp", "bp",
    "lowpass", "highpass", "bandpass",
    "filter", "has_cpp", "__version__",
]

# ---------------------------------------------------------------------
# Internal defaults (NOT part of the public API)
# ---------------------------------------------------------------------
_C_DEFAULT = 0.9
_OFFSET_DEFAULT = 1.0

# Dynamic-decay search parameters (kept internal)
_D_INIT = 10.0
_D_INC = 5.0
_D_THRESHOLD = 1e-4
_D_MAX_ITER = 200
_D_MAX = 1e6


def has_cpp() -> bool:
    return _HAS_CPP


def _move_axis_to_last(x: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(x, axis, -1) if axis != -1 else x


def _move_axis_back(x: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(x, -1, axis) if axis != -1 else x


def _apply_gain_rfft(X: np.ndarray, gain: np.ndarray) -> np.ndarray:
    if _HAS_CPP:
        return _cpp.apply_gain_rfft(X, gain)
    return _fallback.apply_gain_rfft(X, gain)


@lru_cache(maxsize=256)
def _auto_d_lowpass(fs: float, n: int, fc: float) -> float:
    if _HAS_CPP:
        return float(
            _cpp.auto_d_lowpass(
                float(fs), int(n), float(fc),
                float(_C_DEFAULT), float(_OFFSET_DEFAULT),
                float(_D_INIT), float(_D_INC), float(_D_THRESHOLD),
                int(_D_MAX_ITER), float(_D_MAX),
            )
        )
    return float(
        _fallback._auto_d_lowpass(
            float(fs), int(n), float(fc),
            c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT),
            initial_d=float(_D_INIT), d_increment=float(_D_INC),
            threshold=float(_D_THRESHOLD), max_iter=int(_D_MAX_ITER), max_d=float(_D_MAX),
        )
    )


@lru_cache(maxsize=256)
def _auto_d_highpass(fs: float, n: int, fc: float) -> float:
    if _HAS_CPP:
        return float(
            _cpp.auto_d_highpass(
                float(fs), int(n), float(fc),
                float(_C_DEFAULT), float(_OFFSET_DEFAULT),
                float(_D_INIT), float(_D_INC), float(_D_THRESHOLD),
                int(_D_MAX_ITER), float(_D_MAX),
            )
        )
    return float(
        _fallback._auto_d_highpass(
            float(fs), int(n), float(fc),
            c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT),
            initial_d=float(_D_INIT), d_increment=float(_D_INC),
            threshold=float(_D_THRESHOLD), max_iter=int(_D_MAX_ITER), max_d=float(_D_MAX),
        )
    )


@lru_cache(maxsize=256)
def _auto_d_bandpass(fs: float, n: int, f1: float, f2: float) -> float:
    if _HAS_CPP:
        return float(
            _cpp.auto_d_bandpass(
                float(fs), int(n), float(f1), float(f2),
                float(_C_DEFAULT), float(_OFFSET_DEFAULT),
                float(_D_INIT), float(_D_INC), float(_D_THRESHOLD),
                int(_D_MAX_ITER), float(_D_MAX),
            )
        )
    return float(
        _fallback._auto_d_bandpass(
            float(fs), int(n), float(f1), float(f2),
            c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT),
            initial_d=float(_D_INIT), d_increment=float(_D_INC),
            threshold=float(_D_THRESHOLD), max_iter=int(_D_MAX_ITER), max_d=float(_D_MAX),
        )
    )


@lru_cache(maxsize=256)
def _gain_lowpass(fs: float, n: int, fc: float) -> np.ndarray:
    d = _auto_d_lowpass(fs, n, fc)
    if _HAS_CPP:
        g = _cpp.gain_lowpass(float(fs), int(n), float(fc),
                              float(_C_DEFAULT), float(_OFFSET_DEFAULT), float(d))
    else:
        g = _fallback.calculate_gain_lowpass(float(fs), int(n), float(fc),
                                             c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT), d=float(d))
    g = np.ascontiguousarray(np.asarray(g, dtype=np.float64))
    g.setflags(write=False)
    return g


@lru_cache(maxsize=256)
def _gain_highpass(fs: float, n: int, fc: float) -> np.ndarray:
    d = _auto_d_highpass(fs, n, fc)
    if _HAS_CPP:
        g = _cpp.gain_highpass(float(fs), int(n), float(fc),
                               float(_C_DEFAULT), float(_OFFSET_DEFAULT), float(d))
    else:
        g = _fallback.calculate_gain_highpass(float(fs), int(n), float(fc),
                                              c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT), d=float(d))
    g = np.ascontiguousarray(np.asarray(g, dtype=np.float64))
    g.setflags(write=False)
    return g


@lru_cache(maxsize=256)
def _gain_bandpass(fs: float, n: int, f1: float, f2: float) -> np.ndarray:
    d = _auto_d_bandpass(fs, n, f1, f2)
    if _HAS_CPP:
        g = _cpp.gain_bandpass(float(fs), int(n), float(f1), float(f2),
                               float(_C_DEFAULT), float(_OFFSET_DEFAULT), float(d))
    else:
        g = _fallback.calculate_gain_bandpass(float(fs), int(n), float(f1), float(f2),
                                              c=float(_C_DEFAULT), offset=float(_OFFSET_DEFAULT), d=float(d))
    g = np.ascontiguousarray(np.asarray(g, dtype=np.float64))
    g.setflags(write=False)
    return g


def lowpass(data, fs: float, fc: float, axis: int = -1):
    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = int(x_m.shape[-1])

    gain = _gain_lowpass(float(fs), n, float(fc))
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)


def highpass(data, fs: float, fc: float, axis: int = -1):
    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = int(x_m.shape[-1])

    gain = _gain_highpass(float(fs), n, float(fc))
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)


def bandpass(data, fs: float, f1: float, f2: float, axis: int = -1):
    if float(f2) <= float(f1):
        raise ValueError("bandpass requires f2 > f1")

    x = np.asarray(data, dtype=float)
    x_m = _move_axis_to_last(x, axis)
    n = int(x_m.shape[-1])

    gain = _gain_bandpass(float(fs), n, float(f1), float(f2))
    X = np.ascontiguousarray(np.fft.rfft(x_m, axis=-1).astype(np.complex128, copy=False))
    Y = _apply_gain_rfft(X, gain)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return _move_axis_back(y, axis)


def filter(data, fs: float, *, lowcut=None, highcut=None, axis: int = -1):
    if lowcut is None and highcut is None:
        raise ValueError("Provide at least one of lowcut or highcut")
    if lowcut is not None and highcut is not None:
        return bandpass(data, fs, lowcut, highcut, axis=axis)
    if highcut is not None:
        return lowpass(data, fs, highcut, axis=axis)
    return highpass(data, fs, lowcut, axis=axis)


# Short aliases (intended for end users)
def lp(data, fs: float, fc: float, axis: int = -1):
    return lowpass(data, fs, fc, axis=axis)


def hp(data, fs: float, fc: float, axis: int = -1):
    return highpass(data, fs, fc, axis=axis)


def bp(data, fs: float, f1: float, f2: float, axis: int = -1):
    return bandpass(data, fs, f1, f2, axis=axis)