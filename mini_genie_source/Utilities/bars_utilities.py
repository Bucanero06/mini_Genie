import numpy as np
from numba import prange
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from vectorbtpro import _typing as tp


def BARSINCE_genie(df):
    def intervaled_cumsum_(a, trigger_val=1, start_val=0, invalid_specifier=-1):
        out = np.ones(a.size, dtype=int)
        idx = np.flatnonzero(a == trigger_val)
        if len(idx) == 0:
            return np.full(a.size, invalid_specifier)
        else:
            out[idx[0]] = -idx[0] + 1
            out[0] = start_val
            out[idx[1:]] = idx[:-1] - idx[1:] + 1
            np.cumsum(out, out=out)
            out[:idx[0]] = invalid_specifier
            return out

    return df.apply(intervaled_cumsum_).replace(-1, np.nan)


def max_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0, fillna).dtype
    hW = (W - 1) // 2  # Half window size
    out = maximum_filter1d(a, size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W - 1] = fillna
    else:
        out = np.concatenate((np.full(W - 1, fillna), out[W - 1:]))
    return out


def ROLLING_MAX_genie(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `max_filter1d_same`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = max_filter1d_same(arr[:, col], window)
    return out


def min_filter1d_same(a, W, fillna=np.nan):
    out_dtype = np.full(0, fillna).dtype
    hW = (W - 1) // 2  # Half window size
    out = minimum_filter1d(a, size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W - 1] = fillna
    else:
        out = np.concatenate((np.full(W - 1, fillna), out[W - 1:]))
    return out


def ROLLING_MIN_genie(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `max_filter1d_same`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = min_filter1d_same(arr[:, col], window)
    return out
