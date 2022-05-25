import numpy as np
import pandas as pd
import vectorbtpro as vbt
from numba import prange
from scipy.ndimage.filters import maximum_filter1d, minimum_filter1d
from vectorbtpro import _typing as tp


def BARSINCE_genie(df: object) -> object:
    """

    Args:
        df:

    Returns:
        object:

    """
    def intervaled_cumsum_(a: object, trigger_val: object = 1, start_val: object = 0, invalid_specifier: object = -1) -> object:
        """

        Args:
            a:
            trigger_val:
            start_val:
            invalid_specifier:

        Returns:
            object:

        """
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


def max_filter1d_same(a: object, W: object, fillna: object = np.nan) -> object:
    """

    Args:
        a:
        W:
        fillna:

    Returns:
        object:

    """
    out_dtype = np.full(0, fillna).dtype
    hW = (W - 1) // 2  # Half window size
    out = maximum_filter1d(a, size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W - 1] = fillna
    else:
        out = np.concatenate((np.full(W - 1, fillna), out[W - 1:]))
    return out


def ROLLING_MAX_genie(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `max_filter1d_same`.

    Args:
        arr: 
        window:
        minp:

    Returns:
        object:
    """
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = max_filter1d_same(arr[:, col], window)
    return out


def min_filter1d_same(a: object, W: object, fillna: object = np.nan) -> object:
    """

    Args:
        a:
        W:
        fillna:

    Returns:
        object:

    """
    out_dtype = np.full(0, fillna).dtype
    hW = (W - 1) // 2  # Half window size
    out = minimum_filter1d(a, size=W, origin=hW)
    if out.dtype is out_dtype:
        out[:W - 1] = fillna
    else:
        out = np.concatenate((np.full(W - 1, fillna), out[W - 1:]))
    return out


def ROLLING_MIN_genie(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `max_filter1d_same`.

    Args:
        arr: 
        window: 
        minp: 

    Returns:
        object: 
    """
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = min_filter1d_same(arr[:, col], window)
    return out


def resample_olhc_genie(timeframes: object, open_data: object = pd.DataFrame(), low_data: object = pd.DataFrame(),
                        high_data: object = pd.DataFrame(),
                        close_data: object = pd.DataFrame()) -> object:
    """

    Args:
        timeframes: 
        open_data: 
        low_data: 
        high_data: 
        close_data: 

    Returns:
        object: 

    """
    resampled_data_dict = {
        # Data
        'Open': {},
        'Low': {},
        'High': {},
        'Close': {},
    }
    for timeframe in timeframes:
        # '''Pre-Resample Data'''
        # OPEN
        if not open_data.empty:
            resampled_data_dict['Open'][timeframe] = open_data.vbt.resample_opening(
                timeframe) if timeframe != '1m' else open_data
        # LOW
        if not low_data.empty:
            resampled_data_dict['Low'][timeframe] = low_data.vbt.resample_apply(timeframe,
                                                                                vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else low_data
        # HIGH
        if not high_data.empty:
            resampled_data_dict['High'][timeframe] = high_data.vbt.resample_apply(timeframe,
                                                                                  vbt.nb.max_reduce_nb).dropna() if timeframe != '1 min' else high_data
        # CLOSE
        if not close_data.empty:
            resampled_data_dict['Close'][timeframe] = close_data.vbt.resample_apply(timeframe,
                                                                                    vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close_data
    return resampled_data_dict


def split_uptrend_n_downtrend_atr(atr_df: object, dir_df: object) -> object:
    """

    Args:
        atr_df: 
        dir_df: 

    Returns:
        object: 

    """
    atr_uptrend_mask = dir_df.values == 1
    atr_downtrend_mask = dir_df.values == -1
    return atr_df.loc[atr_uptrend_mask], atr_df.loc[atr_downtrend_mask]
# # # Change timeframes
#     # resampled_data_dict = resample_olhc_genie(timeframes, low=low_data, high=high_data, close=close_data)
#     # super_trend_dict = {}
#     # # do loop
#     # for tf in timeframes:
#     #     super_trend_dict[tf] = superfast_supertrend_nb(high_data, low_data, close_data, period=period_windows,
#     #                                                    multiplier=multiplier_windows)
#
#     # return super_trend_dict

# @jit(nopython=False)
# def cache_func(open_data, low_data, high_data, close_data, timeframes='1 min', period=7, multiplier=3):
#     return resample_olhc_genie(timeframes, open_data, low_data, high_data, close_data)

# # cache_func=cache_func,
#         # pass_wrapper=True,

# wrapper = vbt.ArrayWrapper.from_obj(df)
#     df = wrapper.wrap_reduced(result)
