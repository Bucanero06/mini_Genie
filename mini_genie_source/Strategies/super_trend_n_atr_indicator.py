#!/usr/bin/env python3
import numpy as np
from numba import njit
import vectorbtpro as vbt


class SuperTrendAIS(vbt.tp.NamedTuple):
    i: int
    high: float
    low: float
    close: float
    prev_close: float
    prev_upper: float
    prev_lower: float
    prev_dir_: float
    nobs: int
    weighted_avg: float
    old_wt: float
    period: int
    multiplier: float


class SuperTrendAOS(vbt.tp.NamedTuple):
    nobs: int
    weighted_avg: float
    old_wt: float
    upper: float
    lower: float
    trend: float
    dir_: float
    long: float
    short: float


@njit(nogil=True)
def get_tr_nb(high, low, prev_close):
    tr0 = abs(high - low)
    tr1 = abs(high - prev_close)
    tr2 = abs(low - prev_close)
    if np.isnan(tr0) or np.isnan(tr1) or np.isnan(tr2):
        tr = np.nan
    else:
        tr = max(tr0, tr1, tr2)
    return tr


@njit(nogil=True)
def get_med_price_nb(high, low):
    return (high + low) / 2


@njit(nogil=True)
def get_basic_bands_nb(high, low, atr, multiplier):
    med_price = get_med_price_nb(high, low)
    matr = multiplier * atr
    upper = med_price + matr
    lower = med_price - matr
    return upper, lower


@njit(nogil=True)
def get_final_bands_nb(close, upper, lower,
                       prev_upper, prev_lower, prev_dir_):
    if close > prev_upper:
        dir_ = 1
    elif close < prev_lower:
        dir_ = -1
    else:
        dir_ = prev_dir_
        if dir_ > 0 and lower < prev_lower:
            lower = prev_lower
        if dir_ < 0 and upper > prev_upper:
            upper = prev_upper

    if dir_ > 0:
        trend = long = lower
        short = np.nan
    else:
        trend = short = upper
        long = np.nan
    return upper, lower, trend, dir_, long, short


@njit(nogil=True)
def get_final_bands_nb(close, upper, lower,
                       prev_upper, prev_lower, prev_dir_):
    if close > prev_upper:
        dir_ = 1
    elif close < prev_lower:
        dir_ = -1
    else:
        dir_ = prev_dir_
        if dir_ > 0 and lower < prev_lower:
            lower = prev_lower
        if dir_ < 0 and upper > prev_upper:
            upper = prev_upper

    if dir_ > 0:
        trend = long = lower
        short = np.nan
    else:
        trend = short = upper
        long = np.nan
    return upper, lower, trend, dir_, long, short


@njit(nogil=True)
def superfast_supertrend_acc_nb(in_state):
    i = in_state.i
    high = in_state.high
    low = in_state.low
    close = in_state.close
    prev_close = in_state.prev_close
    prev_upper = in_state.prev_upper
    prev_lower = in_state.prev_lower
    prev_dir_ = in_state.prev_dir_
    nobs = in_state.nobs
    weighted_avg = in_state.weighted_avg
    old_wt = in_state.old_wt
    period = in_state.period
    multiplier = in_state.multiplier

    tr = get_tr_nb(high, low, prev_close)

    alpha = vbt.nb.alpha_from_wilder_nb(period)
    ewm_mean_in_state = vbt.nb.EWMMeanAIS(
        i=i,
        value=tr,
        old_wt=old_wt,
        weighted_avg=weighted_avg,
        nobs=nobs,
        alpha=alpha,
        minp=period,
        adjust=False
    )
    ewm_mean_out_state = vbt.nb.ewm_mean_acc_nb(ewm_mean_in_state)
    atr = ewm_mean_out_state.value

    upper, lower = get_basic_bands_nb(high, low, atr, multiplier)

    if i == 0:
        trend, dir_, long, short = np.nan, 1, np.nan, np.nan
    else:
        upper, lower, trend, dir_, long, short = get_final_bands_nb(
            close, upper, lower, prev_upper, prev_lower, prev_dir_)

    return SuperTrendAOS(
        nobs=ewm_mean_out_state.nobs,
        weighted_avg=ewm_mean_out_state.weighted_avg,
        old_wt=ewm_mean_out_state.old_wt,
        upper=upper,
        lower=lower,
        trend=trend,
        dir_=dir_,
        long=long,
        short=short
    )


@njit(nogil=True)
def superfast_supertrend_nb(high, low, close, period=7, multiplier=3):
    trend = np.empty(close.shape, dtype=np.float_)
    dir_ = np.empty(close.shape, dtype=np.int_)
    long = np.empty(close.shape, dtype=np.float_)
    short = np.empty(close.shape, dtype=np.float_)

    if close.shape[0] == 0:
        return trend, dir_, long, short

    nobs = 0
    old_wt = 1.
    weighted_avg = np.nan
    prev_upper = np.nan
    prev_lower = np.nan

    for i in range(close.shape[0]):
        in_state = SuperTrendAIS(
            i=i,
            high=high[i],
            low=low[i],
            close=close[i],
            prev_close=close[i - 1] if i > 0 else np.nan,
            prev_upper=prev_upper,
            prev_lower=prev_lower,
            prev_dir_=dir_[i - 1] if i > 0 else 1,
            nobs=nobs,
            weighted_avg=weighted_avg,
            old_wt=old_wt,
            period=period,
            multiplier=multiplier
        )

        out_state = superfast_supertrend_acc_nb(in_state)

        nobs = out_state.nobs
        weighted_avg = out_state.weighted_avg
        old_wt = out_state.old_wt
        prev_upper = out_state.upper
        prev_lower = out_state.lower
        trend[i] = out_state.trend
        dir_[i] = out_state.dir_
        long[i] = out_state.long
        short[i] = out_state.short

    return trend, dir_, long, short


def apply_function(low, high, close, period_windows, multiplier_windows):
    return superfast_supertrend_nb(high, low, close, period=period_windows, multiplier=multiplier_windows)


def SuperTrendFastAF(open_data, low_data, high_data, close_data, parameter_data):
    period_windows = np.array(parameter_data["period_windows"])
    multiplier_windows = np.array(parameter_data["multiplier_windows"])

    SuperTrend = vbt.IF(
        input_names=['high_data', 'low_data', 'close_data'],
        param_names=['period_windows', 'multiplier_windows'],
        output_names=['supert', 'superd', 'superl', 'supers']
    ).with_apply_func(
        superfast_supertrend_nb,
        takes_1d=True,
        jit_select_params=True,
        jit_kwargs=dict(nogil=True),
        execute_kwargs=dict(
            engine='dask',
            chunk_len='auto',
            show_progress=True
        ),
        period_windows=7,
        multiplier_windows=3
    )

    st = SuperTrend.run(
        high_data, low_data, close_data,
        period_windows=period_windows,
        multiplier_windows=multiplier_windows,
        param_product=False,
    )
    long_entries = (~st.superl.isnull()).vbt.signals.fshift()
    long_exits = (~st.supers.isnull()).vbt.signals.fshift()
    short_entries = (~st.superl.isnull()).vbt.signals.fshift()
    short_exits = (~st.supers.isnull()).vbt.signals.fshift()
    import pandas as pd

    strategy_specific_kwargs = dict(
        allow_multiple_trade_from_entries=False,  # strategy_specific_kwargs['allow_multiple_trade_from_entries'],
        exit_on_opposite_direction_entry=True,  # strategy_specific_kwargs['exit_on_opposite_direction_entry'],
        #
        progressive_bool=False,  # Master_Indicator.progressive_bool,
        long_progressive_condition=False,  # Master_Indicator.long_entry_condition_3.vbt.signals.fshift(),
        short_progressive_condition=False,  # Master_Indicator.short_entry_condition_3.vbt.signals.fshift(),
        #
        breakeven_1_trigger_bool=False,  # Master_Indicator.breakeven_1_trigger_bool,
        breakeven_1_trigger_points=0,  # Master_Indicator.breakeven_1_trigger_points,
        breakeven_1_distance_points=0,  # Master_Indicator.breakeven_1_distance_points,
        #
        breakeven_2_trigger_bool=False,  # Master_Indicator.breakeven_2_trigger_bool,
        breakeven_2_trigger_points=0,  # Master_Indicator.breakeven_2_trigger_points,
        breakeven_2_distance_points=0,  # Master_Indicator.breakeven_2_distance_points,
        #
        take_profit_bool=False,  # Master_Indicator.take_profit_bool,
        take_profit_points=0,
        take_profit_point_parameters=0,
        #
        stoploss_bool=False,  # Master_Indicator.stoploss_bool,
        stoploss_points=0,
        stoploss_points_parameters=0,
    )
    # print(long_entries.head())

    return long_entries, long_exits, short_entries, short_exits, strategy_specific_kwargs
