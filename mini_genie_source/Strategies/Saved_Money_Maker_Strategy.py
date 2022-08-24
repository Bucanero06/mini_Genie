#!/usr/bin/env python3
# --- ↓ Do not remove these libs ↓ -------------------------------------------------------------------------------------

import gc
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from Utilities.bars_utilities import BARSINCE_genie, ROLLING_MAX_genie, ROLLING_MIN_genie


# --- ↑ Do not remove these libs ↑ -------------------------------------------------------------------------------------


def cache_func(low, high, close,
               PEAK_and_ATR_timeframes, atr_windows, data_lookback_windows,
               EMAs_timeframes, ema_1_windows, ema_2_windows,
               take_profit_points, stop_loss_points):
    cache = {
        # Data
        'Low': {},
        'High': {},
        'Close': {},
        # Indicators
        'ATR': {},
        'EMA': {},
        # Resampler
        'Resamplers': {},
        # empty_df_like
        'Empty_df_like': pd.DataFrame().reindex_like(close),
    }

    # Create a set of all timeframes to resample data to
    timeframes = tuple(set(tuple(PEAK_and_ATR_timeframes) + tuple(EMAs_timeframes)))
    #
    '''Pre-Resample Data'''
    #
    for timeframe in timeframes:
        # LOW
        cache['Low'][timeframe] = low.vbt.resample_apply(timeframe,
                                                         vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else low
        # HIGH
        cache['High'][timeframe] = high.vbt.resample_apply(timeframe,
                                                           vbt.nb.max_reduce_nb).dropna() if timeframe != '1 min' else high
        # CLOSE
        cache['Close'][timeframe] = close.vbt.resample_apply(timeframe,
                                                             vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close

        '''Pre-Prepare Resampler'''
        cache['Resamplers'][timeframe] = vbt.Resampler(
            cache['Close'][timeframe].index,
            close.index,
            source_freq=timeframe,
            target_freq="1m") if timeframe != '1 min' else None

    '''Pre-Compute Indicators'''
    #
    # for PEAK_and_ATR_timeframe in PEAK_and_ATR_timeframes:
    #
    #     for atr_window in atr_windows:
    #         cache['ATR'][f'{PEAK_and_ATR_timeframe}_{atr_window}'] = vbt.indicators.ATR.run(
    #             cache['High'][PEAK_and_ATR_timeframe],
    #             cache['Low'][PEAK_and_ATR_timeframe],
    #             cache['Close'][PEAK_and_ATR_timeframe],
    #             window=atr_window,
    #             ewm=False,
    #             short_name='atr').atr

    for PEAK_and_ATR_timeframe, atr_window in zip(PEAK_and_ATR_timeframes, atr_windows):
        cache['ATR'][f'{PEAK_and_ATR_timeframe}_{atr_window}'] = vbt.indicators.ATR.run(
            cache['High'][PEAK_and_ATR_timeframe],
            cache['Low'][PEAK_and_ATR_timeframe],
            cache['Close'][PEAK_and_ATR_timeframe],
            window=atr_window,
            ewm=False,
            short_name='atr').atr
    #
    for _ema_timeframe, _ema_window_1, _ema_window_2 in zip(EMAs_timeframes, ema_1_windows, ema_2_windows):
        if f'{_ema_timeframe}_{_ema_window_1}' not in cache["EMA"]:
            cache['EMA'][f'{_ema_timeframe}_{_ema_window_1}'] = vbt.MA.run(cache['Close'][_ema_timeframe],
                                                                           window=_ema_window_1,
                                                                           ewm=True).ma
        #
        if f'{_ema_timeframe}_{_ema_window_2}' not in cache["EMA"]:
            cache['EMA'][f'{_ema_timeframe}_{_ema_window_2}'] = vbt.MA.run(cache['Close'][_ema_timeframe],
                                                                           window=_ema_window_2,
                                                                           ewm=True).ma

    gc.collect()
    return cache


def apply_function(low_data, high_data, close_data,
                   PEAK_and_ATR_timeframe, atr_window, data_lookback_window,
                   EMAs_timeframe, ema_1_window, ema_2_window,
                   take_profit_points,
                   stop_loss_points,
                   cache
                   ):
    """Function for Indicators"""

    '''Fetch Resampled Data'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Only fetch those that are needed from the cached dict
    PEAK_and_ATR_timeframe_low = cache['Low'][PEAK_and_ATR_timeframe]
    PEAK_and_ATR_timeframe_high = cache['High'][PEAK_and_ATR_timeframe]
    PEAK_and_ATR_timeframe_close = cache['Close'][PEAK_and_ATR_timeframe]
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''ATR Indicator'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch pre-computed atr from cache. Uses PEAK_and_ATR_timeframe
    atr_indicator = cache['ATR'][f'{PEAK_and_ATR_timeframe}_{atr_window}']
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''PeakHigh and PeakLow'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # All indicators and datas in this section use the PEAK_and_ATR_timeframe
    #
    # Compute the rolling max of the high_data using a window of size data_lookback_window "highest(3,high)"
    rolling_max = ROLLING_MAX_genie(PEAK_and_ATR_timeframe_high.to_numpy(), data_lookback_window)
    # Compare where the high_data is the same as the rolling_max "high == highest(3,high)"
    high_eq_highest_in_N = PEAK_and_ATR_timeframe_high == rolling_max
    # Find where the diff b/w the high_data and close_data is bigger than the ATR "( high - close > atr(AtrPeriod) )"
    high_minus_close_gt_atr = (PEAK_and_ATR_timeframe_high.to_numpy() - PEAK_and_ATR_timeframe_close.to_numpy()) \
                              > atr_indicator
    # Compute the PeakHigh "( high == highest(3,high) and ( high - close > atr(AtrPeriod) )"
    PeakHigh = (high_eq_highest_in_N) & (high_minus_close_gt_atr)

    # Compute the rolling min of the low_data using a window of size data_lookback_window "lowest(3,low)"
    rolling_min = ROLLING_MIN_genie(PEAK_and_ATR_timeframe_low.to_numpy(), data_lookback_window)
    # Compare where the low_data is the same as the rolling_min "low == lowest(3,low)"
    low_eq_lowest_in_N = PEAK_and_ATR_timeframe_low == rolling_min
    # Find where the diff b/w the close_data and low_data is bigger than the ATR "( close - low > atr(AtrPeriod) ) "
    close_minus_low_bt_atr = (PEAK_and_ATR_timeframe_close.to_numpy() - PEAK_and_ATR_timeframe_low.to_numpy()) \
                             > atr_indicator
    # Compute the PeakLow "( low == lowest(3,low) and ( close - low > atr(AtrPeriod) )  "
    PeakLow = (low_eq_lowest_in_N) & (close_minus_low_bt_atr)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''EMA Indicators'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch pre-computed ema from cache. Uses EMAs_timeframe
    ema_1_indicator = cache['EMA'][f'{EMAs_timeframe}_{ema_1_window}']
    ema_2_indicator = cache['EMA'][f'{EMAs_timeframe}_{ema_2_window}']
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Resample Indicators Back To 1 minute'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch the resamplers from cache for a given timeframe
    PEAK_and_ATR_timeframe_to_1min_Resampler = cache['Resamplers'][PEAK_and_ATR_timeframe]
    EMAs_timeframe_to_1min_Resampler = cache['Resamplers'][EMAs_timeframe]

    # Resample indicators to 1m
    atr_indicator = atr_indicator.vbt.resample_closing(
        PEAK_and_ATR_timeframe_to_1min_Resampler) if PEAK_and_ATR_timeframe_to_1min_Resampler else atr_indicator
    PeakHigh = PeakHigh.vbt.resample_closing(
        PEAK_and_ATR_timeframe_to_1min_Resampler) if PEAK_and_ATR_timeframe_to_1min_Resampler else PeakHigh
    PeakLow = PeakLow.vbt.resample_closing(
        PEAK_and_ATR_timeframe_to_1min_Resampler) if PEAK_and_ATR_timeframe_to_1min_Resampler else PeakLow
    ema_1_indicator = ema_1_indicator.vbt.resample_closing(
        EMAs_timeframe_to_1min_Resampler) if EMAs_timeframe_to_1min_Resampler else ema_1_indicator
    ema_2_indicator = ema_2_indicator.vbt.resample_closing(
        EMAs_timeframe_to_1min_Resampler) if EMAs_timeframe_to_1min_Resampler else ema_2_indicator
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Long Entries Conditions'''
    # Bars since last PeakLow are less than Bars PeakHigh "barssince(PeakLow) < barssince(PeakHigh)"
    long_entry_condition_1 = BARSINCE_genie(PeakLow).lt(BARSINCE_genie(PeakHigh))
    # EMA 1 crosses above EMA 2 "crossover(ema_EmaTF(13) , ema_EmaTF(50) )"
    long_entry_condition_2 = ema_1_indicator.vbt.crossed_above(ema_2_indicator)

    '''Short Entries Conditions'''
    # Bars since last PeakLow are greater than Bars PeakHigh "barssince(PeakLow) > barssince(PeakHigh)"
    short_entry_condition_1 = BARSINCE_genie(PeakLow).gt(BARSINCE_genie(PeakHigh))
    # EMA 1 crosses below EMA 2 "crossunder(ema_EmaTF(13) , ema_EmaTF(50) )"
    short_entry_condition_2 = ema_1_indicator.vbt.crossed_below(ema_2_indicator)

    '''Fill Rest of Parameters for Sim'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Used to fill signals and parameter dfs into the correct size (just a workaround for now, fast)
    empty_df_like = cache['Empty_df_like']

    stop_loss_points = empty_df_like.fillna(stop_loss_points)
    take_profit_points = empty_df_like.fillna(take_profit_points)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Define Entries and Exits Signals'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    long_entries = (
            long_entry_condition_1
            & long_entry_condition_2.to_numpy()
    ).vbt.signals.fshift()
    long_exits = pd.DataFrame().reindex_like(long_entries).fillna(False)

    short_entries = (
            short_entry_condition_1
            & short_entry_condition_2.to_numpy()
    ).vbt.signals.fshift()
    short_exits = long_exits
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    return long_entries, long_exits, short_entries, short_exits, take_profit_points, stop_loss_points


def MMT_Strategy(open_data, low_data, high_data, close_data, parameter_data, ray_sim_n_cpus):
    """MMT_Strategy"""
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    PEAK_and_ATR_timeframes = np.array(parameter_data["PEAK_and_ATR_timeframes"])
    atr_windows = np.array(parameter_data["atr_windows"])
    data_lookback_windows = np.array(parameter_data["data_lookback_windows"])
    #
    EMAs_timeframes = np.array(parameter_data["EMAs_timeframes"])
    ema_1_windows = np.array(parameter_data["ema_1_windows"])
    ema_2_windows = np.array(parameter_data["ema_2_windows"])
    #
    take_profit_points = np.array(parameter_data["take_profit_points"])
    #
    stop_loss_points = np.array(parameter_data["stop_loss_points"])
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #
    '''Compile Structure and Run Master Indicator'''
    Master_Indicator = vbt.IF(
        input_names=[
            'low_data', 'high_data', 'close_data',
        ],
        param_names=['PEAK_and_ATR_timeframes', 'atr_windows', 'data_lookback_windows',
                     'EMAs_timeframes', 'ema_1_windows', 'ema_2_windows',
                     'take_profit_points',
                     'stop_loss_points'
                     ],
        output_names=[
            'long_entries', 'long_exits', 'short_entries', 'short_exits',
            'take_profit_points', 'stop_loss_points'
        ]
    ).with_apply_func(
        apply_func=apply_function,
        cache_func=cache_func,
        keep_pd=True,
        param_product=False,
        execute_kwargs=dict(
            engine='ray',
            init_kwargs={
                'address': 'auto',
                'num_cpus': ray_sim_n_cpus,
            },
            show_progress=True
        ),
        PEAK_and_ATR_timeframes='1d',
        atr_windows=5,
        data_lookback_windows=3,
        EMAs_timeframes='1h',
        ema_1_windows=13,
        ema_2_windows=50,
        take_profit_points=300,
        stop_loss_points=-600,
    ).run(
        low_data, high_data, close_data,
        PEAK_and_ATR_timeframes=PEAK_and_ATR_timeframes,
        atr_windows=atr_windows,
        data_lookback_windows=data_lookback_windows,
        EMAs_timeframes=EMAs_timeframes,
        ema_1_windows=ema_1_windows,
        ema_2_windows=ema_2_windows,
        take_profit_points=take_profit_points,
        stop_loss_points=stop_loss_points,
        # ##############################################
    )

    '''Type C conditions'''
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
        take_profit_bool=True,  # Master_Indicator.take_profit_bool,
        take_profit_points=Master_Indicator.take_profit_points,
        take_profit_point_parameters=take_profit_points,
        #
        stop_loss_bool=True,  # Master_Indicator.stop_loss_bool,
        stop_loss_points=Master_Indicator.stop_loss_points,
        stop_loss_points_parameters=stop_loss_points,
    )

    # strategy_specific_kwargs = dict()
    return Master_Indicator.long_entries, Master_Indicator.long_exits, \
           Master_Indicator.short_entries, Master_Indicator.short_exits, \
           strategy_specific_kwargs