import gc
from time import perf_counter

import pandas as pd
import vectorbtpro as vbt
from logger_tt import logger

from Utilities.bars_utilities import BARSINCE_genie

def cache_func(open_data, low_data, high_data, close_data,
               rsi_timeframes, rsi_windows,
               sma_on_rsi_1_windows, sma_on_rsi_2_windows, sma_on_rsi_3_windows,
               T1_ema_timeframes, T1_ema_1_windows, T1_ema_2_windows,
               # T2_ema_timeframes, T2_ema_1_windows, T2_ema_2_windows,
               #
               take_profit_points, stop_loss_points
               ):
    #
    cache = {
        # Data
        'Open': {},
        'Low': {},
        'High': {},
        'Close': {},
        # Indicators
        'RSI': {},
        'SMA': {},
        'EMA': {},
        # Resampler
        'Resamplers': {},
        # empty_df_like
        'Empty_df_like': pd.DataFrame().reindex_like(close_data),
    }
    #
    '''Pre-Resample Data'''
    # Create a set of all timeframes to resample data to and sets of windows
    timeframes_set = tuple(set(tuple(rsi_timeframes) + tuple(T1_ema_timeframes)))  # + tuple(T2_ema_timeframes)))
    for timeframe in timeframes_set:
        # LOW
        cache['Low'][timeframe] = low_data.vbt.resample_apply(timeframe,
                                                              vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else low_data
        # HIGH
        cache['High'][timeframe] = high_data.vbt.resample_apply(timeframe,
                                                                vbt.nb.max_reduce_nb).dropna() if timeframe != '1 min' else high_data
        # CLOSE
        cache['Close'][timeframe] = close_data.vbt.resample_apply(timeframe,
                                                                  vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close_data

        '''Pre-Prepare Resampler'''
        cache['Resamplers'][timeframe] = vbt.Resampler(
            cache['Close'][timeframe].index,
            close_data.index,
            source_freq=timeframe,
            target_freq="1m") if timeframe != '1 min' else None
    #
    '''Pre-Compute Indicators'''
    for _rsi_timeframe, _rsi_window, _sma_window_1, _sma_window_2, _sma_window_3 in zip(rsi_timeframes, rsi_windows,
                                                                                        sma_on_rsi_1_windows,
                                                                                        sma_on_rsi_2_windows,
                                                                                        sma_on_rsi_3_windows):
        #
        if f'{_rsi_timeframe}_{_rsi_window}' not in cache["RSI"]:
            cache['RSI'][f'{_rsi_timeframe}_{_rsi_window}'] = vbt.RSI.run(cache['Close'][_rsi_timeframe],
                                                                          window=_rsi_window,
                                                                          ewm=False).rsi
            #
        if f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_1}' not in cache["SMA"]:
            cache["SMA"][f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_1}'] = vbt.MA.run(
                cache['RSI'][f'{_rsi_timeframe}_{_rsi_window}'], window=_sma_window_1, ewm=False).ma
            #
        if f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_2}' not in cache["SMA"]:
            cache["SMA"][f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_2}'] = vbt.MA.run(
                cache['RSI'][f'{_rsi_timeframe}_{_rsi_window}'], window=_sma_window_2, ewm=False).ma
            #
        if f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_3}' not in cache["SMA"]:
            cache["SMA"][f'{_rsi_timeframe}_{_rsi_window}_{_sma_window_3}'] = vbt.MA.run(
                cache['RSI'][f'{_rsi_timeframe}_{_rsi_window}'], window=_sma_window_3, ewm=False).ma
    #
    '''Create a set of rsi_windows and compute indicator for all timeframes and windows'''
    for _ema_timeframe, _ema_window_1, _ema_window_2 in zip(T1_ema_timeframes, T1_ema_1_windows, T1_ema_2_windows):
        if f'{_ema_timeframe}_{_ema_window_1}' not in cache["EMA"]:
            cache['EMA'][f'{_ema_timeframe}_{_ema_window_1}'] = vbt.MA.run(close_data, window=_ema_window_1,
                                                                           ewm=True).ma
        #
        if f'{_ema_timeframe}_{_ema_window_2}' not in cache["EMA"]:
            cache['EMA'][f'{_ema_timeframe}_{_ema_window_2}'] = vbt.MA.run(close_data, window=_ema_window_2,
                                                                           ewm=True).ma
    #
    gc.collect()
    return cache


def apply_function(open_data, low_data, high_data, close_data,
                   rsi_timeframe, rsi_window,
                   sma_on_rsi_1_window, sma_on_rsi_2_window, sma_on_rsi_3_window,
                   T1_ema_timeframe, T1_ema_1_window, T1_ema_2_window,
                   # T2_ema_timeframe, T2_ema_1_window, T2_ema_2_window,
                   #
                   take_profit_points, stop_loss_points,
                   cache):
    """Function for Indicators"""

    '''RSI and SMA Indicators'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    rsi_indicator = cache['RSI'][f'{rsi_timeframe}_{rsi_window}']
    sma_on_rsi_1_indicator = cache['SMA'][f'{rsi_timeframe}_{rsi_window}_{sma_on_rsi_1_window}']
    sma_on_rsi_2_indicator = cache['SMA'][f'{rsi_timeframe}_{rsi_window}_{sma_on_rsi_2_window}']
    sma_on_rsi_3_indicator = cache['SMA'][f'{rsi_timeframe}_{rsi_window}_{sma_on_rsi_3_window}']
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Trend I EMA Indicators'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    T1_ema_1_indicator = cache['EMA'][f'{T1_ema_timeframe}_{T1_ema_1_window}']
    T1_ema_2_indicator = cache['EMA'][f'{T1_ema_timeframe}_{T1_ema_2_window}']
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # '''Trend II EMA Indicators'''
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # T2_ema_1_indicator = cache['EMA'][f'{T2_ema_timeframe}_{T2_ema_1_window}']
    # T2_ema_2_indicator = cache['EMA'][f'{T2_ema_timeframe}_{T2_ema_2_window}']
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Resample Indicators Back To 1 minute'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch the resamplers from cache for a given timeframe
    rsi_timeframe_to_1min_Resampler = cache['Resamplers'][rsi_timeframe]
    T1_ema_timeframe_to_1min_Resampler = cache['Resamplers'][T1_ema_timeframe]
    # T2_ema_timeframe_to_1min_Resampler = cache['Resamplers'][T2_ema_timeframe]

    # Resample indicators to 1m
    rsi_indicator = rsi_indicator.vbt.resample_closing(
        rsi_timeframe_to_1min_Resampler) if rsi_timeframe_to_1min_Resampler else rsi_indicator
    sma_on_rsi_1_indicator = sma_on_rsi_1_indicator.vbt.resample_closing(
        rsi_timeframe_to_1min_Resampler) if rsi_timeframe_to_1min_Resampler else sma_on_rsi_1_indicator
    sma_on_rsi_2_indicator = sma_on_rsi_2_indicator.vbt.resample_closing(
        rsi_timeframe_to_1min_Resampler) if rsi_timeframe_to_1min_Resampler else sma_on_rsi_2_indicator
    sma_on_rsi_3_indicator = sma_on_rsi_3_indicator.vbt.resample_closing(
        rsi_timeframe_to_1min_Resampler) if rsi_timeframe_to_1min_Resampler else sma_on_rsi_3_indicator
    #
    T1_ema_1_indicator = T1_ema_1_indicator.vbt.resample_closing(
        T1_ema_timeframe_to_1min_Resampler) if T1_ema_timeframe_to_1min_Resampler else T1_ema_1_indicator
    T1_ema_2_indicator = T1_ema_2_indicator.vbt.resample_closing(
        T1_ema_timeframe_to_1min_Resampler) if T1_ema_timeframe_to_1min_Resampler else T1_ema_2_indicator
    # T2_ema_1_indicator = T2_ema_1_indicator.vbt.resample_closing(
    #     T2_ema_timeframe_to_1min_Resampler) if T2_ema_timeframe_to_1min_Resampler else T2_ema_1_indicator
    # T2_ema_2_indicator = T2_ema_2_indicator.vbt.resample_closing(
    #     T2_ema_timeframe_to_1min_Resampler) if T2_ema_timeframe_to_1min_Resampler else T2_ema_2_indicator
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Long Entries Conditions'''
    #    1. TrendType == 'T1' or (ema_4h(13) > ema_4h(50))
    ...  # fixme not adding this right now
    #    2. crossover( ema(13) , ema(50) )
    long_entry_condition_2 = T1_ema_1_indicator.vbt.crossed_above(T1_ema_2_indicator)

    #    3. barssince( crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) ) <
    #       barssince( crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) )
    inside_long_condition_3a1 = sma_on_rsi_1_indicator.vbt.crossed_above(sma_on_rsi_3_indicator)
    inside_long_condition_3a2 = sma_on_rsi_1_indicator.vbt.crossed_above(sma_on_rsi_2_indicator)
    inside_long_condition_3a = inside_long_condition_3a1 | inside_long_condition_3a2.to_numpy()
    #
    inside_long_condition_3b1 = sma_on_rsi_1_indicator.vbt.crossed_below(sma_on_rsi_3_indicator)
    inside_long_condition_3b2 = sma_on_rsi_1_indicator.vbt.crossed_below(sma_on_rsi_2_indicator)
    inside_long_condition_3b = inside_long_condition_3b1 | inside_long_condition_3b2.to_numpy()
    #
    long_entry_condition_3 = BARSINCE_genie(inside_long_condition_3a).lt(BARSINCE_genie(inside_long_condition_3b))

    '''Short Entries Conditions'''
    #    1. TrendType == 'T1' or (ema_4h(13) < ema_4h(50))
    ...  # fixme not adding this right now
    #    2. crossunder( ema(13) , ema(50) )
    short_entry_condition_2 = T1_ema_1_indicator.vbt.crossed_below(T1_ema_2_indicator)

    #    3. barssince( crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) ) >
    #       barssince( crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) )
    inside_short_condition_3a1 = sma_on_rsi_1_indicator.vbt.crossed_above(sma_on_rsi_3_indicator)
    inside_short_condition_3a2 = sma_on_rsi_1_indicator.vbt.crossed_above(sma_on_rsi_2_indicator)
    inside_short_condition_3a = inside_short_condition_3a1 | inside_short_condition_3a2.to_numpy()
    #
    inside_short_condition_3b1 = sma_on_rsi_1_indicator.vbt.crossed_below(sma_on_rsi_3_indicator)
    inside_short_condition_3b2 = sma_on_rsi_1_indicator.vbt.crossed_below(sma_on_rsi_2_indicator)
    inside_short_condition_3b = inside_short_condition_3b1 | inside_short_condition_3b2.to_numpy()
    #
    short_entry_condition_3 = BARSINCE_genie(inside_short_condition_3a).gt(BARSINCE_genie(inside_short_condition_3b))

    '''Fill Rest of Parameters for Sim'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Used to fill signals and parameter dfs into the correct size (just a workaround for now, fast)
    empty_df_like = cache['Empty_df_like']
    #
    take_profit_points = empty_df_like.fillna(take_profit_points)
    stop_loss_points = empty_df_like.fillna(stop_loss_points)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Define Entries and Exits Signals'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    long_entries = (
            long_entry_condition_2
            & long_entry_condition_3.to_numpy()
    ).vbt.signals.fshift()
    long_exits = pd.DataFrame().reindex_like(long_entries).fillna(False)

    short_entries = (
            short_entry_condition_2
            & short_entry_condition_3.to_numpy()
    ).vbt.signals.fshift()
    short_exits = long_exits
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    gc.collect()
    return long_entries, long_exits, short_entries, short_exits, \
           take_profit_points, stop_loss_points

    # T2_ema_1_indicator, T2_ema_2_indicator, \


def RLGL_Strategy(open_data, low_data, high_data, close_data, parameter_data, ray_sim_n_cpus):
    """Red Light Geen Light Strategy"""

    '''RSI and SMA Information'''
    rsi_timeframes = parameter_data["rsi_timeframes"]
    rsi_windows = parameter_data["rsi_windows"]
    #
    sma_on_rsi_1_windows = parameter_data["sma_on_rsi_1_windows"]
    sma_on_rsi_2_windows = parameter_data["sma_on_rsi_2_windows"]
    sma_on_rsi_3_windows = parameter_data["sma_on_rsi_3_windows"]

    '''Trend I EMA Information'''
    T1_ema_timeframes = parameter_data["T1_ema_timeframes"]  # refers to timeframe of loaded chart.
    T1_ema_1_windows = parameter_data["T1_ema_1_windows"]
    T1_ema_2_windows = parameter_data["T1_ema_2_windows"]

    # '''Trend II EMA Information'''
    # T2_ema_timeframes = parameter_data["T2_ema_timeframes"]
    # T2_ema_1_windows = parameter_data["T2_ema_1_windows"]
    # T2_ema_2_windows = parameter_data["T2_ema_2_windows"]

    '''TP and SL'''
    take_profit_points = parameter_data["take_profit_points"]
    stop_loss_points = parameter_data["stop_loss_points"]

    '''Compile Structure and Run Master Indicator'''
    Master_Indicator = vbt.IF(
        input_names=['open_data', 'low_data', 'high_data', 'close_data'],
        #
        param_names=['rsi_timeframes', 'rsi_windows',
                     'sma_on_rsi_1_windows', 'sma_on_rsi_2_windows', 'sma_on_rsi_3_windows',
                     'T1_ema_timeframes', 'T1_ema_1_windows', 'T1_ema_2_windows',
                     # 'T2_ema_timeframes', 'T2_ema_1_windows', 'T2_ema_2_windows',
                     'take_profit_points', 'stop_loss_points'],
        #
        output_names=['long_entries', 'long_exits', 'short_entries', 'short_exits',
                      'take_profit_points', 'stop_loss_points']
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
                # 'memory': 100 * 10 ** 9,
                # 'object_store_memory': 100 * 10 ** 9,
            },
            show_progress=True
        ),
        #
        rsi_timeframes='4h', rsi_windows=13,
        sma_on_rsi_1_windows=2, sma_on_rsi_2_windows=7, sma_on_rsi_3_windows=34,
        T1_ema_timeframes='1m', T1_ema_1_windows=13, T1_ema_2_windows=50,
        # T2_ema_timeframes='4h', T2_ema_1_windows=13, T2_ema_2_windows=50,
        take_profit_points=300, stop_loss_points=-300
    ).run(
        open_data, low_data, high_data, close_data,
        rsi_timeframes=rsi_timeframes, rsi_windows=rsi_windows,
        #
        sma_on_rsi_1_windows=sma_on_rsi_1_windows,
        sma_on_rsi_2_windows=sma_on_rsi_2_windows,
        sma_on_rsi_3_windows=sma_on_rsi_3_windows,
        #
        T1_ema_timeframes=T1_ema_timeframes,
        T1_ema_1_windows=T1_ema_1_windows, T1_ema_2_windows=T1_ema_2_windows,
        #
        # T2_ema_timeframes=T2_ema_timeframes,
        # T2_ema_1_windows=T2_ema_1_windows, T2_ema_2_windows=T2_ema_2_windows,
        take_profit_points=take_profit_points, stop_loss_points=stop_loss_points
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

