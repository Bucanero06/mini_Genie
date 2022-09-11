#!/usr/bin/env python3
# --- ↓ Do not remove these libs ↓ -------------------------------------------------------------------------------------

import gc
import warnings

from mini_Genie.mini_genie_source.Indicators.simple_indicators import ATR_EWM, EMA

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from logger_tt import logger
from mini_Genie.mini_genie_source.Utilities.bars_utilities import BARSINCE_genie, ROLLING_MAX_genie, ROLLING_MIN_genie

# --- ↑ Do not remove these libs ↑ -------------------------------------------------------------------------------------


Strategy_Settings = dict(
    Strategy="MMT_Strategy",
    # The order of parameter key_names should be honored across all files
    parameter_windows=dict(
        Trend_filter_1_timeframes=dict(type='timeframe', values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),
        Trend_filter_atr_windows=dict(type='window', lower_bound=7, upper_bound=14, min_step=1),
        Trend_filter_1_data_lookback_windows=dict(type='window', lower_bound=3, upper_bound=8, min_step=1),
        #
        PEAK_and_ATR_timeframes=dict(type='timeframe', values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),
        #
        atr_windows=dict(type='window', lower_bound=7, upper_bound=14, min_step=1),
        data_lookback_windows=dict(type='window', lower_bound=3, upper_bound=8, min_step=1),
        EMAs_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
        ema_1_windows=dict(type='window', lower_bound=7, upper_bound=50, min_step=1),
        ema_2_windows=dict(type='window', lower_bound=20, upper_bound=80, min_step=1),
        #
        take_profit_points=dict(type='take_profit', lower_bound=1, upper_bound=10000000, min_step=1000),
        stop_loss_points=dict(type='stop_loss', lower_bound=1, upper_bound=10000000, min_step=1000),
    ),
    strategy_user_picked_params=dict(
        output_file_name='backtest_result.csv',
        compute_product=True,
        read_user_defined_param_file=None,
        parameter_windows=dict(
            PEAK_and_ATR_timeframes=dict(type='timeframe', values=['5 min']),
            #
            atr_windows=dict(type='window', values=[5]),
            data_lookback_windows=dict(type='window', values=[5]),
            EMAs_timeframes=dict(type='timeframe', values=['15 min']),
            ema_1_windows=dict(type='window', values=[27]),
            ema_2_windows=dict(type='window', values=[28]),
            #
            take_profit_points=dict(type='take_profit', values=[909]),
            stop_loss_points=dict(type='stop_loss', values=[556]),
        )
    ),
)


def comb_split_price_n_datetime_index(price_cols, datetime_cols, n_splits):
    from Utils import comb_price_and_range_index

    if np.ndim(price_cols) == 2:
        new_split_data = [int(0) for x in range(n_splits)]
        for j in range(n_splits):
            new_split_data[j] = comb_price_and_range_index(price_cols[j], datetime_cols[j])
        return new_split_data
    else:
        logger.exception("split price_cols is not 2D array")


def resample_split_data(split_data, timeframe):
    n_splits = len(split_data)
    print(split_data)
    exit()
    new_split_data = [int(0) for x in range(n_splits)]
    for j in range(n_splits):
        # print(split_data[j])

        new_split_data[j] = split_data[j].vbt.resample_apply(timeframe,
                                                             vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else \
            split_data[j]

    return new_split_data


def cache_func(low, high, close,
               # datetime_index,
               Trend_filter_1_timeframes, Trend_filter_atr_windows, Trend_filter_1_data_lookback_windows,
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
    timeframes = tuple(set(
        tuple(PEAK_and_ATR_timeframes) + tuple(EMAs_timeframes) + tuple(Trend_filter_1_timeframes)
    ))
    #
    '''Pre-Resample Data'''
    #
    for timeframe in timeframes:
        # cache['Low'][timeframe] = resample_split_data(low,  timeframe=timeframe)
        # cache['High'][timeframe] = resample_split_data(high,  timeframe=timeframe)
        # cache['Close'][timeframe] = resample_split_data(close,  timeframe=timeframe)
        #
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

    return cache


def apply_function(low_data, high_data, close_data,
                   # datetime_index,
                   Trend_filter_1_timeframe, Trend_filter_atr_window, Trend_filter_1_data_lookback_window,
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
    #
    Trend_filter_1_timeframe_low = cache['Low'][Trend_filter_1_timeframe]
    Trend_filter_1_timeframe_high = cache['High'][Trend_filter_1_timeframe]
    Trend_filter_1_timeframe_close = cache['Close'][Trend_filter_1_timeframe]
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    ''' Trend_Filter 1 ATR Indicator'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch pre-computed atr from cache. Uses Trend_filter_1_timeframe
    # Trend_filter_1_atr_indicator = vbt.indicators.ATR.run(
    #     Trend_filter_1_timeframe_high,
    #     Trend_filter_1_timeframe_low,
    #     Trend_filter_1_timeframe_close,
    #     window=Trend_filter_atr_window,
    #     ewm=False,
    #     short_name='atr').atr

    # Trend_filter_1_atr_indicator=vbt.ATR.run(high=Trend_filter_1_timeframe_high, low=Trend_filter_1_timeframe_low,
    #                                            close=Trend_filter_1_timeframe_close, window=Trend_filter_atr_window).atr
    Trend_filter_1_atr_indicator = ATR_EWM.run(high=Trend_filter_1_timeframe_high, low=Trend_filter_1_timeframe_low,
                                               close=Trend_filter_1_timeframe_close, window=Trend_filter_atr_window).atr

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Trend_Filter 1 PeakHigh and PeakLow'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # All indicators and datas in this section use the Trend_filter_1_timeframe
    #
    # Compute the rolling max of the high_data using a window of size data_lookback_window "highest(3,high)"
    Trend_filter_1_rolling_max = ROLLING_MAX_genie(Trend_filter_1_timeframe_high.to_numpy(),
                                                   Trend_filter_1_data_lookback_window)
    # Compare where the high_data is the same as the rolling_max "high == highest(3,high)"
    Trend_filter_1_high_eq_highest_in_N = Trend_filter_1_timeframe_high == Trend_filter_1_rolling_max
    # Find where the diff b/w the high_data and close_data is bigger than the ATR "( high - close > atr(AtrPeriod) )"
    Trend_filter_1_high_minus_close_gt_atr = (
                                                     Trend_filter_1_timeframe_high.to_numpy() - Trend_filter_1_timeframe_close.to_numpy()) \
                                             > Trend_filter_1_atr_indicator
    # Compute the PeakHigh "( high == highest(3,high) and ( high - close > atr(AtrPeriod) )"
    Trend_filter_1_PeakHigh = (Trend_filter_1_high_eq_highest_in_N) & (Trend_filter_1_high_minus_close_gt_atr)

    # Compute the rolling min of the low_data using a window of size data_lookback_window "lowest(3,low)"
    Trend_filter_1_rolling_min = ROLLING_MIN_genie(Trend_filter_1_timeframe_low.to_numpy(),
                                                   Trend_filter_1_data_lookback_window)
    # Compare where the low_data is the same as the rolling_min "low == lowest(3,low)"
    Trend_filter_1_low_eq_lowest_in_N = Trend_filter_1_timeframe_low == Trend_filter_1_rolling_min
    # Find where the diff b/w the close_data and low_data is bigger than the ATR "( close - low > atr(AtrPeriod) ) "
    Trend_filter_1_close_minus_low_bt_atr = (
                                                    Trend_filter_1_timeframe_close.to_numpy() - Trend_filter_1_timeframe_low.to_numpy()) \
                                            > Trend_filter_1_atr_indicator
    # Compute the PeakLow "( low == lowest(3,low) and ( close - low > atr(AtrPeriod) )  "
    Trend_filter_1_PeakLow = (Trend_filter_1_low_eq_lowest_in_N) & (Trend_filter_1_close_minus_low_bt_atr)
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    #########################################################################################33
    #########################################################################################33
    #########################################################################################33
    #########################################################################################33

    '''ATR Indicator'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch pre-computed atr from cache. Uses PEAK_and_ATR_timeframe
    # atr_indicator = vbt.indicators.ATR.run(
    #     PEAK_and_ATR_timeframe_high,
    #     PEAK_and_ATR_timeframe_low,
    #     PEAK_and_ATR_timeframe_close,
    #     window=atr_window,
    #     ewm=False,
    #     short_name='atr').atr
    atr_indicator = ATR_EWM.run(high=PEAK_and_ATR_timeframe_high, low=PEAK_and_ATR_timeframe_low,
                                close=PEAK_and_ATR_timeframe_close, window=atr_window).atr
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''PeakHigh and PeakLow'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # All indicators and datas in this section use the PEAK_and_ATR_timeframe
    #
    # Compute the rolling max of the high_data using a window of size data_lookback_window "highest(3,high)"

    rolling_max = ROLLING_MAX_genie(PEAK_and_ATR_timeframe_high.to_numpy(), data_lookback_window)
    # from Utilities.test_utilities import rolling_max as rolling_max
    # rolling_max_ai = rolling_max(PEAK_and_ATR_timeframe_high.to_numpy(), data_lookback_window)
    # print(f'{rolling_max == rolling_max_ai = }')

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
    # ema_1_indicator = vbt.MA.run(cache['Close'][EMAs_timeframe], window=ema_1_window, ewm=True).ma
    # ema_2_indicator = vbt.MA.run(cache['Close'][EMAs_timeframe], window=ema_2_window, ewm=True).ma
    ema_1_indicator = EMA.run(close=cache['Close'][EMAs_timeframe], window=ema_1_window).ema
    ema_2_indicator = EMA.run(close=cache['Close'][EMAs_timeframe], window=ema_2_window).ema

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Resample Indicators Back To 1 minute'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch the resamplers from cache for a given timeframe
    Trend_filter_1_timeframe_to_1min_Resampler = cache['Resamplers'][Trend_filter_1_timeframe]
    PEAK_and_ATR_timeframe_to_1min_Resampler = cache['Resamplers'][PEAK_and_ATR_timeframe]
    EMAs_timeframe_to_1min_Resampler = cache['Resamplers'][EMAs_timeframe]

    # Resample indicators to 1m
    Trend_filter_1_PeakHigh = Trend_filter_1_PeakHigh.vbt.resample_closing(
        Trend_filter_1_timeframe_to_1min_Resampler) if Trend_filter_1_timeframe_to_1min_Resampler else Trend_filter_1_PeakHigh
    Trend_filter_1_PeakLow = Trend_filter_1_PeakLow.vbt.resample_closing(
        Trend_filter_1_timeframe_to_1min_Resampler) if Trend_filter_1_timeframe_to_1min_Resampler else Trend_filter_1_PeakLow
    #
    PeakHigh = PeakHigh.vbt.resample_closing(
        PEAK_and_ATR_timeframe_to_1min_Resampler) if PEAK_and_ATR_timeframe_to_1min_Resampler else PeakHigh
    PeakLow = PeakLow.vbt.resample_closing(
        PEAK_and_ATR_timeframe_to_1min_Resampler) if PEAK_and_ATR_timeframe_to_1min_Resampler else PeakLow
    ema_1_indicator = ema_1_indicator.vbt.resample_closing(
        EMAs_timeframe_to_1min_Resampler) if EMAs_timeframe_to_1min_Resampler else ema_1_indicator
    ema_2_indicator = ema_2_indicator.vbt.resample_closing(
        EMAs_timeframe_to_1min_Resampler) if EMAs_timeframe_to_1min_Resampler else ema_2_indicator
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # Trend_filter_1_
    '''Long Entries Conditions'''
    # Bars since last PeakLow are less than Bars PeakHigh "barssince(PeakLow) < barssince(PeakHigh)"
    long_peak_condition = BARSINCE_genie(PeakLow).lt(BARSINCE_genie(PeakHigh))
    long_trend_1_peak_condition = BARSINCE_genie(Trend_filter_1_PeakLow).lt(BARSINCE_genie(Trend_filter_1_PeakHigh))
    long_entry_condition_1 = long_peak_condition & long_trend_1_peak_condition.to_numpy()
    #
    # EMA 1 crosses above EMA 2 "crossover(ema_EmaTF(13) , ema_EmaTF(50) )"
    long_entry_condition_2 = ema_1_indicator.vbt.crossed_above(ema_2_indicator)

    '''Short Entries Conditions'''
    # Bars since last PeakLow are greater than Bars PeakHigh "barssince(PeakLow) > barssince(PeakHigh)"
    short_peak_condition = BARSINCE_genie(PeakLow).gt(BARSINCE_genie(PeakHigh))
    short_trend_1_peak_condition = BARSINCE_genie(Trend_filter_1_PeakLow).gt(BARSINCE_genie(Trend_filter_1_PeakHigh))
    short_entry_condition_1 = short_peak_condition & short_trend_1_peak_condition.to_numpy()

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
    gc.collect()

    return long_entries, long_exits, short_entries, short_exits, take_profit_points, stop_loss_points


def MMT_Strategy(open_data, low_data, high_data, close_data, parameter_data, param_product, ray_sim_n_cpus):
    """MMT_Strategy"""
    # ATR_EWM = vbt.IF.from_expr("""
    #                         ATR:
    #                         tr0 = abs(high - low)
    #                         tr1 = abs(high - fshift(close))
    #                         tr2 = abs(low - fshift(close))
    #                         tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
    #
    #                         print(window)
    #                         exit()
    #                         atr = @talib_ema(tr, 2 * window - 1)  # Wilder's EMA
    #                         tr, atr
    #                         """)

    #
    # print(ATR_EWM.run(high=high_data, low=low_data,
    #                   close=close_data, window=np.array(parameter_data["Trend_filter_atr_windows"])).atr)
    # exit()
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Trend_filter_1_timeframes, Trend_filter_atr_windows, Trend_filter_1_data_lookback_windows
    Trend_filter_1_timeframes = np.array(parameter_data["Trend_filter_1_timeframes"])
    Trend_filter_atr_windows = np.array(parameter_data["Trend_filter_atr_windows"])
    Trend_filter_1_data_lookback_windows = np.array(parameter_data["Trend_filter_1_data_lookback_windows"])
    #
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

    # print(high_data)
    # print(pd.concat(high_data, axis=2))
    # exit()
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #`1

    '''Compile Structure and Run Master Indicator'''
    Master_Indicator = vbt.IF(
        input_names=[
            'low_data', 'high_data', 'close_data',
            # 'datetime_index',
        ],
        param_names=[
            'Trend_filter_1_timeframes', 'Trend_filter_atr_windows', 'Trend_filter_1_data_lookback_windows',
            'PEAK_and_ATR_timeframes', 'atr_windows', 'data_lookback_windows',
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
        param_product=param_product,
        execute_kwargs=dict(
            engine='ray',
            init_kwargs={
                # 'address': 'auto',
                'num_cpus': ray_sim_n_cpus,
                'ignore_reinit_error': True,
            },
            show_progress=True
        ),
        Trend_filter_1_timeframes='1d',
        Trend_filter_atr_windows=5,
        Trend_filter_1_data_lookback_windows=3,
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

        # comb_split_price_n_datetime_index(low_data[0], close_data[1], 10),
        # comb_split_price_n_datetime_index(high_data[0], close_data[1], 10),
        # comb_split_price_n_datetime_index(close_data[0], close_data[1], 10),
        # try_align_to_datetime_index(
        #     low_data.index,
        #     datetime_index,
        # ),
        # try_align_to_datetime_index(
        #     high_data.index,
        #     datetime_index,
        # ),
        # try_align_to_datetime_index(
        #     low_data.index,
        #     close_data,
        # ),

        # datetime_index if datetime_index is not None else close_data.index,
        Trend_filter_1_timeframes=Trend_filter_1_timeframes,
        Trend_filter_atr_windows=Trend_filter_atr_windows,
        Trend_filter_1_data_lookback_windows=Trend_filter_1_data_lookback_windows,
        PEAK_and_ATR_timeframes=PEAK_and_ATR_timeframes,
        atr_windows=atr_windows,
        data_lookback_windows=data_lookback_windows,
        EMAs_timeframes=EMAs_timeframes,
        ema_1_windows=ema_1_windows,
        ema_2_windows=ema_2_windows,
        take_profit_points=take_profit_points,
        stop_loss_points=stop_loss_points,
    )

    '''Type C conditions'''
    strategy_specific_kwargs = dict(
        exit_on_opposite_direction_entry=True,  # strategy_specific_kwargs['exit_on_opposite_direction_entry'],
        #
        progressive_bool=True,  # Master_Indicator.progressive_bool,
        #
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
