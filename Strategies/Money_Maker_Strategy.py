import numpy as np
import pandas as pd
import vectorbtpro as vbt

from mini_genie_source.Utilities.bars_utilities import BARSINCE_genie, ROLLING_MAX_genie, ROLLING_MIN_genie


def cache_func(open, low, high, close,
               PEAK_and_ATR_timeframes, atr_windows, data_lookback_windows,
               EMAs_timeframes, ema_1_windows, ema_2_windows, ema_3_windows,
               # progressive_bool,
               # breakeven_1_trigger_bool,
               # breakeven_1_trigger_points, breakeven_1_distance_points,
               # breakeven_2_trigger_bool,
               # breakeven_2_trigger_points, breakeven_2_distance_points,
               # take_profit_bool,
               take_profit_points,
               # stoploss_bool,
               stoploss_points):
    cache = {
        # Data
        'Open': {},
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
    timeframes = set(PEAK_and_ATR_timeframes + EMAs_timeframes)
    for timeframe in timeframes:
        # '''Pre-Resample Data'''
        # OPEN
        # cache['Open'][timeframe] = open.vbt.resample_opening(timeframe) if timeframe != '1m' else open
        # LOW
        cache['Low'][timeframe] = low.vbt.resample_apply(timeframe,
                                                         vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else low
        # HIGH
        cache['High'][timeframe] = high.vbt.resample_apply(timeframe,
                                                           vbt.nb.max_reduce_nb).dropna() if timeframe != '1 min' else high
        # CLOSE
        # cache['Close'][timeframe] = close.vbt.resample_closing(timeframe).dropna() if timeframe != '1 min' else close
        cache['Close'][timeframe] = close.vbt.resample_apply(timeframe,
                                                             vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close

        # # OPEN
        # # cache['Open'][timeframe] = open.vbt.resample_opening(timeframe) if timeframe != '1 min' else open
        # # LOW
        # cache['Low'][timeframe] = low.vbt.resample_apply(timeframe,
        #                                                  vbt.nb.min_reduce_nb) if timeframe != '1 min' else low
        # # HIGH
        # cache['High'][timeframe] = high.vbt.resample_apply(timeframe,
        #                                                    vbt.nb.max_reduce_nb) if timeframe != '1 min' else high
        # # CLOSE
        # cache['Close'][timeframe] = close.vbt.resample_apply(timeframe,
        #                                                      vbt.nb.last_reduce_nb) if timeframe != '1 min' else close

        '''Pre-Prepare Resampler'''
        cache['Resamplers'][timeframe] = vbt.Resampler(
            cache['Close'][timeframe].index,
            close.index,
            source_freq=timeframe,
            target_freq="1m") if timeframe != '1 min' else None

    '''Pre-Compute Indicators'''
    # Create a set of PEAK_and_ATR_timeframes
    PEAK_and_ATR_timeframes = set(PEAK_and_ATR_timeframes)
    # Create a set of atr_windows
    atr_windows = set(atr_windows)
    for PEAK_and_ATR_timeframe in PEAK_and_ATR_timeframes:
        for atr_window in atr_windows:
            cache['ATR'][f'{PEAK_and_ATR_timeframe}_{atr_window}'] = vbt.indicators.ATR.run(
                cache['High'][PEAK_and_ATR_timeframe],
                cache['Low'][PEAK_and_ATR_timeframe],
                cache['Close'][PEAK_and_ATR_timeframe],
                window=atr_window,
                ewm=False,
                short_name='atr').atr

    # Create a set of EMAs_timeframes
    EMAs_timeframes = set(EMAs_timeframes)
    # Create a set of ema_windows
    ema_windows = set(ema_1_windows + ema_2_windows + ema_3_windows)
    for EMAs_timeframe in EMAs_timeframes:
        for ema_window in ema_windows:
            cache['EMA'][f'{EMAs_timeframe}_{ema_window}'] = vbt.MA.run(cache['Close'][EMAs_timeframe],
                                                                        window=ema_window,
                                                                        ewm=True).ma
    return cache


def apply_function(open_data, low_data, high_data, close_data,
                   PEAK_and_ATR_timeframe, atr_window, data_lookback_window,
                   EMAs_timeframe, ema_1_window, ema_2_window, ema_3_window,
                   # progressive_bool,
                   # breakeven_1_trigger_bool,
                   # breakeven_1_trigger_points, breakeven_1_distance_points,
                   # breakeven_2_trigger_bool,
                   # breakeven_2_trigger_points, breakeven_2_distance_points,
                   # take_profit_bool,
                   take_profit_points,
                   # stoploss_bool,
                   stoploss_points,
                   cache
                   ):
    """Function for Indicators"""

    '''Fetch Resampled Data'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Only fetch those that are needed from the cached dict
    # PEAK_and_ATR_timeframe_open = cache['Open'][PEAK_and_ATR_timeframe]
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
    ema_3_indicator = cache['EMA'][f'{EMAs_timeframe}_{ema_3_window}']
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
    ema_3_indicator = ema_3_indicator.vbt.resample_closing(
        EMAs_timeframe_to_1min_Resampler) if EMAs_timeframe_to_1min_Resampler else ema_3_indicator
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Long Entries Conditions'''
    # Bars since last PeakLow are less than Bars PeakHigh "barssince(PeakLow) < barssince(PeakHigh)"
    long_entry_condition_1 = BARSINCE_genie(PeakLow).lt(BARSINCE_genie(PeakHigh))
    # EMA 1 crosses above EMA 2 "crossover(ema_EmaTF(13) , ema_EmaTF(50) )"
    long_entry_condition_2 = ema_1_indicator.vbt.crossed_above(ema_2_indicator)
    # Close price is above PeakLow
    long_entry_condition_2_b = close_data.to_numpy() > PeakLow

    '''Short Entries Conditions'''
    # Bars since last PeakLow are greater than Bars PeakHigh "barssince(PeakLow) > barssince(PeakHigh)"
    short_entry_condition_1 = BARSINCE_genie(PeakLow).gt(BARSINCE_genie(PeakHigh))
    # EMA 1 crosses below EMA 2 "crossunder(ema_EmaTF(13) , ema_EmaTF(50) )"
    short_entry_condition_2 = ema_1_indicator.vbt.crossed_below(ema_2_indicator)
    # Close price is below PeakHigh
    short_entry_condition_2_b = close_data.to_numpy() < PeakHigh

    '''Type C Conditions'''
    # Progressive long entry condition not being used right now "crossover(ema_EmaTF(34) , ema_EmaTF(50) )"
    # long_entry_condition_3 = ema_3_indicator.vbt.crossed_above(ema_2_indicator)
    long_entry_condition_3 = pd.DataFrame().reindex_like(long_entry_condition_1).fillna(False)
    # long_entry_condition_3 = long_entry_condition_2
    # Progressive short entry condition not being used right now "crossover(ema_EmaTF(34) , ema_EmaTF(50) )"
    # short_entry_condition_3 = ema_3_indicator.vbt.crossed_below(ema_2_indicator)
    short_entry_condition_3 = pd.DataFrame().reindex_like(short_entry_condition_1).fillna(False)
    # short_entry_condition_3 = short_entry_condition_2

    '''Fill Rest of Parameters for Sim'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Used to fill signals and parameter dfs into the correct size (just a workaround for now, fast)
    empty_df_like = cache['Empty_df_like']

    # progressive_bool = empty_df_like.fillna(progressive_bool)

    # stoploss_bool = empty_df_like.fillna(True) if breakeven_1_trigger_bool or breakeven_2_trigger_bool \
    #     else empty_df_like.fillna(stoploss_bool)

    stoploss_points = empty_df_like.fillna(stoploss_points)

    # breakeven_1_trigger_bool = empty_df_like.fillna(breakeven_1_trigger_bool)
    # breakeven_1_trigger_points = empty_df_like.fillna(breakeven_1_trigger_points)
    # breakeven_1_distance_points = empty_df_like.fillna(breakeven_1_distance_points)

    # breakeven_2_trigger_bool = empty_df_like.fillna(breakeven_2_trigger_bool)
    # breakeven_2_trigger_points = empty_df_like.fillna(breakeven_2_trigger_points)
    # breakeven_2_distance_points = empty_df_like.fillna(breakeven_2_distance_points)

    # take_profit_bool = empty_df_like.fillna(take_profit_bool)
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
    # # DEBUGGING
    # print(
    #     f"{long_entries.sum() =}"
    #     f"{short_entries.sum() =}"
    # )
    # exit()
    return long_entries, long_exits, short_entries, short_exits, \
           atr_indicator, PeakLow, PeakHigh, ema_1_indicator, ema_2_indicator, ema_3_indicator, \
           long_entry_condition_1, long_entry_condition_2, long_entry_condition_2_b, long_entry_condition_3, \
           short_entry_condition_1, short_entry_condition_2, short_entry_condition_2_b, short_entry_condition_3, \
           take_profit_points, \
           stoploss_points


# No Bools
# PeakLow, PeakHigh, \
# long_entry_condition_1, long_entry_condition_2, long_entry_condition_3, \
# short_entry_condition_1, short_entry_condition_2, short_entry_condition_3, \
# breakeven_1_trigger_points, breakeven_1_distance_points, \
# breakeven_2_trigger_points, breakeven_2_distance_points, \
# take_profit_points, \
# stoploss_points

# ALL
# PeakLow, PeakHigh, \
# long_entry_condition_1, long_entry_condition_2, long_entry_condition_3, \
# short_entry_condition_1, short_entry_condition_2, short_entry_condition_3, \
# progressive_bool, \
# breakeven_1_trigger_bool, breakeven_1_trigger_points, breakeven_1_distance_points, \
# breakeven_2_trigger_bool, breakeven_2_trigger_points, breakeven_2_distance_points, \
# take_profit_bool, take_profit_points, \
# stoploss_bool, stoploss_points


def MMT_Strategy(open_data, low_data, high_data, close_data, parameter_data):
    """MMT_Strategy"""
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    PEAK_and_ATR_timeframes = np.array(parameter_data["PEAK_and_ATR_timeframes"])
    atr_windows = np.array(parameter_data["atr_windows"])
    data_lookback_windows = np.array(parameter_data["data_lookback_windows"])

    EMAs_timeframes = np.array(parameter_data["EMAs_timeframes"])
    ema_1_windows = np.array(parameter_data["ema_1_windows"])
    ema_2_windows = np.array(parameter_data["ema_2_windows"])
    ema_3_windows = np.array(parameter_data["ema_3_windows"])
    # progressive_bool = np.array(data_and_parameter_dictionary["progressive_bool"])
    #
    # breakeven_1_trigger_bool = np.array(data_and_parameter_dictionary["breakeven_1_trigger_bool"])
    # breakeven_1_trigger_points = np.array(data_and_parameter_dictionary["breakeven_1_trigger_points"])
    # breakeven_1_distance_points = np.array(data_and_parameter_dictionary["breakeven_1_distance_points"])
    #
    # breakeven_2_trigger_bool = np.array(data_and_parameter_dictionary["breakeven_2_trigger_bool"])
    # breakeven_2_trigger_points = np.array(data_and_parameter_dictionary["breakeven_2_trigger_points"])
    # breakeven_2_distance_points = np.array(data_and_parameter_dictionary["breakeven_2_distance_points"])
    #
    # take_profit_bool = np.array(data_and_parameter_dictionary["take_profit_bool"])
    take_profit_points = np.array(parameter_data["take_profit_points"])
    #
    # stoploss_bool = np.array(data_and_parameter_dictionary["stoploss_bool"])
    stoploss_points = np.array(parameter_data["stoploss_points"])
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #
    '''Compile Structure and Run Master Indicator'''
    Master_Indicator = vbt.IF(
        input_names=[
            'open_data', 'low_data', 'high_data', 'close_data',
        ],
        param_names=['PEAK_and_ATR_timeframes', 'atr_windows', 'data_lookback_windows',
                     'EMAs_timeframes', 'ema_1_windows', 'ema_2_windows', 'ema_3_windows',
                     # 'progressive_bool',
                     # 'breakeven_1_trigger_bool',
                     # 'breakeven_1_trigger_points', 'breakeven_1_distance_points',
                     # 'breakeven_2_trigger_bool',
                     # 'breakeven_2_trigger_points', 'breakeven_2_distance_points',
                     # 'take_profit_bool',
                     'take_profit_points',
                     # 'stoploss_bool',
                     'stoploss_points'
                     ],
        output_names=[
            'long_entries', 'long_exits', 'short_entries', 'short_exits',
            'atr_indicator', 'PeakLow', 'PeakHigh', 'ema_1_indicator', 'ema_2_indicator', 'ema_3_indicator',
            'long_entry_condition_1', 'long_entry_condition_2', 'long_entry_condition_2_b',
            'long_entry_condition_3',
            'short_entry_condition_1', 'short_entry_condition_2', 'short_entry_condition_2_b',
            'short_entry_condition_3',
            # 'progressive_bool',
            # 'breakeven_1_trigger_bool',
            # 'breakeven_1_trigger_points', 'breakeven_1_distance_points',
            # 'breakeven_2_trigger_bool',
            # 'breakeven_2_trigger_points', 'breakeven_2_distance_points',
            # 'take_profit_bool',
            'take_profit_points',
            # 'stoploss_bool',
            'stoploss_points'
        ]
    ).with_apply_func(
        apply_func=apply_function,
        cache_func=cache_func,
        keep_pd=True,
        param_product=False,
        execute_kwargs=dict(
            engine='ray',
            # n_chunks='auto',
            chunk_len='auto',  # del_refs
            # init_kwargs=dict(),
            init_kwargs={
                # 'num_cpus': 18,
                'num_cpus': 28,
                # 'del_refs': False,
                # 'ignore_reinit_error': True
            },
            # # remote_kwargs={'num_cpus': 18},
            # remote_kwargs={'num_cpus': 14},
            show_progress=True
        ),
        PEAK_and_ATR_timeframes='1d',
        atr_windows=5,
        data_lookback_windows=3,
        EMAs_timeframes='1h',
        ema_1_windows=13,
        ema_2_windows=50,
        ema_3_windows=34,
        # #
        # progressive_bool=True,
        # #
        # breakeven_1_trigger_bool=True,
        # breakeven_1_trigger_points=100,
        # breakeven_1_distance_points=50,
        # #
        # breakeven_2_trigger_bool=True,
        # breakeven_2_trigger_points=200,
        # breakeven_2_distance_points=100,
        # #
        # take_profit_bool=True,
        take_profit_points=300,
        # #
        # stoploss_bool=True,
        stoploss_points=-600,
    ).run(
        open_data, low_data, high_data, close_data,
        #
        # # ####################################
        # PEAK_and_ATR_timeframes='1d',
        # atr_windows=5,
        # data_lookback_windows=3,
        # EMAs_timeframes='1h',
        # ema_1_windows=13,
        # ema_2_windows=50,
        # ema_3_windows=34,
        # # #
        # # progressive_bool=True,
        # # #
        # # breakeven_1_trigger_bool=True,
        # # breakeven_1_trigger_points=100,
        # # breakeven_1_distance_points=50,
        # # #
        # # breakeven_2_trigger_bool=True,
        # # breakeven_2_trigger_points=200,
        # # breakeven_2_distance_points=100,
        # # #
        # # take_profit_bool=True,
        # take_profit_points=300,
        # # #
        # # stoploss_bool=True,
        # stoploss_points=-600,
        # #
        # # ##############################################
        # # ####################################
        # PEAK_and_ATR_timeframes='15 min',
        # atr_windows=6,
        # data_lookback_windows=27,
        # EMAs_timeframes='1 min',  # 1 min
        # ema_1_windows=25,
        # ema_2_windows=62,
        # ema_3_windows=34,
        # # #
        # # progressive_bool=True,
        # # #
        # # breakeven_1_trigger_bool=True,
        # # breakeven_1_trigger_points=100,
        # # breakeven_1_distance_points=50,
        # # #
        # # breakeven_2_trigger_bool=True,
        # # breakeven_2_trigger_points=200,
        # # breakeven_2_distance_points=100,
        # # #
        # # take_profit_bool=True,
        # take_profit_points=680,
        # # #
        # # stoploss_bool=True,
        # stoploss_points=-380,
        # #
        # # ##############################################
        # # ####################################
        # PEAK_and_ATR_timeframes='4 h',
        # atr_windows=23,
        # data_lookback_windows=5,
        # EMAs_timeframes='30 min',
        # ema_1_windows=23,
        # ema_2_windows=43,
        # ema_3_windows=34,
        # # #
        # # progressive_bool=True,
        # # #
        # # breakeven_1_trigger_bool=True,
        # # breakeven_1_trigger_points=100,
        # # breakeven_1_distance_points=50,
        # # #
        # # breakeven_2_trigger_bool=True,
        # # breakeven_2_trigger_points=200,
        # # breakeven_2_distance_points=100,
        # # #
        # # take_profit_bool=True,
        # take_profit_points=580,
        # # #
        # # stoploss_bool=True,
        # stoploss_points=-380,
        # #
        # # ##############################################
        # ##############################################
        PEAK_and_ATR_timeframes=PEAK_and_ATR_timeframes,
        atr_windows=atr_windows,
        data_lookback_windows=data_lookback_windows,
        EMAs_timeframes=EMAs_timeframes,
        ema_1_windows=ema_1_windows,
        ema_2_windows=ema_2_windows,
        ema_3_windows=ema_3_windows,
        # #
        # progressive_bool=progressive_bool,
        # #
        # breakeven_1_trigger_bool=breakeven_1_trigger_bool,
        # breakeven_1_trigger_points=breakeven_1_trigger_points,
        # breakeven_1_distance_points=breakeven_1_distance_points,
        # #
        # breakeven_2_trigger_bool=breakeven_2_trigger_bool,
        # breakeven_2_trigger_points=breakeven_2_trigger_points,
        # breakeven_2_distance_points=breakeven_2_distance_points,
        # #
        # take_profit_bool=take_profit_bool,
        take_profit_points=take_profit_points,
        # #
        # stoploss_bool=stoploss_bool,
        stoploss_points=stoploss_points,
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
        stoploss_bool=True,  # Master_Indicator.stoploss_bool,
        stoploss_points=Master_Indicator.stoploss_points,
        stoploss_points_parameters=stoploss_points,
    )
    # strategy_specific_kwargs = dict()
    return Master_Indicator.long_entries, Master_Indicator.long_exits, \
           Master_Indicator.short_entries, Master_Indicator.short_exits, \
           strategy_specific_kwargs
