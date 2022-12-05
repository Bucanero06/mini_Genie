import pandas as pd
import vectorbtpro as vbt

from Modules.Actors_Old.Utils import rsi_params_filter
from mini_genie_source.Indicators.simple_indicators import EMA
from mini_genie_source.Utilities.bars_utilities import BARSINCE_genie

def rlgl_post_cartesian_product_filter_function(parameters_record, **kwargs):
    import numpy as np
    from Modules.Actors_Old.Utils import convert_to_seconds
    parameters_record = parameters_record[
        np.where(np.less_equal([convert_to_seconds(i) for i in parameters_record["T1_ema_timeframes"]],
                               [convert_to_seconds(i) for i in parameters_record["rsi_timeframes"]]))[0]]

    parameters_record = parameters_record[
        np.where(parameters_record["T1_ema_1_windows"] <= parameters_record["T1_ema_2_windows"])[0]]

    parameters_record = parameters_record[
        np.where(parameters_record["sma_on_rsi_1_windows"] <= parameters_record["sma_on_rsi_2_windows"])[
            0]]

    parameters_record = parameters_record[
        np.where(parameters_record["sma_on_rsi_2_windows"] <= parameters_record["sma_on_rsi_3_windows"])[
            0]]
    return parameters_record


Strategy_Settings = dict(
    Strategy="RLGL_Strategy",
    # The order of parameter key_names should be honored across all files
    _pre_cartesian_product_filter=dict(
        function=rsi_params_filter,
        kwargs=dict(
            low_rsi=35,
            high_rsi=65
        )
    ),
    _post_cartesian_product_filter=dict(
        function=rlgl_post_cartesian_product_filter_function,
        kwargs=dict()
    ),

    parameter_windows=dict(
            rsi_timeframes=dict(type='timeframe', values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),
            rsi_windows=dict(type='window', lower_bound=2, upper_bound=98, min_step=1),
            #
            sma_on_rsi_1_windows=dict(type='window', lower_bound=2, upper_bound=50, min_step=1),
            sma_on_rsi_2_windows=dict(type='window', lower_bound=5, upper_bound=70, min_step=1),
            sma_on_rsi_3_windows=dict(type='window', lower_bound=15, upper_bound=90, min_step=1),
            #
            T1_ema_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
            T1_ema_1_windows=dict(type='window', lower_bound=2, upper_bound=65, min_step=1),
            T1_ema_2_windows=dict(type='window', lower_bound=15, upper_bound=70, min_step=1),
            #
            # T2_ema_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h', '1d']),
            # T2_ema_1_windows=dict(type='window', lower_bound=2, upper_bound=10, min_step=1),
            # T2_ema_2_windows=dict(type='window', lower_bound=2, upper_bound=10, min_step=1),
            #
            take_profit_points=dict(type='take_profit', lower_bound=1, upper_bound=10000000, min_step=100000),
            stop_loss_points=dict(type='stop_loss', lower_bound=1, upper_bound=10000000, min_step=100000),
            #
            #
            #
            # breakeven_1_trigger_bool=False,
            # breakeven_1_trigger_points=dict(step_n_type='break_even_trigger', lower_bound=50, upper_bound=2000),
            # breakeven_1_distance_points=dict(step_n_type='break_even_distance', lower_bound=20, upper_bound=2000),
            # #
            # breakeven_2_trigger_bool=False,
            # breakeven_2_trigger_points=dict(step_n_type='break_even_trigger', lower_bound=50, upper_bound=2000),
            # breakeven_2_distance_points=dict(step_n_type='break_even_distance', lower_bound=20, upper_bound=2000),

        ),
    # strategy_user_picked_params
    strategy_user_picked_params=dict(
        output_file_name='backtest_result.csv',
        # if compute_product then will compute the product of all the parameter values passed,
        #   else parameter values length must be equal
        compute_product=True,
        #
        # Can Read Parameters from file instead if the path to it is provided
        # read_user_defined_param_file='backtest_result.csv',
        read_user_defined_param_file=None,
        #
        # Can use  -->  values = np.arrange(start,stop,step) or np.linespace(start,stop,#)
        # The order of parameter key_names should be honored across all files
        parameter_windows=dict(
            rsi_timeframes=dict(type='timeframe', values=['15 min', '15 min']),
            rsi_windows=dict(type='window', values=[41, 30]),
            #
            sma_on_rsi_1_windows=dict(type='window', values=[32, 43]),
            sma_on_rsi_2_windows=dict(type='window', values=[26, 26]),
            sma_on_rsi_3_windows=dict(type='window', values=[15, 15]),
            #
            T1_ema_timeframes=dict(type='timeframe', values=['1 min', "5 min"]),
            T1_ema_1_windows=dict(type='window', values=[2, 3]),
            T1_ema_2_windows=dict(type='window', values=[15, 20]),
            #
            # T2_ema_timeframes=dict(type='timeframe', values=['5 min']),
            # T2_ema_1_windows=dict(type='window', values=[5]),
            # T2_ema_2_windows=dict(type='window', values=[5]),
            #
            take_profit_points=dict(type='take_profit', values=[86, 200]),
            stop_loss_points=dict(type='stop_loss', values=[-43, 200]),
        )
    ),
)





def cache_func(close_data,
               rsi_timeframes, rsi_windows,
               sma_on_rsi_1_windows, sma_on_rsi_2_windows, sma_on_rsi_3_windows,
               T1_ema_timeframes, T1_ema_1_windows, T1_ema_2_windows,
               # T2_ema_timeframes, T2_ema_1_windows, T2_ema_2_windows,
               #
               take_profit_points, stop_loss_points
               ):
    #
    cached_data = {
        # Data
        'Close': {},
        'Resamplers': {},
        'Empty_df_like': pd.DataFrame().reindex_like(close_data),
    }
    # if type(rsi_timeframes) == type(T1_ema_timeframes) == list:
    '''Pre-Resample Data'''
    timeframes_set = tuple(set(tuple(rsi_timeframes) + tuple(T1_ema_timeframes)))  # + tuple(T2_ema_timeframes)))

    for timeframe in timeframes_set:
        # CLOSE
        cached_data['Close'][timeframe] = close_data.vbt.resample_apply(timeframe,
                                                                        vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close_data

        '''Pre-Prepare Resampler'''
        cached_data['Resamplers'][timeframe] = vbt.Resampler(
            cached_data['Close'][timeframe].index,
            close_data.index,
            source_freq=timeframe,
            target_freq="1m") if timeframe != '1 min' else None

    return cached_data


def apply_function(close_data,
                   rsi_timeframe, rsi_window,
                   sma_on_rsi_1_window, sma_on_rsi_2_window, sma_on_rsi_3_window,
                   T1_ema_timeframe, T1_ema_1_window, T1_ema_2_window,
                   # T2_ema_timeframe, T2_ema_1_window, T2_ema_2_window,
                   #
                   take_profit_points, stop_loss_points,
                   cached_data):
    """Function for RLGL Strategy/Indicators"""

    '''RSI and SMA Indicators'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    rsi_indicator = vbt.RSI.run(cached_data['Close'][rsi_timeframe], window=rsi_window).rsi
    sma_on_rsi_1_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_1_window).ma
    sma_on_rsi_2_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_2_window).ma
    sma_on_rsi_3_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_3_window).ma
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Trend I EMA Indicators'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # T1_ema_1_indicator = vbt.MA.run(cached_data['Close'][T1_ema_timeframe], window=T1_ema_1_window, ewm=True).ma
    # T1_ema_2_indicator = vbt.MA.run(cached_data['Close'][T1_ema_timeframe], window=T1_ema_2_window, ewm=True).ma

    T1_ema_1_indicator = EMA.run(close=cached_data['Close'][T1_ema_timeframe], window=T1_ema_1_window).ema
    T1_ema_2_indicator = EMA.run(close=cached_data['Close'][T1_ema_timeframe], window=T1_ema_2_window).ema
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    # '''Trend II EMA Indicators'''
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # T2_ema_1_indicator = cached_indicator['EMA'][f'{T2_ema_timeframe}_{T2_ema_1_window}']
    # T2_ema_2_indicator = cached_indicator['EMA'][f'{T2_ema_timeframe}_{T2_ema_2_window}']
    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    '''Resample Indicators Back To 1 minute'''
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Fetch the resamplers from cached_data for a given timeframe
    rsi_timeframe_to_1min_Resampler = cached_data['Resamplers'][rsi_timeframe]
    T1_ema_timeframe_to_1min_Resampler = cached_data['Resamplers'][T1_ema_timeframe]
    # T2_ema_timeframe_to_1min_Resampler = cached_data['Resamplers'][T2_ema_timeframe]

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
    empty_df_like = cached_data['Empty_df_like']
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
        input_names=['close_data'],
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
        close_data,
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
