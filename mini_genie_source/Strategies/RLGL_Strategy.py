import vectorbtpro as vbt
import numpy as np
from logger_tt import logger
from genie_trader.utility_modules.Utils import convert_to_minutes

from time import perf_counter

# Public Parameters	   (Default Value)   #these are the parameters that are currently exposed to the end user

LotSize = 0.25  # A 'standard lot' is equal to $100,000. default trade size is 0.25 standard lot, or $25,000
TakeProfit = 500
StopLoss = 500
TrendType = 'T1'  # Can be 'T1' or 'T2', see Entry Logic below
CloseReversal = False
EnableBreakEven = True
BreakEven1 = 300
BreakEvenDis1 = 200
BreakEven2 = 400
BreakEvenDis2 = 300
EnableTrailing = False
EnableTrailing2 = False
TrailingStopDis = 80
EnableMaxTrades = False
MaxTrades = 10
EnableSession = False
SessionStart = '07:00'
SessionEnd = '22:00'


def apply_function(open_data, low_data, high_data, close_data,
                   rsi_windows, rsi_timeframes,
                   sma_on_rsi_len1_windows, sma_on_rsi_len2_windows, sma_on_rsi_len3_windows,
                   T1_ema_len1_windows, T1_ema_len2_windows, T1_ema_timeframes,
                   T2_ema_len1_windows, T2_ema_len2_windows, T2_ema_timeframes):
    """Function for Indicators"""

    '''Transform Windows'''
    # Convert desired timeframe to minutes
    rsi_windows_multiplier = convert_to_minutes(rsi_timeframes)
    T1_ema_len1_windows_multiplier = convert_to_minutes(T1_ema_timeframes)
    T2_ema_len1_windows_multiplier = convert_to_minutes(T2_ema_timeframes)
    # Transform windows
    rsi_windows = np.multiply(rsi_windows, rsi_windows_multiplier)
    T1_ema_len1_windows = np.multiply(T1_ema_len1_windows, T1_ema_len1_windows_multiplier)
    T2_ema_len1_windows = np.multiply(T2_ema_len1_windows, T2_ema_len1_windows_multiplier)

    '''RSI and SMA Indicators'''
    rsi_indicator = vbt.RSI.run(close_data, window=rsi_windows).rsi
    sma_on_rsi_len1_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_len1_windows).ma
    sma_on_rsi_len2_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_len2_windows).ma
    sma_on_rsi_len3_indicator = vbt.MA.run(rsi_indicator, window=sma_on_rsi_len3_windows).ma

    '''Trend I EMA Indicators'''
    T1_ema_len1_indicator = vbt.MA.run(close_data, window=T1_ema_len1_windows, ewm=True).ma
    T1_ema_len2_indicator = vbt.MA.run(close_data, window=T1_ema_len2_windows, ewm=True).ma

    '''Trend II EMA Indicators'''
    T2_ema_len1_indicator = vbt.MA.run(close_data, window=T2_ema_len1_windows, ewm=True).ma
    T2_ema_len2_indicator = vbt.MA.run(close_data, window=T2_ema_len2_windows, ewm=True).ma

    return rsi_indicator, \
           sma_on_rsi_len1_indicator, sma_on_rsi_len2_indicator, sma_on_rsi_len3_indicator, \
           T1_ema_len1_indicator, T1_ema_len2_indicator, \
           T2_ema_len1_indicator, T2_ema_len2_indicator


def RLGL_Strategy(symbols_data, **parameter_windows):
    """Red Light Geen Light Strategy"""

    '''Load Data'''
    open_data = symbols_data.get('Open')
    low_data = symbols_data.get('Low')
    high_data = symbols_data.get('High')
    close_data = symbols_data.get('Close')

    '''RSI and SMA Information'''
    rsi_windows = parameter_windows["rsi_windows"]["batch_values"]
    rsi_timeframes = parameter_windows["rsi_timeframes"]["batch_values"]
    #
    sma_on_rsi_len1_windows = parameter_windows["sma_on_rsi_len1_windows"]["batch_values"]
    sma_on_rsi_len2_windows = parameter_windows["sma_on_rsi_len2_windows"]["batch_values"]
    sma_on_rsi_len3_windows = parameter_windows["sma_on_rsi_len3_windows"]["batch_values"]

    '''Trend I EMA Information'''
    T1_ema_len1_windows = parameter_windows["T1_ema_len1_windows"]["batch_values"]
    T1_ema_len2_windows = parameter_windows["T1_ema_len2_windows"]["batch_values"]
    T1_ema_timeframes = parameter_windows["T1_ema_timeframes"]["batch_values"]  # refers to timeframe of loaded chart.

    '''Trend II EMA Information'''
    T2_ema_len1_windows = parameter_windows["T2_ema_len1_windows"]["batch_values"]
    T2_ema_len2_windows = parameter_windows["T2_ema_len2_windows"]["batch_values"]
    T2_ema_timeframes = parameter_windows["T2_ema_timeframes"]["batch_values"]

    '''Compile Structure and Run Master Indicator'''
    Master_Indicator = vbt.IF(
        input_names=['open_data', 'low_data', 'high_data', 'close_data'],
        param_names=['rsi_windows', 'rsi_timeframes',
                     'sma_on_rsi_len1_windows', 'sma_on_rsi_len2_windows', 'sma_on_rsi_len3_windows',
                     'T1_ema_len1_windows', 'T1_ema_len2_windows', 'T1_ema_timeframes',
                     'T2_ema_len1_windows', 'T2_ema_len2_windows', 'T2_ema_timeframes'],
        output_names=['rsi_period_ind',
                      'sma_on_rsi_len1_ind', 'sma_on_rsi_len2_ind', 'sma_on_rsi_len3_ind',
                      'T1_ema_len1_ind', 'T1_ema_len2_ind',
                      'T2_ema_len1_ind', 'T2_ema_len2_ind']
    ).with_apply_func(
        apply_func=apply_function,
        rsi_windows=13, rsi_timeframes='4h',
        sma_on_rsi_len1_windows=2, sma_on_rsi_len2_windows=7, sma_on_rsi_len3_windows=34,
        T1_ema_len1_windows=13, T1_ema_len2_windows=50, T1_ema_timeframes='1m',
        T2_ema_len1_windows=13, T2_ema_len2_windows=50, T2_ema_timeframes='4h'
    ).run(
        open_data, low_data, high_data, close_data,
        rsi_windows=rsi_windows, rsi_timeframes=rsi_timeframes,
        sma_on_rsi_len1_windows=sma_on_rsi_len1_windows,
        sma_on_rsi_len2_windows=sma_on_rsi_len2_windows,
        sma_on_rsi_len3_windows=sma_on_rsi_len3_windows,
        T1_ema_len1_windows=T1_ema_len1_windows, T1_ema_len2_windows=T1_ema_len2_windows,
        T1_ema_timeframes=T1_ema_timeframes,
        T2_ema_len1_windows=T2_ema_len1_windows, T2_ema_len2_windows=T2_ema_len2_windows,
        T2_ema_timeframes=T2_ema_timeframes,
        param_product=False,
        execute_kwargs=dict(
            engine='dask',
            chunk_len='auto',
            show_progress=True
        )
    )

    '''Buy Entries'''
    '''Condition I'''
    # TrendType I ema_timeframe#(N#) > ema_timeframe#(N#)

    # TrendType II ema_4h(13) > ema_4h(50)

    '''Condition II'''
    # crossover( ema(13) , ema(50) )

    '''Condition III'''

    logger.info(f'\n{Master_Indicator.rsi_period_ind.head()}')
    exit()

    '''Sell Entry'''
    '''Condition I'''
    # TrendType == 'T1' or (ema_4h(13) < ema_4h(50))
    '''Condition II'''
    # crossunder( ema(13) , ema(50) )
    '''Condition III'''

    return entries, exits

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
