#!/usr/bin/env python3
import datetime

from Simulation_Handler.flexible_simulation import Flexible_Simulation_Optimization, Flexible_Simulation_Backtest
from Strategies.Money_Maker_Strategy import MMT_Strategy

Run_Time_Settings = dict(
    # Data Settings
    Data_Settings=dict(
        load_CSV_from_pickle=False,  # momentary
        data_files_dir='Datas',  # momentary
        data_files_names=[
            'AUDUSD',  # momentary
            'EURUSD',  # momentary
            'GBPUSD',  # momentary
            'NZDUSD',  # momentary
            'USDCAD',  # momentary
            'USDCHF',  # momentary
            # "DAX",  # momentary

        ],  # momentary

        delocalize_data=True,
        drop_nan=False,
        ffill=False,
        fill_dates=False,
        saved_data_file='SymbolData',
        tick_size=0.01
        # tick_size=0.00001
    ),

    Simulation_Settings=dict(
        study_name='debugging',
        # study_name='MMT_0',
        optimization_period=dict(
            start_date=datetime.datetime(month=9, day=1, year=2021),
            # end_date=datetime.datetime(month=3, day=16, year=2022)
            end_date=datetime.datetime(month=10, day=1, year=2021)
        ),
        #
        timer_limit=datetime.timedelta(days=0, hours=7, minutes=0, seconds=0),  # todo: logic missing,not used/needed
        Continue=False,
        speed_mode="eco_friendly",  # ["eco_friendly","ludicrous","plaid_plus"]
        #
        batch_size=2,
        save_every_nth_chunk=1,
        Initial_Search_Space=dict(
            # _extensions available -> csv and gzip
            path_of_initial_metrics_record='saved_param_metrics.csv',
            path_of_initial_params_record='saved_initial_params.csv',
            #
            max_initial_combinations=500,
            stop_after_n_epoch=None,
            #
            parameter_selection=dict(
                timeframes='all',
                windows='grid',
                tp_sl=dict(
                    bar_atr_days=datetime.timedelta(days=120, hours=0, minutes=0, seconds=0),
                    bar_atr_periods=[14],
                    bar_atr_multiplier=[3],
                    #
                    n_ratios=[0.2, 0.5, 1, 1.5, 2],
                    gamma_ratios=[0.5, 1, 1.5, 2, 2.5, 3],
                    number_of_bar_trends=1,
                ),
            ),
        ),

        Loss_Function=dict(
            metrics=[
                #
                'Total Return [%]',
                'Benchmark Return [%]',
                'Max Gross Exposure [%]',
                'Total Fees Paid',
                'Max Drawdown [%]',
                'Total Trades',
                'Win Rate [%]',
                'Best Trade [%]',
                'Worst Trade [%]',
                'Avg Winning Trade [%]',
                'Avg Losing Trade [%]',
                'Profit Factor',
                'Sharpe Ratio',
                'Omega Ratio',
                'Sortino Ratio',
            #
            ],
        ),
        #
        Optuna_Study=dict(
            sampler_name=None,
            multi_objective_bool=None, )

    ),
    Portfolio_Settings=dict(
        Simulator=dict(
            backtesting=Flexible_Simulation_Backtest,
            optimization=Flexible_Simulation_Optimization,
        ),
        #
        sim_timeframe='1m',
        JustLoadpf=False,
        saved_pf_backtest='My_pf_backtest',
        saved_pf_optimization='My_pf_optimization',
        slippage=0,  # 0.0001,
        trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
        cash_sharing=False,
        group_by=[],  # Leave blank

        # Strategy
        # max_orders=-1,
        init_cash=1_000_000,
        size_type='cash',  # 'shares',  # cash or shares
        size=25_000,  # cash, else set size type to shares for share amount
        type_percent=False,  # if true then take_profit and stoploss are given in percentages, else cash amount

    ),
    Strategy_Settings=dict(
        Strategy=MMT_Strategy,
        # The order of parameter key_names should be honored across all files
        parameter_windows=dict(
            PEAK_and_ATR_timeframes=dict(type='timeframe',
                                         values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),

            atr_windows=dict(type='window', lower_bound=1, upper_bound=10, min_step=1),
            data_lookback_windows=dict(type='window', lower_bound=2, upper_bound=16, min_step=1),
            EMAs_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
            ema_1_windows=dict(type='window', lower_bound=5, upper_bound=45, min_step=1),
            ema_2_windows=dict(type='window', lower_bound=20, upper_bound=60, min_step=1),
            #
            take_profit_points=dict(type='take_profit', lower_bound=1, upper_bound=1000, min_step=1),
            stoploss_points=dict(type='stop_loss', lower_bound=1, upper_bound=1000, min_step=1),

        ),
        strategy_user_picked_params=dict(
            output_file_name='backtest_result.csv',
            # if compute_product then will compute the product of all the parameter values passed,
            #   else parameter values length must be equal
            compute_product=False,
            #
            # Can Read Parameters from file instead if the path to it is provided
            # read_user_defined_param_file='backtest_result.csv',
            read_user_defined_param_file='backtest_parameters.csv',  # todo: logic missing
            #
            # Can use  -->  values = np.arrange(start,stop,step) or np.linespace(start,stop,#)
            # The order of parameter key_names should be honored across all files
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
                stoploss_points=dict(type='stop_loss', values=[556]),
            )
        ),
    ),
    # It faster when values given, if not pass 'auto' and I will do my best
    RAY_SETTINGS=dict(
        ray_init_num_cpus=28,
        simulate_signals_num_cpus=28
    )
)
