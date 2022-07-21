#!/usr/bin/env python3
import datetime

debug_settings = dict(
    # Data Settings
    Data_Settings=dict(
        load_CSV_from_pickle=True,  # momentary
        data_files_dir='Datas',  # momentary
        data_files_names=[
            "XAUUSD",  # momentary
            # "OILUSD",  # momentary

        ],  # momentary

        delocalize_data=True,
        drop_nan=False,
        ffill=False,
        fill_dates=False,
        saved_data_file='SymbolData',
        # tick_size=0.01
        tick_size=[0.01]
    ),

    Simulation_Settings=dict(
        study_name='rlgl_debug',
        optimization_period=dict(
            start_date=datetime.datetime(month=2, day=1, year=2022),
            end_date=datetime.datetime(month=3, day=1, year=2022)
        ),
        #
        timer_limit=datetime.timedelta(days=0, hours=7, minutes=0, seconds=0),  # todo: logic missing,not used/needed
        Continue=False,
        run_mode="plaid_plus",  # todo: ["ludicrous","plaid_plus"]
        #
        batch_size=2,
        save_every_nth_chunk=1,
        Initial_Search_Space=dict(
            # _extensions available -> csv and gzip
            path_of_initial_metrics_record='saved_param_metrics.csv',
            path_of_initial_params_record='saved_initial_params.csv',
            #
            max_initial_combinations=1_000,
            stop_after_n_epoch=None,
            # force_to_finish=True,  # todo: logic missing
            #
            parameter_selection=dict(
                timeframes='all',  # todo: needs to add settings for how to reduce, these dont do anything
                windows='grid',  # todo: needs to add settings for how to reduce, these dont do anything
                tp_sl=dict(
                    bar_atr_days=datetime.timedelta(days=120, hours=0, minutes=0, seconds=0),
                    bar_atr_periods=[7],  # todo multiple inputs
                    bar_atr_multiplier=[3],  # todo multiple inputs
                    #
                    n_ratios=[0.5, 1, 1.5],  # Scaling factor for \bar{ATR}
                    gamma_ratios=[1, 1.5],  # Risk Reward Ratio
                    number_of_bar_trends=1,
                ),
            ),
        ),

        Loss_Function=dict(
            metrics=[
                'Total Return [%]',
                # 'Benchmark Return [%]',
                # 'Max Gross Exposure [%]',
                # 'Total Fees Paid',
                # 'Max Drawdown [%]',
                'Expectancy',
                'Total Trades',
                # 'Win Rate [%]',
                # 'Best Trade [%]',
                # 'Worst Trade [%]',
                # 'Avg Winning Trade [%]',
                # 'Avg Losing Trade [%]',
                # 'Profit Factor',
                # 'Sharpe Ratio',
                # 'Omega Ratio',
                # 'Sortino Ratio',
            ],
        ),
        #
        Optuna_Study=dict(
            sampler_name=None,
            multi_objective_bool=None, )

    ),
    Portfolio_Settings=dict(
        # Simulation Settings
        Simulator=dict(
            backtesting="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Backtest",
            optimization="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Optimization",
        ),
        #
        sim_timeframe='1m',
        JustLoadpf=False,
        slippage=0,  # 0.0001,
        max_spread_allowed=250,  # 0.0001,
        trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
        cash_sharing=False,
        group_by=[],  # Leave blank

        # Strategy
        # max_orders=-1,
        init_cash=1_000_000,
        size_type='cash',  # 'shares',  # cash or shares
        size=100_000,  # cash, else set size type to shares for share amount
        type_percent=False,  # if true then take_profit and stop_loss are given in percentages, else cash amount
    ),
    Strategy_Settings=dict(
        Strategy="mini_genie_source/Strategies/RLGL_Strategy.py.RLGL_Strategy",
        # The order of parameter key_names should be honored across all files
        parameter_windows=dict(
            rsi_timeframes=dict(type='timeframe', values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),
            rsi_windows=dict(type='window', lower_bound=2, upper_bound=100, min_step=1),
            #
            sma_on_rsi_1_windows=dict(type='window', lower_bound=2, upper_bound=63, min_step=1),
            sma_on_rsi_2_windows=dict(type='window', lower_bound=5, upper_bound=70, min_step=1),
            sma_on_rsi_3_windows=dict(type='window', lower_bound=15, upper_bound=80, min_step=1),
            #
            T1_ema_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
            T1_ema_1_windows=dict(type='window', lower_bound=2, upper_bound=63, min_step=1),
            T1_ema_2_windows=dict(type='window', lower_bound=15, upper_bound=80, min_step=1),
            #
            # T2_ema_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h', '1d']),
            # T2_ema_1_windows=dict(type='window', lower_bound=2, upper_bound=10, min_step=1),
            # T2_ema_2_windows=dict(type='window', lower_bound=2, upper_bound=10, min_step=1),
            #
            take_profit_points=dict(type='take_profit', lower_bound=1, upper_bound=100000, min_step=1000),
            stop_loss_points=dict(type='stop_loss', lower_bound=1, upper_bound=100000, min_step=1000),
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
    ),
    # It faster when values given, if not pass 'auto' and I will do my best
    RAY_SETTINGS=dict(
        ray_init_num_cpus=28,
        simulate_signals_num_cpus=28
    )
)
