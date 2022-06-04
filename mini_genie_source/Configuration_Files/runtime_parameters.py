#!/usr/bin/env python3
import datetime

from Simulation_Handler.flexible_simulation import Flexible_Simulation_Optimization, Flexible_Simulation_Backtest
from Strategies.Money_Maker_Strategy import MMT_Strategy

Run_Time_Settings = dict(
    # Data Settings
    Data_Settings=dict(
        load_CSV_from_pickle=True,  # momentary
        data_files_dir='Datas/Forex Majors/',  # momentary
        data_files_names=[
            'AUDUSD',  # momentary
            'EURUSD',  # momentary
            'GBPUSD',  # momentary
            'NZDUSD',  # momentary
            'USDCAD',  # momentary
            'USDCHF',  # momentary
        ],  # momentary
        #
        delocalize_data=True,
        drop_nan=False,
        ffill=False,
        fill_dates=False,
        saved_data_file='SymbolData',
        tick_size=0.00001
    ),

    Simulation_Settings=dict(
        study_name='MMT_0',
        optimization_period=dict(
            start_date=datetime.datetime(month=9, day=1, year=2021),
            end_date=datetime.datetime(month=3, day=16, year=2022)
        ),
        #
        timer_limit=datetime.timedelta(days=0, hours=7, minutes=0, seconds=0),  # todo: logic missing,not used/needed
        Continue=True,
        #
        # # whenever continuing, load parameter combs, then delete all with no trade ones, shuffle,
        # # fill in values with trades, then start run with the ones that are missing
        # soft_reset=False,  # todo: section of code missing
        # # Same as soft reset, but saves ones with trades in a different file and computes the rest
        # medium_reset=False,  # todo: section of code missing
        # # Deletes files and starts again from scratch with settings
        # hard_reset=False,  # todo: section of code missing
        # #
        # reset_withouts_those_with_no_trials=False,  # todo: section of code missing
        # reset_withouts_those_with_negative_profits=False,  # todo: section of code missing
        # reset_withouts_any_ran_ones=False,  # todo: section of code missing
        #
        batch_size=2,
        save_every_nth_chunk=1,
        Initial_Search_Space=dict(
            # _extensions available -> csv and gzip
            path_of_initial_metrics_record='saved_param_metrics.csv',
            path_of_initial_params_record='saved_initial_params.csv',
            #
            # max_initial_combinations=33_000_000,
            max_initial_combinations=500,
            stop_after_n_epoch=None,
            # force_to_finish=True,  # todo: logic missing
            #
            parameter_selection=dict(
                timeframes='all',  # todo: needs to add settings for how to reduce, these dont do anything
                windows='grid',  # todo: needs to add settings for how to reduce, these dont do anything
                tp_sl=dict(
                    bar_atr_days=datetime.timedelta(days=120, hours=0, minutes=0, seconds=0),
                    bar_atr_periods=[14],  # todo multiple inputs
                    bar_atr_multiplier=[3],  # todo multiple inputs
                    #
                    n_ratios=[0.2, 0.5, 1, 1.5, 2],
                    gamma_ratios=[0.5, 1, 1.5, 2, 2.5, 3],
                    number_of_bar_trends=1,
                ),
            ),
        ),

        Loss_Function=dict(
            metrics=[
                # # 'Risk/Reward Ratio',
                # # 'Risk Adjusted Return',
                # 'Total Trades',
                # 'Win Rate [%]',
                # 'Profit Factor',
                # 'Sortino Ratio',
                # 'Omega Ratio',
                # 'Total Return [%]',
                # 'Max Drawdown [%]',
                # # 'Max Gross Exposure [%]'
                #
                'Total Return [%]',
                'Benchmark Return [%]',
                'Max Gross Exposure [%]',
                'Total Fees Paid',
                'Max Drawdown [%]',
                # 'Max Drawdown Duration',
                'Total Trades',
                'Win Rate [%]',
                'Best Trade [%]',
                'Worst Trade [%]',
                'Avg Winning Trade [%]',
                'Avg Losing Trade [%]',
                # 'Avg Winning Trade Duration',
                # 'Avg Losing Trade Duration',
                'Profit Factor',
                # 'Expectancy',
                'Sharpe Ratio',
                # 'Calmar Ratio',
                'Omega Ratio',
                'Sortino Ratio',
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
            take_profit_points=dict(type='take_profit', lower_bound=0, upper_bound=1000, min_step=1),
            stoploss_points=dict(type='stop_loss', lower_bound=0, upper_bound=1000, min_step=1),
        ),
        # strategy_user_picked_params

        strategy_user_picked_params=dict(
            output_file_name='backtest_result.csv',
            # if compute_product then will compute the product of all the parameter values passed,
            #   else parameter values length must be equal
            compute_product=False,
            #
            # Can Read Parameters from file instead if the path to it is provided
            # read_user_defined_param_file='backtest_result.csv',
            read_user_defined_param_file=None,  # todo: logic missing
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

# loss_settings=dict(
#                 total_profit=dict(
#                     total_profit_weight=1,
#                     expected_total_profit=1,
#                     total_profit_threshold_high=400,
#                     total_profit_threshold_low=0,
#                 ),
#                 total_trades=dict(
#                     total_trades_weight=0,
#                     expected_total_trades=200,
#                     total_trades_threshold_high=2000,
#                     total_trades_threshold_low=20,
#                 ),
#                 win_ratio=dict(
#                     win_ratio_weight=0,
#                     expected_win_ratio=80,
#                     win_ratio_threshold_high=90,
#                     win_ratio_threshold_low=40,
#                 ),
#                 risk_reward_ratio=dict(
#                     risk_reward_ratio_weight=0,
#                     expected_risk_reward_ratio=1.5,
#                     risk_reward_ratio_threshold_high=5,
#                     risk_reward_ratio_threshold_low=1,
#                 ),
#                 risk_adjusted_return=dict(
#                     risk_adjusted_return_weight=1,
#                     expected_risk_adjusted_return=0.05,
#                     risk_adjusted_return_threshold_high=1,
#                     risk_adjusted_return_threshold_low=0,
#                 ),
#                 profit_factor=dict(
#                     profit_factor_weight=0,
#                     expected_profit_factor=1.5,
#                     profit_factor_threshold_high=5,
#                     profit_factor_threshold_low=1,
#                 ),
#                 sortino_ratio=dict(
#                     sortino_ratio_weight=0,
#                     expected_sortino_ratio=1.5,
#                     sortino_ratio_threshold_high=10000000,
#                     sortino_ratio_threshold_low=1,
#                 ),
#                 omega_ratio=dict(
#                     omega_ratio_weight=1,
#                     expected_omega_ratio=1.5,
#                     omega_ratio_threshold_high=10,
#                     omega_ratio_threshold_low=1,
#                 ),
#                 average_profit=dict(
#                     average_profit_weight=0,
#                     expected_average_profit=0.3,
#                     average_profit_threshold_high=12,
#                     average_profit_threshold_low=0.1,
#                 ),
#                 average_trade_duration=dict(
#                     average_trade_duration_weight=0,
#                     expected_average_trade_duration=1440,
#                     average_trade_duration_threshold_high=1440 * 3,
#                     average_trade_duration_threshold_low=5,
#                 ),
#                 max_drawdown=dict(
#                     max_drawdown_weight=0,
#                     expected_max_drawdown=6,
#                     max_drawdown_threshold_high=60,
#                     max_drawdown_threshold_low=0,
#                 ),
#             ),
