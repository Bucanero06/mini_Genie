import datetime

Run_Time_Settings = dict(
    # Data Settings
    Data_Settings=dict(
        load_CSV_from_pickle=True,  # momentary
        data_files_dir='Datas/Forex Majors/',  # momentary
        data_files_names=[
            # 'AUDUSD',  # momentary
            # 'EURUSD',  # momentary
            'GBPUSD',  # momentary
            # 'NZDUSD',  # momentary
            # 'USDCAD',  # momentary
            # 'USDCHF',  # momentary
        ],  # momentary
        #
        delocalize_data=True,
        drop_nan=False,
        ffill=False,
        fill_dates=False,
        saved_data_file='SymbolData',
        tick_size=0.00001
    ),

    Optimization_Settings=dict(
        study_name='test_name',
        optimization_period=dict(
            start_date=datetime.datetime(month=8, day=1, year=2021),
            end_date=datetime.datetime(month=1, day=1, year=2022)
        ),
        timer_limit=datetime.timedelta(days=0, hours=7, minutes=0, seconds=0),  # fixme: logic missing
        Continue=False,

        Initial_Search=dict(
            # max_initial_combinations=1_000_000,
            max_initial_combinations=1000,
            initial_batch_size=2,
            force_to_finish=True,  # fixme: logic missing
            #
            parameter_selection=dict(
                timeframes='all',  # fixme: needs to add settings for how to reduce, these dont do anything
                windows='grid',  # fixme: needs to add settings for how to reduce, these dont do anything
                tp_sl=dict(
                    bar_atr_days=datetime.timedelta(days=60, hours=0, minutes=0, seconds=0),
                    bar_atr_periods=[14],  # todo multiple inputs
                    bar_atr_multiplier=[3],  # todo multiple inputs
                    #
                    n_ratios=[0.5, 1, 1.5, 2, 2.5],
                    gamma_ratios=[1, 1.2, 1.5, 1.75, 2, 2.5, 3],
                    number_of_bar_trends=1,
                ),

            ),

        ),

        Loss_Function=dict(
            metrics=[
                'Risk/Reward Ratio',
                'Risk Adjusted Return',
                'Total Trades',
                'Win Rate [%]',
                'Profit Factor',
                'Sortino Ratio',
                'Omega Ratio',
                'Total Return [%]',
                'Max Drawdown [%]',
                # 'Max Gross Exposure [%]'
            ],
            loss_settings=dict(
                total_profit=dict(
                    total_profit_weight=1,
                    expected_total_profit=1,
                    total_profit_threshold_high=400,
                    total_profit_threshold_low=0,
                ),
                total_trades=dict(
                    total_trades_weight=0,
                    expected_total_trades=200,
                    total_trades_threshold_high=2000,
                    total_trades_threshold_low=20,
                ),
                win_ratio=dict(
                    win_ratio_weight=0,
                    expected_win_ratio=80,
                    win_ratio_threshold_high=90,
                    win_ratio_threshold_low=40,
                ),
                risk_reward_ratio=dict(
                    risk_reward_ratio_weight=0,
                    expected_risk_reward_ratio=1.5,
                    risk_reward_ratio_threshold_high=5,
                    risk_reward_ratio_threshold_low=1,
                ),
                risk_adjusted_return=dict(
                    risk_adjusted_return_weight=1,
                    expected_risk_adjusted_return=0.05,
                    risk_adjusted_return_threshold_high=1,
                    risk_adjusted_return_threshold_low=0,
                ),
                profit_factor=dict(
                    profit_factor_weight=0,
                    expected_profit_factor=1.5,
                    profit_factor_threshold_high=5,
                    profit_factor_threshold_low=1,
                ),
                sortino_ratio=dict(
                    sortino_ratio_weight=0,
                    expected_sortino_ratio=1.5,
                    sortino_ratio_threshold_high=10000000,
                    sortino_ratio_threshold_low=1,
                ),
                omega_ratio=dict(
                    omega_ratio_weight=1,
                    expected_omega_ratio=1.5,
                    omega_ratio_threshold_high=10,
                    omega_ratio_threshold_low=1,
                ),
                average_profit=dict(
                    average_profit_weight=0,
                    expected_average_profit=0.3,
                    average_profit_threshold_high=12,
                    average_profit_threshold_low=0.1,
                ),
                average_trade_duration=dict(
                    average_trade_duration_weight=0,
                    expected_average_trade_duration=1440,
                    average_trade_duration_threshold_high=1440 * 3,
                    average_trade_duration_threshold_low=5,
                ),
                max_drawdown=dict(
                    max_drawdown_weight=0,
                    expected_max_drawdown=6,
                    max_drawdown_threshold_high=60,
                    max_drawdown_threshold_low=0,
                ),
            ),
        ),
        #
        Optuna_Study=dict(
            sampler_name=None,
            multi_objective_bool=None, )

    ),
    Simulation_Settings=dict(
        Portfolio_Settings=dict(
            # Simulation Settings
            JustLoadpf=False,
            saved_pf_backtest='My_pf_backtest',
            saved_pf_optimization='My_pf_optimization',
            slippage=0,  # 0.0001,
            trading_fees=0.,  # 0.0000005,
            cash_sharing=False,
            group_by=[],  # Leave blank

            # Strategy
            # max_orders=-1,
            init_cash=1_000_000,
            size_type='cash',  # 'shares',  # cash or shares
            size=1_000,  # cash, else set size type to shares for share amount
            type_percent=False,  # if true then take_profit and stoploss are given in percentages, else cash amount

        ),
    ),
    Strategy_Settings=dict(
        # Strategy_Name=MMT_Strategy, FIXME: uncomment and import MMT_Strategy
        arguements=dict(
            fixed_initial_params=[
            ],
            optional_initial_params_to_product=dict(
                PEAK_and_ATR_timeframes=['1 min', '5 min', '15 min', '30 min', '1h', '4h', '1d'],
                # np.random.choice(['5 min', '15 min', '30 min', '1h', '4h', '1d'], size=size_of_test_batch),
                atr_windows=[2],  # np.random.randint(1, 30, size_of_test_batch),
                data_lookback_windows=[16],  # np.random.randint(1, 30, size_of_test_batch),

                EMAs_timeframes=['1 min', '5 min', '15 min', '30 min', '1h', '4h', '1d'],
                # np.random.choice(['1 min', '5 min', '15 min', '30 min', '1h', '4h'], size=size_of_test_batch),
                ema_1_windows=[20],  # np.random.randint(5, 35, size_of_test_batch),
                ema_2_windows=[30],  # np.random.randint(35, 65, size_of_test_batch),
                ema_3_windows=[19],  # np.random.randint(19, 49, size_of_test_batch),
                take_profit_points=[170.0],  # np.random.randint(20, 700, size_of_test_batch),
                #
                stoploss_points=[-90.0],  # np.random.randint(20, 700, size_of_test_batch),
            ),
        ),
        parameter_windows=dict(
            PEAK_and_ATR_timeframes=dict(type='timeframe',
                                         choices=['5 min', '15 min', '30 min', '1h', '4h', '1d']),

            atr_windows=dict(type='window', lower_bound=1, upper_bound=10, min_step=1),
            data_lookback_windows=dict(type='window', lower_bound=2, upper_bound=16, min_step=1),
            EMAs_timeframes=dict(type='timeframe', choices=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
            ema_1_windows=dict(type='window', lower_bound=5, upper_bound=45, min_step=1),
            ema_2_windows=dict(type='window', lower_bound=20, upper_bound=60, min_step=1),
            ema_3_windows=dict(type='window', lower_bound=19, upper_bound=20, min_step=1),
            #
            take_profit_points=dict(type='take_profit', lower_bound=0, upper_bound=1000, min_step=1),
            stoploss_points=dict(type='stop_loss', lower_bound=0, upper_bound=1000, min_step=1),
        )
    ),

)
