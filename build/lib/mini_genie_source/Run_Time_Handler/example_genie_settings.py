config_template = """dict(
    # Data Settings
    Data_Settings=dict(
        load_CSV_from_pickle=True,  # momentary
        data_files_dir='Datas',  # momentary
        data_files_names=data_files_names,  # momentary

        delocalize_data=True,
        drop_nan=False,
        ffill=False,
        fill_dates=False,
        saved_data_file='SymbolData',
        tick_size=tick_size,
        minute_data_input_format="%m.%d.%Y %H:%M:%S",
        minute_data_output_format="%m.%d.%Y %H:%M:%S",
        #
        accompanying_tick_data_input_format="%m.%d.%Y %H:%M:%S.%f",
        accompanying_tick_data_output_format="%m.%d.%Y %H:%M:%S.%f",
        data_randomness=this_elsethis(data_randomness,None),
        #     2021-10-03 22:04:00
    ),

    Simulation_Settings=dict(
        study_name=study_name,
        optimization_period=dict(
            start_date=start_date,  # 01.03.2022
            end_date=end_date  # 07.27.2022
            # end_date=datetime.datetime(month=10, day=1, year=2021)
        ),
        #
        timer_limit=this_elsethis(timer_limit,datetime.timedelta(days=365, hours=0, minutes=0, seconds=0)) ,  
        Continue=False,
        run_mode="plaid_plus",  # todo: ["ludicrous","plaid_plus"]
        #
        batch_size=this_elsethis(batch_size,int(10)),
        save_every_nth_chunk=1,
        Initial_Search_Space=dict(
            path_of_initial_metrics_record='saved_param_metrics.csv',
            path_of_initial_params_record='saved_initial_params.csv',
            max_initial_combinations=this_elsethis(max_initial_combinations,int(1000)),
            stop_after_n_epoch=this_elsethis(stop_after_n_epoch,int(200)),
            #
            parameter_selection=dict(
                timeframes='all',  # todo: needs to add settings for how to reduce, these dont do anything
                windows='grid',  # todo: needs to add settings for how to reduce, these dont do anything
                tp_sl=dict(
                    bar_atr_days=datetime.timedelta(days=90, hours=0, minutes=0, seconds=0),
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
            Strategy=this_elsethis(Strategy,None),
            backtesting="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Backtest",
            optimization="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Optimization",
        ),
        #
        sim_timeframe='1m',
        JustLoadpf=False,
        slippage=0,  # 0.0001,
        max_spread_allowed=this_elsethis(max_spread_allowed,np.inf), 
        trading_fees=this_elsethis(trading_fees,int(0.00005)),  # 0.00005 or 0.005%, $5 per $100_000
        cash_sharing=False,
        group_by=[],  # Leave blank
        max_orders=this_elsethis(max_orders,int(1)),
        init_cash=this_elsethis(init_cash,int(1_000_000)),
        size_type='cash',  # 'shares',  # cash or shares
        size=this_elsethis(size,int(100_000)),  # cash, else set size type to shares for share amount
        type_percent=False,  # if true then take_profit and stop_loss are given in percentages, else cash amount
    ),
    # It faster when values given, if not pass 'auto' and I will do my best
    RAY_SETTINGS=dict(
        ray_init_num_cpus=this_elsethis(ray_init_num_cpus,cpu_count()-4),
        simulate_signals_num_cpus=this_elsethis(simulate_signals_num_cpus,cpu_count()-4),
    )
)
"""
