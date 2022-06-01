def simulate_suggestions(self):
    """
    In chunks/batches:
       1.  Simulate N parameters' indicators
       2.  Simulate N parameters' events
       3.  Compute Metrics
       4.  Save Results to file
    """
    logger.info("in simulate suggestions")

    def _analyze_n_save(portfolio, params_rec_id, highest_profit_, best_parameters_, initial_cash_total_, epoch_n_,
                        save_every_nth_chunk=None):
        '''Reconstruct Metrics from Order Records and Save'''
        logger.info(f"Preparing to Save epoch {epoch_n_}")

        tell_metrics_start_timer = perf_counter()
        #
        params_rec = ray.get(params_rec_id)
        #
        # Used for Printing
        highest_profit_this_epoch = portfolio['Total Return [%]'].max()
        highest_cash_profit_this_epoch = highest_profit_this_epoch * initial_cash_total_ / 100
        best_parameters_this_epoch = portfolio['Total Return [%]'].idxmax()
        #
        if highest_cash_profit_this_epoch > highest_profit_:
            highest_profit_ = highest_cash_profit_this_epoch
            best_parameters_ = best_parameters_this_epoch
        #
        logger.info(f'Highest Profit so far: {highest_profit_}   \N{money-mouth face}\N{money bag}')
        logger.info(f'Best Param so far: {best_parameters_}  \N{money with wings}')
        #
        logger.info(f' -> highest_profit this epoch {highest_cash_profit_this_epoch} ')
        logger.info(f' -> best_param this epoch {best_parameters_this_epoch}')
        #
        # clean up porfolio
        portfolio.fillna(0.0)
        # portfolio.replace(pd.NaT, pd.Timedelta(seconds=0), inplace=True)
        #
        # Fill metric record with new metric values
        for _index, param_record in enumerate(params_rec):
            trial_id = param_record["trial_id"]
            param_record = rm_field_from_record(param_record, 'trial_id')
            #
            for ass_index, asset in enumerate(self.asset_names):
                param_tuple_ = tuple(param_record)
                param_tuple_ = param_tuple_ + (asset,)
                metrics_np = tuple(portfolio[:].loc[tuple(param_tuple_)])
                self.metrics_record[trial_id + (self.parameters_record_length * ass_index)] = (trial_id,
                                                                                               asset) + metrics_np
        logger.info(f'Time to Analyze Metrics {perf_counter() - tell_metrics_start_timer}')
        #
        # Concat and save the parameter and metric records to file every Nth epoch
        if save_every_nth_chunk:
            if epoch_n_ % save_every_nth_chunk == 0:
                logger.info(f"Saving epoch {epoch_n_}")
                save_start_timer = perf_counter()
                self._save_computed_params_metrics()
                #
                logger.info(f'Time to Save Records {perf_counter() - save_start_timer} during epoch {epoch_n_}')
        #
        return highest_profit_, best_parameters_

    batch_size = self.batch_size
    # self.parameters_record_length = len(ray.get(self.parameters_record))
    self.parameters_record_length = len(self.parameters_record)
    #
    from Simulation_Handler.simulation_handler import Simulation_Handler
    from Analysis_Handler.analysis_handler import Analysis_Handler
    simulation_handler = Simulation_Handler(self)
    analysis_handler = Analysis_Handler(self)
    #
    from Utilities.general_utilities import put_objects_list_to_ray
    simulation_handler_id, analysis_handler_id = put_objects_list_to_ray([simulation_handler, analysis_handler])
    # simulation_handler_id, analysis_handler_id = simulation_handler, analysis_handler
    #
    highest_profit = -sys.maxsize
    best_parameters = None
    #
    initial_cash_total = self.runtime_settings["Portfolio_Settings.init_cash"]
    stop_after_n_epoch = self.runtime_settings["Simulation_Settings.Initial_Search.stop_after_n_epoch"]
    save_every_nth_chunk = self.runtime_settings["Simulation_Settings.save_every_nth_chunk"]
    #
    # If metrics record empty then initiate
    if not any(self.metrics_record):
        self._initiate_metric_records(add_ids=True, params_size=self.parameters_record_length * len(
            self.asset_names))  # we need n_assets as many metric elements as there are trial.params
    else:
        highest_profit = np.max(self.metrics_record["Total Return [%]"])

        # self.parameters_record[]

    # Get an array of indexes remaining to compute
    from Utilities.general_utilities import fetch_non_filled_elements_indexes
    # Since metric_record is n_assets times bigger than parameters_record,and because metrics record just
    #   repeats every 1/n_assets of the array we only need the first portion it
    trials_ids_not_computed = fetch_non_filled_elements_indexes(self.metrics_record[:self.parameters_record_length])

    # Take elements from parameter record that match with trials_ids_not_computed
    # params_left_to_compute = np.take(ray.get(self.parameters_record), trials_ids_not_computed)
    params_left_to_compute = np.take(self.parameters_record, trials_ids_not_computed)
    #
    # Get max n_chunks given max batch_size
    n_chunks = int(np.floor(len(params_left_to_compute) / batch_size)) if batch_size < len(
        params_left_to_compute) else 1
    # Split arrays into n_chunks
    chunks_of_params_left_to_compute = np.array_split(params_left_to_compute, n_chunks)
    #
    from Utilities.general_utilities import put_objects_list_to_ray
    chunks_ids_of_params_left_to_compute = put_objects_list_to_ray(chunks_of_params_left_to_compute)

    # for epoch_n, epoch_params_record_id in enumerate(chunks_of_params_left_to_compute):
    for epoch_n, epoch_params_record in enumerate(chunks_of_params_left_to_compute):
        if epoch_n == stop_after_n_epoch:
            break
        #
        start_time = perf_counter()
        CHECKTEMPS(TEMP_DICT)
        #
        long_entries, long_exits, short_entries, short_exits, \
        strategy_specific_kwargs = simulation_handler.simulate_signals(epoch_params_record)
        #
        pf, extra_sim_info = simulation_handler.simulate_events(long_entries, long_exits,
                                                                short_entries, short_exits,
                                                                strategy_specific_kwargs)
        #
        '''Reconstruct Metrics from Order Records and Save'''
        portfolio_combined = analysis_handler.compute_stats(pf, self.metrics_key_names, groupby=self.group_by)
        analyze_start_timer = perf_counter()
        metrics_this_epoch = portfolio_combined.to_numpy()
        #
        # Fill metric record with new metric values
        for _index, overall_index in enumerate(epoch_params_record["trial_id"]):
            self.metrics_record[overall_index] = tuple(np.insert(metrics_this_epoch[_index], 0, overall_index))
        #
        # Used for Printing
        highest_profit_this_epoch = portfolio_combined['Total Return [%]'].max()
        highest_cash_profit_this_epoch = highest_profit_this_epoch * initial_cash_total / 100
        best_parameters_this_epoch = portfolio_combined['Total Return [%]'].idxmax()
        #
        if highest_cash_profit_this_epoch > highest_profit:
            highest_profit = highest_cash_profit_this_epoch
            best_parameters = best_parameters_this_epoch
        #
        logger.info(f'Highest Profit so far: {highest_profit}   \N{money-mouth face}\N{money bag}')
        logger.info(f'Best Param so far: {best_parameters}  \N{money with wings}')
        #
        logger.info(f' -> highest_profit this epoch {highest_cash_profit_this_epoch} ')
        logger.info(f' -> best_param this epoch {best_parameters_this_epoch}')
        #
        logger.info(f'Time to Analyze Metrics {perf_counter() - analyze_start_timer}')
        #
        # Concat and save the parameter and metric records to file every Nth epoch
        if save_every_nth_chunk:
            if epoch_n % save_every_nth_chunk == 0:
                save_start_timer = perf_counter()
                self._save_computed_params_metrics()
                #
                logger.info(f'Time to Save Records {perf_counter() - save_start_timer} during epoch {epoch_n}')
        #
        logger.info(f'Epoch {epoch_n} took {perf_counter() - start_time} seconds')
        logger.info(f'\n\n')
    #
    # Do a final save !
    self._save_computed_params_metrics()