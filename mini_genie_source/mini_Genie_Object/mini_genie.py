import os.path

import numpy as np
import ray
import vectorbtpro as vbt
from logger_tt import logger


class mini_genie_trader:
    """
    genie_object to be acted on

    Args:
        symbol_data (vbt.Data):  symbol_data object from vbt
        ...
    Attributes:
        open_df (df or series):  price during opening of candle
        ...
    Note:
        ...
    """

    def __init__(self, runtime_kwargs):
        """Constructor for mini_genie_trader"""

        # for key, value in runtime_kwargs.items():
        #     print(key)
        #     setattr(self, key, value)
        # Init Ray
        # ray.init(num_cpus=24)

        self.status = '__init__'
        import flatdict
        self.runtime_settings = flatdict.FlatDict(runtime_kwargs, delimiter='.')

        # Define frequently used variables in first level
        self.study_name = self.runtime_settings["Optimization_Settings.study_name"]
        #
        self.initial_batch_size = self.runtime_settings["Optimization_Settings.Initial_Search.initial_batch_size"]
        #
        self.parameter_windows = self.runtime_settings["Strategy_Settings.parameter_windows"]._values
        #
        self.key_names = tuple(self.parameter_windows.keys())
        #
        self.group_by = [f'custom_{key_name}' for key_name in self.key_names]
        #
        self.metrics_key_names = self.runtime_settings['Optimization_Settings.Loss_Function.metrics']
        self.loss_metrics_settings_dict = self.runtime_settings['Optimization_Settings.Loss_Function.loss_settings']
        number_of_outputs = 0
        for metric_name in self.loss_metrics_settings_dict._values:
            if self.loss_metrics_settings_dict[f'{metric_name}.{metric_name}_weight'] != 0:
                number_of_outputs += 1
        self.number_of_outputs = number_of_outputs
        #
        self.optimization_start_date = self.runtime_settings['Optimization_Settings.optimization_period.start_date']
        self.optimization_end_date = self.runtime_settings['Optimization_Settings.optimization_period.end_date']
        #
        # Miscellaneous
        from datetime import datetime
        self.stop_sim_time = self.runtime_settings['Optimization_Settings.timer_limit'] + datetime.now()
        self.ACCEPTED_TIMEFRAMES = ['1 min', '5 min', '15 min', '30 min', '1h', '4h', '1d']
        self.ACCEPTED_TF_TYPES = ['timeframe', 'tf']
        self.ACCEPTED_TP_TYPES = ['take_profit', 'tp']
        self.ACCEPTED_SL_TYPES = ['stop_loss', 'sl']
        self.ACCEPTED_TP_SL_TYPES = self.ACCEPTED_TP_TYPES + self.ACCEPTED_SL_TYPES

        self.ACCEPTED_WINDOW_TYPES = ['window', 'w']
        #
        self.window_key_names = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_WINDOW_TYPES)
        self.timeframe_keynames = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_TF_TYPES)
        self.tp_sl_keynames = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_TP_SL_TYPES)
        self.tp_keynames = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_TP_TYPES)
        self.sl_keynames = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_SL_TYPES)
        #
        self.tp_sl_selection_space = self.runtime_settings[
            "Optimization_Settings.Initial_Search.parameter_selection.tp_sl"]
        # Initial Actions
        #
        # Prepare directories and save file paths
        self._prepare_directory_paths_for_study()
        #
        # Load precomputed params, values, and stats if continuing study
        self._load_precomputed_params()

    def print_dict(self, optional_object: object = None) -> object:
        """

        Returns:
            object:
        """
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    def _prepare_directory_paths_for_study(self) -> object:
        """
        Returns:
            object:

        """

        def CreateDir(dir: object) -> object:
            """

            Args:
                dir:

            Returns:
                object:
            """
            from os import path, mkdir
            if not path.exists(dir):
                logger.info(f'Creating directory {dir}')
                mkdir(dir)
            else:
                logger.info(f'Found {dir}')

        logger.debug('''Create Folders if needed''')
        studies_directory = 'Studies'
        study_dir_path = f'{studies_directory}/Study_{self.study_name}'
        saved_runs_path = f'{study_dir_path}/saved_parameter_history'
        portfolio_dir_path = f'{study_dir_path}/Portfolio'
        reports_dir_path = f'{study_dir_path}/Reports'
        data_dir_path = f'Datas'
        #
        CreateDir(studies_directory)
        CreateDir(study_dir_path)
        CreateDir(portfolio_dir_path)
        CreateDir(reports_dir_path)
        CreateDir(data_dir_path)
        CreateDir(f'Strategies')

        self.study_path = study_dir_path
        self.saved_runs_path = saved_runs_path
        self.portfolio_path = portfolio_dir_path
        self.reports_path = reports_dir_path
        self.data_path = data_dir_path

    def _load_precomputed_params(self) -> object:
        """
        If continuing a previously started study, then we can assume we have computed some parameters already.
        The parameter combinations, trial_number, values and stats should have been saved to a file.
        Here we only want to load the necessary, thus load trial number (to be used as index),
        trial params, and trial objective value. Place this object in ray to be referred back to already
        computed params and values.

        Returns:
            object:
        """
        import pandas as pd
        def _write_header_df(path: object, cols_names: object, compression: object = 'gzip') -> object:
            header_df = pd.DataFrame(columns=cols_names)
            header_df.to_parquet(path, compression=compression)
            return header_df

        #
        header_df = pd.DataFrame()
        saved_parameter_history = pd.DataFrame()
        #
        column_names = list(self.key_names) + self.runtime_settings['Optimization_Settings.Loss_Function.metrics']
        #
        # If does not exist, create file, write file head, and save path to paths place to ray
        if not os.path.exists(f'{self.saved_runs_path}.gzip'):
            header_df = _write_header_df(f'{self.saved_runs_path}.gzip', column_names)
        #
        # else, load file and place to ray
        elif not self.runtime_settings["Optimization_Settings.Continue"]:
            os.remove(f'{self.saved_runs_path}.gzip')
            header_df = _write_header_df(f'{self.saved_runs_path}.gzip', column_names)
        else:
            saved_parameter_history = pd.read_parquet(f'{self.saved_runs_path}.gzip')

        # fixme: self.previously_computed_parameters_df = ray.put(previously_computed_parameters_df if not header_df
        #  else header_df)
        self.saved_parameter_history = ray.put(
            saved_parameter_history if not saved_parameter_history.empty else header_df)
        # self.saved_parameter_history = saved_parameter_history if header_df.empty else header_df

    # todo
    def _save_newly_computed_params(self):
        """Add to previously created file (or new if it does not exist), in a memory conscious way, the trial number,
        parameter combination, values, and combination stats"""
        # TODO:
        #   code _save_newly_computed_params
        #   Im assuming that it will be similar to  _load_precomputed_params except now we are adding the newly computed
        #   parameters to the saved parameters and saving them as gzip (expected to get very large)
        ...

    # todo
    def _grade_parameter_combinations(self):
        """Use a loss function to give a value to each parameter combination,..."""
        # TODO:
        #   code _grade_parameter_combinations
        # TODO:
        #   How should we grade them? For now I think we get stats for as many parameter combinations
        #       as possible, then we can group strategies to determine the most stable regions, neighbors
        #       etc ...

    def fetch_and_prepare_input_data(self) -> object:
        self.status = 'fetch_and_prepare_input_data'

        from mini_genie_source.Data_Handler.data_handler import Data_Handler
        data_processing = Data_Handler(self).fetch_data()  # Load symbols_data (attr)
        data_processing.break_up_olhc_data_from_symbols_data()  # splits ^ into open, low, high, close, *alt (attrs)

    def _initiate_parameters_records(self, add_ids: object = None) -> object:
        def _total_possible_values_in_window(lower: object, upper: object, step: object) -> object:
            return int((upper - lower) / step)

        def _reduce_initial_parameter_space(lengths_dict: object, max_initial_combinations: object) -> object:
            """
            Reduce initial parameter space combinations by reducing the number of param suggestions:
                1. For TP and SL parameters
                2. For Windowed parameters
            Output:
                n_initial_combinations to be used in the construction of the initial parameters record

            Args:
                lengths_dict:
                max_initial_combinations:

            Returns:
                object:
            """
            max_initial_combinations_original = max_initial_combinations

            def _compute_n_initial_combinations_carefully(dict: object) -> object:  # fixme: naming is horrible
                """

                Args:
                    dict:

                Returns:

                """
                n_reduced_lengths = [dict[f'{key_name}_length'] for key_name in self.key_names
                                     if self.parameter_windows[key_name][
                                         'type'].lower() not in self.ACCEPTED_TP_SL_TYPES]
                n_reduced_lengths.append(dict["tp_sl_length"])

                return np.product(n_reduced_lengths)

            def _compute_windows_lengths_now(dict: object, _keynames: object = None) -> object:
                """

                Args:
                    dict:
                    _keynames:

                Returns:
                    object:

                """
                _keynames = self.key_names if not _keynames else _keynames

                return np.product(
                    [dict[f'{key_name}_length'] for key_name in _keynames if
                     self.parameter_windows[key_name]['type'].lower() in self.ACCEPTED_WINDOW_TYPES])

            # if n_initial_combinations > max_initial_combinations: reduce initial search space
            if lengths_dict["n_initial_combinations"] > max_initial_combinations:
                # First try the reduced TP and SL space

                lengths_dict['tp_sl_length'] = len(self.tp_sl_selection_space["n_ratios"]) * len(
                    self.tp_sl_selection_space["gamma_ratios"]) * self.tp_sl_selection_space[
                                                   "number_of_bar_trends"] * len(
                    lengths_dict["all_tf_in_this_study"])
                #
                lengths_dict["n_initial_combinations"] = _compute_n_initial_combinations_carefully(lengths_dict)
                lengths_dict["using_reduced_tp_sl_space"] = True
            #

            if lengths_dict["n_initial_combinations"] > max_initial_combinations:
                # Second try the reduced windowed space
                temp_lengths_dict = dict(
                    windows_product=1,
                    window_keynames_to_be_reduced=[],
                )
                from copy import deepcopy
                for key_name in self.window_key_names:
                    if lengths_dict[f'{key_name}_length'] > 1:
                        temp_lengths_dict[f'{key_name}_length'] = deepcopy(lengths_dict[f'{key_name}_length'])
                        #
                        temp_lengths_dict[f'windows_product'] = temp_lengths_dict[f'windows_product'] * \
                                                                temp_lengths_dict[f'{key_name}_length']
                        #
                        temp_lengths_dict["window_keynames_to_be_reduced"].append(key_name)

                temp_lengths_dict["big_r_scaling_factor"] = lengths_dict[
                                                                "n_initial_combinations"] / max_initial_combinations
                temp_lengths_dict[f'max_windows_product'] = temp_lengths_dict[
                                                                f'windows_product'] / temp_lengths_dict[
                                                                "big_r_scaling_factor"]

                temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                        1 / len(temp_lengths_dict["window_keynames_to_be_reduced"]))

                # we_are_good = all(
                #     temp_lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"] for
                #     key_name in
                #     temp_lengths_dict["window_keynames_to_be_reduced"])
                # logger.info(f'{temp_lengths_dict["window_keynames_to_be_reduced"] = }')
                #                 logger.info(f'{we_are_good = }')
                #                 logger.info(f'{_compute_n_initial_combinations_carefully(temp_lengths_dict) = }')
                #
                #
                #                 we_are_good=_compute_n_initial_combinations_carefully(temp_lengths_dict)

                # Refine small_r_scaling_factor
                we_are_good = False
                temp_lengths_dict = dict(
                    big_r_scaling_factor=temp_lengths_dict["big_r_scaling_factor"],
                    small_r_scaling_factor=temp_lengths_dict["small_r_scaling_factor"],
                    max_windows_product=temp_lengths_dict[f'max_windows_product'],
                    windows_product=1,
                    n_windows_to_be_reduced=0,
                    window_keynames_to_be_reduced=[],
                    window_keynames_above_1=temp_lengths_dict["window_keynames_to_be_reduced"],
                )
                while not we_are_good:
                    temp_lengths_dict = dict(
                        big_r_scaling_factor=temp_lengths_dict["big_r_scaling_factor"],
                        small_r_scaling_factor=temp_lengths_dict["small_r_scaling_factor"],
                        max_windows_product=temp_lengths_dict[f'max_windows_product'],
                        windows_product=1,
                        n_windows_to_be_reduced=0,
                        window_keynames_to_be_reduced=[],
                        window_keynames_above_1=temp_lengths_dict["window_keynames_above_1"],
                    )
                    #
                    for key_name in temp_lengths_dict["window_keynames_above_1"]:
                        if lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"]:
                            temp_lengths_dict[f'{key_name}_length'] = deepcopy(lengths_dict[f'{key_name}_length'])
                            #
                            temp_lengths_dict[f"windows_product"] = temp_lengths_dict[f'windows_product'] * \
                                                                    temp_lengths_dict[f'{key_name}_length']
                            #
                            temp_lengths_dict["window_keynames_to_be_reduced"].append(key_name)
                        #
                    if temp_lengths_dict["window_keynames_to_be_reduced"]:
                        #
                        temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                                1 / len(temp_lengths_dict["window_keynames_to_be_reduced"]))
                        #
                        we_are_good = all(
                            temp_lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"]
                            for key_name in
                            temp_lengths_dict["window_keynames_to_be_reduced"])
                        #
                    else:
                        max_initial_combinations = max_initial_combinations + (max_initial_combinations * 0.01)
                        we_are_good = False
                        #
                        temp_lengths_dict["big_r_scaling_factor"] = lengths_dict[
                                                                        "n_initial_combinations"] / max_initial_combinations
                        temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                                1 / len(temp_lengths_dict["window_keynames_above_1"]))
                        temp_lengths_dict[f'max_windows_product'] = temp_lengths_dict[f'windows_product'] / \
                                                                    temp_lengths_dict["big_r_scaling_factor"]
                        #

                # Scale down lengths
                for key_name in temp_lengths_dict["window_keynames_to_be_reduced"]:
                    #
                    temp_value = lengths_dict[f'{key_name}_length'] / temp_lengths_dict["small_r_scaling_factor"]
                    if temp_value < 1:
                        temp_lengths_dict[f'{key_name}_length'] = 1
                    else:
                        temp_lengths_dict[f'{key_name}_length'] = temp_value
                #
                # Redefine window lengths in length_dict
                for key_name in temp_lengths_dict["window_keynames_to_be_reduced"]:
                    # if self.parameter_windows[key_name]['type'].lower() in self.ACCEPTED_WINDOW_TYPES:
                    lengths_dict[f'{key_name}_length'] = int(temp_lengths_dict[f'{key_name}_length'])

                lengths_dict["using_reduced_window_space"] = True
                temp_lengths_dict["windows_product"] = _compute_windows_lengths_now(lengths_dict,
                                                                                    _keynames=
                                                                                    temp_lengths_dict[
                                                                                        "window_keynames_to_be_reduced"])

            lengths_dict["n_initial_combinations"] = _compute_n_initial_combinations_carefully(lengths_dict)
            #
            return lengths_dict

        parameters_record_dtype = []

        # Keep track of miscellaneous settings for reducing the space
        parameters_lengths_dict = dict(
            using_reduced_tp_sl_space=False,
            using_reduced_window_space=False,
            n_total_combinations=0,
            n_initial_combinations=0,
            all_tf_in_this_study=[],
            tp_sl_length=1,
        )

        if add_ids:
            parameters_record_dtype.append(('trial_id', np.int_))
        for key_name in self.key_names:
            parameters_lengths_dict[f'{key_name}_length'] = 0

            if self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_TF_TYPES:
                tf_in_key_name = [tf.lower() for tf in self.parameter_windows[key_name]['choices']]

                if set(tf_in_key_name).issubset(set(self.ACCEPTED_TIMEFRAMES)):
                    parameters_record_dtype.append((key_name, 'U8'))
                else:
                    erroneous_timeframes = [tf for tf in tf_in_key_name if tf not in self.ACCEPTED_TIMEFRAMES]
                    logger.error(
                        f'These timeframes provided are not accepted or understood, please revise.\n'
                        f'      {erroneous_timeframes = }.\n'
                        f'      {self.ACCEPTED_TIMEFRAMES = }.\n'
                        f'      {tf_in_key_name = }.\n'
                    )

                parameters_lengths_dict[f'{key_name}_length'] = len(tf_in_key_name)
                parameters_lengths_dict["all_tf_in_this_study"].extend(tf_in_key_name)

            elif self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_WINDOW_TYPES:
                if isinstance(self.parameter_windows[key_name]['lower_bound'], int) and isinstance(
                        self.parameter_windows[key_name]['upper_bound'], int):
                    parameters_record_dtype.append((key_name, 'i8'))
                elif isinstance(self.parameter_windows[key_name]['lower_bound'], float) or isinstance(
                        self.parameter_windows[key_name]['upper_bound'], float):
                    parameters_record_dtype.append((key_name, 'f8'))
                else:
                    logger.error(f'Parameter {key_name} is defined as type window but inputs are inconsistent.\n'
                                 f'     (e.g. -> Either lower_bound or upper_bound is a float => float)\n'
                                 f'     (e.g. -> both lower_bound or upper_bound are integers => int)\n'
                                 f' upper_bound type:   {type(self.parameter_windows[key_name]["upper_bound"])}'
                                 f' lower_bound type:   {type(self.parameter_windows[key_name]["lower_bound"])}'
                                 )
                    exit()
                parameters_lengths_dict[f'{key_name}_length'] = _total_possible_values_in_window(
                    self.parameter_windows[key_name]["lower_bound"], self.parameter_windows[key_name]["upper_bound"],
                    self.parameter_windows[key_name]["min_step"])

            elif self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_TP_SL_TYPES:
                if isinstance(self.parameter_windows[key_name]['lower_bound'], int) and isinstance(
                        self.parameter_windows[key_name]['upper_bound'], int):
                    parameters_record_dtype.append((key_name, 'i8'))
                elif isinstance(self.parameter_windows[key_name]['lower_bound'], float) or isinstance(
                        self.parameter_windows[key_name]['upper_bound'], float):
                    parameters_record_dtype.append((key_name, 'f8'))
                #
                parameters_lengths_dict[f'{key_name}_length'] = _total_possible_values_in_window(
                    self.parameter_windows[key_name]["lower_bound"], self.parameter_windows[key_name]["upper_bound"],
                    self.parameter_windows[key_name]["min_step"])

                parameters_lengths_dict[f'tp_sl_length'] = parameters_lengths_dict[f'tp_sl_length'] * \
                                                           parameters_lengths_dict[f'{key_name}_length'] if \
                    parameters_lengths_dict[f'tp_sl_length'] else parameters_lengths_dict[f'{key_name}_length']

            else:
                logger.error(f'Parameter {key_name} is defined as type {self.parameter_windows[key_name]["type"]}'
                             f'but that type is not accepted ... yet ;)')
                exit()

        parameters_lengths_dict["all_tf_in_this_study"] = list(set(parameters_lengths_dict["all_tf_in_this_study"]))
        # Determine size of complete parameter space combinations with settings given as well reduced space
        parameters_lengths_dict["n_total_combinations"] = parameters_lengths_dict["n_initial_combinations"] = \
            np.product([parameters_lengths_dict[f'{key_name}_length'] for key_name in self.key_names])

        self.parameters_lengths_dict = _reduce_initial_parameter_space(parameters_lengths_dict, self.runtime_settings[
            "Optimization_Settings.Initial_Search.max_initial_combinations"])

        if self.parameters_lengths_dict["n_initial_combinations"] > self.runtime_settings[
            "Optimization_Settings.Initial_Search.max_initial_combinations"]:
            logger.warning(
                f'I know max_initial_combinations was set to '
                f'{self.runtime_settings["Optimization_Settings.Initial_Search.max_initial_combinations"]} '
                f'but, I needed at least {self.parameters_lengths_dict["n_initial_combinations"]} initial combinations'
                f"\N{smiling face with smiling eyes}"
            )

        self.parameter_data_dtype = np.dtype(parameters_record_dtype)
        self.parameters_record = np.empty(self.parameters_lengths_dict["n_initial_combinations"],
                                          dtype=self.parameter_data_dtype)
        #

    def _compute_params_product_n_fill_record(self, params):
        from itertools import product
        initial_param_combinations = list(
            set(
                product(
                    *[
                        params[key_name] for key_name in self.key_names if key_name not in self.tp_sl_keynames
                    ], params["tp_sl"]
                )
            )
        )

        #
        for i in range(len(initial_param_combinations)):
            value = (initial_param_combinations[i][:-1] + initial_param_combinations[i][-1])
            self.parameters_record[i] = value

    def _compute_bar_atr(self):
        from mini_genie_source.Simulation_Handler.compute_bar_atr import compute_bar_atr
        self.bar_atr = compute_bar_atr(self)

    @staticmethod
    def _expand_tp_sl_0(tp_sl_0, n_ratios, gamma_ratios, tick_size):
        """
        tp_sl_0: base
        n_ratios: you multiply the atr with
        gamma_ratios: risk reward ratio
        """
        tp = []
        sl = []

        for n in n_ratios:
            for gamma in gamma_ratios:
                adj_tp_sl_0 = tp_sl_0 * n
                diff = adj_tp_sl_0 * gamma
                #
                if gamma < 1:
                    tp.append(adj_tp_sl_0 / tick_size)
                    sl.append(-(adj_tp_sl_0 + diff) / tick_size)
                else:
                    tp.append((adj_tp_sl_0 + diff) / tick_size)
                    sl.append((-adj_tp_sl_0) / tick_size)
                #

        return tp, sl

    def _compute_tp_n_sl_from_tp_sl_0(self) -> object:
        """

        Returns:
            object:

        """
        result = {}
        for tf in self.parameters_lengths_dict[f'all_tf_in_this_study']:
            tp_sl_0 = self.bar_atr[tf]["mean_atr"]
            tp, sl = self._expand_tp_sl_0(tp_sl_0,
                                          self.tp_sl_selection_space["n_ratios"],
                                          self.tp_sl_selection_space["gamma_ratios"],
                                          self.runtime_settings["Data_Settings.tick_size"])
            result[tf] = dict(
                take_profits=tp,
                stop_losses=sl
            )

        return result

    def _initiate_metric_records(self):
        self.metric_data_dtype = np.dtype([(metric_name, 'U8') for metric_name in self.metrics_key_names])
        self.metrics_record = np.empty(self.parameters_lengths_dict["n_initial_combinations"],
                                       dtype=self.metric_data_dtype)

    # todo continue (everything)
    def suggest_parameters(self):
        """
          List of Initial Params:
               Product of:
                   All Categorical Params
                   Use a grid-like approach to windows for indicators
                TODO:
                   For TP and SL use the avg ATR for the 3 months prior to the optimization date window for every
                       timeframe (when possible do this separately for upwards, downwards and sideways, then use
                       these values separately during the strategy or average them for a single value) then:
                           1.  Using \bar{ATR}(TF), define multiple TP_0 and SL_0 for and scale with n ratios [ 0.5, 1, 1.5, 2, 2.5]
                           2.  Use (TP/SL) \gamma ratios like [ 1, 1.2, 1.5, 1.75, 2, 2.5, 3]
                               (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.5, -> R=\bar{ATR}(TF='1h')/n=500
                                   ==> TP=750 & SL=-500)
                               (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.0, -> R=\bar{ATR}(TF='1h')/n=500
                                   ==> TP=500 & SL=-500)
                               (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=0.5, -> R=\bar{ATR}(TF='1h')/n=500
                                   ==> TP=500 & SL=-750)
           Run product of unique param values in each category, use the best N params params as the starting seeds
               for Optimization ... next
        """

        # Get the lens and sizes of each parameter to determine number of combinations and create a numpy record
        self._initiate_parameters_records(add_ids=False)
        self._initiate_metric_records()

        # logger.info(self.parameters_record)
        # logger.info(self.parameters_lengths_dict)
        # logger.info(f'\n\n')
        # Fill initial parameter space
        #
        timeframe_params = {}
        window_params = {}
        #
        if not self.runtime_settings["Optimization_Settings.Continue"]:
            # Timeframe type parameters
            for key_name in self.timeframe_keynames:
                '''All Categorical Params'''
                number_of_suggestions = self.parameters_lengths_dict[f'{key_name}_length']
                #
                values = self.parameter_windows[key_name]["choices"]
                assert number_of_suggestions == len(values)
                #
                timeframe_params[key_name] = values

            # Window type parameters
            for key_name in self.window_key_names:
                '''Use a grid-like approach to windows for indicators'''
                number_of_suggestions = self.parameters_lengths_dict[f'{key_name}_length']
                upper_bound = self.parameter_windows[key_name]["upper_bound"]
                lower_bound = self.parameter_windows[key_name]["lower_bound"]
                #
                values = np.linspace(start=lower_bound, stop=upper_bound, num=number_of_suggestions).astype(int)
                assert number_of_suggestions == len(values)

                window_params[key_name] = values

                ...

            # TP_SL type parameters
            # todo
            if not self.parameters_lengths_dict["using_reduced_tp_sl_space"]:
                ''''''
                # TODO:
                #  this means that we can run all the tp_sl combinations, doubtful we will hit this but here
                #  in the unlikelihood it occurs
                ...
            else:
                #
                self._compute_bar_atr()
                #
                # todo
                if not self.tp_sl_selection_space["number_of_bar_trends"] == 1:
                    ''''''
                    # TODO:
                    #  this means that we need to apply the next few steps for each trend type
                    ...
                else:
                    number_of_suggestions = self.parameters_lengths_dict[f'tp_sl_length']
                    #
                    tp_sl_dict = self._compute_tp_n_sl_from_tp_sl_0()
                    #
                    tp_sl = []
                    for tf in self.parameters_lengths_dict["all_tf_in_this_study"]:
                        tp_sl.extend(
                            (take_profit, stop_loss) for take_profit, stop_loss in
                            zip(tp_sl_dict[tf]["take_profits"], tp_sl_dict[tf]["stop_losses"])
                        )
                    #
                    assert number_of_suggestions == len(tp_sl)

            #
        else:
            # TODO: code else ... the continuation of the study
            ...
        params = timeframe_params | window_params
        params["tp_sl"] = tp_sl
        self._compute_params_product_n_fill_record(params)

    # todo
    def simulate_strategy(self):
        # TODO:
        #   In batches or similar to Genie[full]:
        #       1.  Simulate N parameters' indicators
        #       2.  Simulate N parameters' events

        ...

    # todo
    def tell_genie(self):
        # TODO:
        #   In batches or similar to Genie[full]:
        #       3.  Tell N parameters' to Genie
        ...

    # todo
    def initiate_optuna_study(self):
        ...

    # todo
    def run_refining_epochs(self):
        ...

    # todo
    def analyze(self):
        ...
