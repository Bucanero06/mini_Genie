import os.path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from logger_tt import logger
from numpy import random


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
        # todo: need to upgrade to ray
        # Init Ray
        # ray.init(num_cpus=24)

        self.status = '__init__'
        import flatdict
        self.runtime_settings = flatdict.FlatDict(runtime_kwargs, delimiter='.')

        # Define frequently used variables in first level
        self.study_name = self.runtime_settings["Optimization_Settings.study_name"]
        #
        self.batch_size = self.runtime_settings["Optimization_Settings.batch_size"]
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
        self.continuing = self.runtime_settings["Optimization_Settings.Continue"]
        #
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
        self.tp_keyname = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_TP_TYPES)
        self.sl_keyname = tuple(key_name for key_name in self.key_names if self.parameter_windows[key_name][
            'type'].lower() in self.ACCEPTED_SL_TYPES)
        assert len(self.tp_keyname) == len(self.sl_keyname) == 1
        #
        self.tp_sl_selection_space = self.runtime_settings[
            "Optimization_Settings.Initial_Search.parameter_selection.tp_sl"]
        #
        self.parameters_record = [None]
        self.metrics_record = [None]
        self.parameters_lengths_dict = [None]
        #
        # Initial Actions
        #
        # Prepare directories and save file paths
        self._prepare_directory_paths_for_study()
        #
        if self.continuing:
            # Load precomputed params, values, and stats if continuing study
            self._load_initial_params_n_precomputed_metrics()

    def print_dict(self, optional_object=None):
        """

        Returns:
            object:
        """
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)
        # pprint.pprint(dict(self) if not optional_object else dict(optional_object))

    def _prepare_directory_paths_for_study(self):
        """
        Returns:
            object:

        """
        from Utilities.general_utilities import create_dir

        logger.debug('''Create Folders if needed''')
        studies_directory = 'Studies'
        study_dir_path = f'{studies_directory}/Study_{self.study_name}'
        portfolio_dir_path = f'{study_dir_path}/Portfolio'
        reports_dir_path = f'{study_dir_path}/Reports'
        data_dir_path = f'Datas'
        misc_dir_path = f'{study_dir_path}/misc'
        #
        file_name_of_initial_params_record = self.runtime_settings[
            'Optimization_Settings.Initial_Search.path_of_initial_params_record']
        file_name_of_initial_metrics_record = self.runtime_settings[
            'Optimization_Settings.Initial_Search.path_of_initial_metrics_record']
        #
        create_dir(studies_directory)
        create_dir(study_dir_path)
        create_dir(portfolio_dir_path)
        create_dir(reports_dir_path)
        create_dir(data_dir_path)
        create_dir(misc_dir_path)
        create_dir(f'Strategies')

        self.study_dir_path = study_dir_path
        self.portfolio_dir_path = portfolio_dir_path
        self.reports_dir_path = reports_dir_path
        self.data_dir_path = data_dir_path
        self.misc_dir_path = misc_dir_path
        #
        self.path_of_initial_params_record = f'{self.study_dir_path}/{file_name_of_initial_params_record}'
        self.path_of_initial_metrics_record = f'{self.study_dir_path}/{file_name_of_initial_metrics_record}'
        #
        # pathlib.Path('my_file.txt').suffix
        self.compression_of_initial_params_record = os.path.splitext(file_name_of_initial_params_record)[-1]
        self.compression_of_initial_metrics_record = os.path.splitext(file_name_of_initial_metrics_record)[-1]
        #

    def _load_initial_params_n_precomputed_metrics(self):
        """
        If continuing a previously started study, then we can assume we have computed some parameters already.
        The parameter combinations, trial_number, values and stats should have been saved to a file.
        Here we only want to load the necessary, thus load trial number (to be used as index),
        trial params, and trial objective value. Place this object in ray to be referred back to already
        computed params and values.

        Returns:
            object:
        """

        path_of_initial_params_record = self.path_of_initial_params_record
        path_of_initial_metrics_record = self.path_of_initial_metrics_record
        logger.info("_load_initial_params_n_precomputed_metrics")
        logger.info(path_of_initial_params_record)
        if os.path.exists(path_of_initial_params_record):
            if self.compression_of_initial_params_record != 'gzip':
                parameter_df = pd.read_csv(path_of_initial_params_record)
            else:
                parameter_df = pd.read_parquet(path_of_initial_params_record)
            #
            initial_params_size = len(parameter_df)
            #
            # Initiate_params_record
            self._initiate_parameters_records(add_ids=True)
            #
            # Fill values
            for key_name, values in parameter_df.items():
                self.parameters_record[key_name] = values
            #
            if os.path.exists(path_of_initial_metrics_record):
                # column_names = list(self.key_names) + self.runtime_settings['Optimization_Settings.Loss_Function.metrics']
                if self.compression_of_initial_metrics_record != 'gzip':
                    metrics_df = pd.read_csv(path_of_initial_metrics_record)
                else:
                    metrics_df = pd.read_parquet(path_of_initial_metrics_record)
                #
                # Initiate_metrics_record
                self._initiate_metric_records(add_ids=True, initial_params_size=initial_params_size)
                #
                # Fill values
                trial_ids_computed = metrics_df["trial_id"]
                for key_name in self.metrics_record.dtype.names:
                    for trial_id in trial_ids_computed:
                        value = metrics_df[key_name][metrics_df["trial_id"] == trial_id].values[0]
                        self.metrics_record[key_name][trial_id] = value
                #

    # todo everything
    def _grade_parameter_combinations(self):
        """Use a loss function to give a value to each parameter combination,..."""
        # TODO:
        #   code _grade_parameter_combinations
        # TODO:
        #   How should we grade them? For now I think we get stats for as many parameter combinations
        #       as possible, then we can group strategies to determine the most stable regions, neighbors
        #       etc ...

    def fetch_and_prepare_input_data(self):
        """
        Returns:
            object: 

        """
        self.status = 'fetch_and_prepare_input_data'

        from Data_Handler.data_handler import Data_Handler
        data_processing = Data_Handler(self).fetch_data()  # Load symbols_data (attr)
        data_processing.break_up_olhc_data_from_symbols_data()  # splits ^ into open, low, high, close, *alt (attrs)

    def _initiate_parameters_records(self, add_ids=None, initial_params_size=None):
        def _total_possible_values_in_window(lower, upper, step):
            return int((upper - lower) / step)

        def _reduce_initial_parameter_space(lengths_dict, max_initial_combinations):
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

            def _compute_n_initial_combinations_carefully(dict):  # fixme: naming is horrible
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

            def _compute_windows_lengths_now(dict, _keynames=None):
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
                if isinstance(self.parameter_windows[key_name]['min_step'], int):
                    parameters_record_dtype.append((key_name, 'i8'))
                elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                    parameters_record_dtype.append((key_name, 'f8'))
                else:
                    logger.error(f'Parameter {key_name} is defined as type window but inputs are inconsistent.\n'
                                 f'     (e.g. -> Either lower_bound or upper_bound is a float => float)\n'
                                 f'     (e.g. -> both lower_bound or upper_bound are integers => int)\n'
                                 f' upper_bound type:   {type(self.parameter_windows[key_name]["upper_bound"])}'
                                 f' lower_bound type:   {type(self.parameter_windows[key_name]["lower_bound"])}'
                                 )
                    sys.exit()
                parameters_lengths_dict[f'{key_name}_length'] = _total_possible_values_in_window(
                    self.parameter_windows[key_name]["lower_bound"], self.parameter_windows[key_name]["upper_bound"],
                    self.parameter_windows[key_name]["min_step"])

            elif self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_TP_SL_TYPES:
                if isinstance(self.parameter_windows[key_name]['min_step'], int):
                    parameters_record_dtype.append((key_name, 'i8'))
                elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                    parameters_record_dtype.append((key_name, 'f8'))
                else:
                    logger.error(f'Parameter {key_name} is defined as type window but inputs are inconsistent.\n'
                                 f'     (e.g. -> Either lower_bound or upper_bound is a float => float)\n'
                                 f'     (e.g. -> both lower_bound or upper_bound are integers => int)\n'
                                 f' upper_bound type:   {type(self.parameter_windows[key_name]["upper_bound"])}'
                                 f' lower_bound type:   {type(self.parameter_windows[key_name]["lower_bound"])}'
                                 )
                    sys.exit()
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
                sys.exit()

        parameters_lengths_dict["all_tf_in_this_study"] = list(set(parameters_lengths_dict["all_tf_in_this_study"]))
        # Determine size of complete parameter space combinations with settings given as well reduced space
        parameters_lengths_dict["n_total_combinations"] = parameters_lengths_dict["n_initial_combinations"] = \
            np.product([parameters_lengths_dict[f'{key_name}_length'] for key_name in self.key_names])

        if not initial_params_size:
            self.parameters_lengths_dict = _reduce_initial_parameter_space(parameters_lengths_dict,
                                                                           self.runtime_settings[
                                                                               "Optimization_Settings.Initial_Search.max_initial_combinations"])
        else:
            parameters_lengths_dict["n_initial_combinations"] = initial_params_size
            self.parameters_lengths_dict = parameters_lengths_dict

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

        from Utilities.general_utilities import shuffle_it
        #
        logger.info(f"Shuffling Once \N{Face with Finger Covering Closed Lips}\n"
                    f"Shuffling Twice \N{Grinning Face with One Large and One Small Eye}\n"
                    f"Shuffling a Third time \N{Hugging Face}\n")
        initial_param_combinations = shuffle_it(initial_param_combinations, n_times=3)
        #
        for index in range(len(initial_param_combinations)):
            value = ((index,) + initial_param_combinations[index][:-1] + initial_param_combinations[index][-1])
            self.parameters_record[index] = value

    def _compute_bar_atr(self):
        """
        Returns:
            object: 

        """
        from mini_genie_source.Simulation_Handler.compute_bar_atr import compute_bar_atr
        self.bar_atr = compute_bar_atr(self)

    @staticmethod
    def _fill_tp_sl_n_skip_out_of_bound_suggestions(tp_sl_record, tp_sl_0, n_ratios,
                                                    gamma_ratios, tick_size,
                                                    tp_upper_bound, tp_lower_bound,
                                                    sl_upper_bound,
                                                    sl_lower_bound,
                                                    skipped_indexes, tf_index):
        """
        tp_sl_0: base
        n_ratios: you multiply the atr with
        gamma_ratios: risk reward ratio

        Args:
            tp_sl_record:
            tp_sl_0:
            n_ratios:
            gamma_ratios:
            tick_size:
            tp_upper_bound:
            tp_lower_bound:
            sl_upper_bound:
            sl_lower_bound:
            skipped_indexes:
            tf_index:

        Returns:
            object:
        """

        _index = -1
        batch_ = len(n_ratios) * len(gamma_ratios)
        for n in n_ratios:
            for gamma in gamma_ratios:
                _index = (_index + 1)
                index = (tf_index * batch_) + _index
                #
                adj_tp_sl_0 = tp_sl_0 * n
                diff = adj_tp_sl_0 * gamma
                #
                if gamma < 1:
                    tp_value = adj_tp_sl_0 / tick_size
                    sl_value = -(adj_tp_sl_0 + diff) / tick_size
                #
                else:
                    tp_value = (adj_tp_sl_0 + diff) / tick_size
                    sl_value = (-adj_tp_sl_0) / tick_size
                #
                if (tp_lower_bound <= abs(tp_value) <= tp_upper_bound) \
                        and (sl_lower_bound <= abs(sl_value) <= sl_upper_bound):
                    tp_sl_record["take_profit"][index] = tp_value
                    tp_sl_record["stop_loss"][index] = sl_value
                    #
                else:
                    skipped_indexes.append(index)

        return tp_sl_record, skipped_indexes

    @property
    # todo: we need to apply steps in case number of trends is more than 1
    def _compute_tp_n_sl_from_tp_sl_0(self):
        """

        Returns:
            object:

        """
        #
        n_ratios = sorted(self.tp_sl_selection_space["n_ratios"])
        gamma_ratios = self.tp_sl_selection_space["gamma_ratios"]
        tick_size = self.runtime_settings["Data_Settings.tick_size"]
        #
        tp_upper_bound, tp_lower_bound, tp_min_step = self.parameter_windows[self.tp_keyname[0]]["upper_bound"], \
                                                      self.parameter_windows[self.tp_keyname[0]]["lower_bound"], \
                                                      self.parameter_windows[self.tp_keyname[0]]["min_step"]
        sl_upper_bound, sl_lower_bound, sl_min_step = self.parameter_windows[self.sl_keyname[0]]["upper_bound"], \
                                                      self.parameter_windows[self.sl_keyname[0]]["lower_bound"], \
                                                      self.parameter_windows[self.sl_keyname[0]]["min_step"]
        #

        dtype = 'i8' if isinstance(self.parameter_windows[self.tp_keyname[0]]['lower_bound'], int) else 'f4'
        tp_sl_record = np.empty(self.parameters_lengths_dict["tp_sl_length"],
                                dtype=np.dtype([
                                    ('take_profit', dtype),
                                    ('stop_loss', dtype)
                                ]))

        #
        skipped_indexes = []
        #
        for tf_index, tf in enumerate(self.parameters_lengths_dict[f'all_tf_in_this_study']):
            '''Fill with tp and sl that lay within their bounds'''
            # todo
            if not self.tp_sl_selection_space["number_of_bar_trends"] == 1:
                ''''''
                # TODO:
                #  this means that we need to apply the next few steps for each trend type
                ...
            else:
                tp_sl_0 = self.bar_atr[tf]["mean_atr"]
                #
                tp_sl_record, skipped_indexes = self._fill_tp_sl_n_skip_out_of_bound_suggestions(tp_sl_record, tp_sl_0,
                                                                                                 n_ratios, gamma_ratios,
                                                                                                 tick_size,
                                                                                                 tp_upper_bound,
                                                                                                 tp_lower_bound,
                                                                                                 sl_upper_bound,
                                                                                                 sl_lower_bound,
                                                                                                 skipped_indexes,
                                                                                                 tf_index)

        #
        #   Fill missing indexes that weren't in bounds
        tp_range = np.arange(tp_lower_bound + tp_min_step, tp_upper_bound, tp_min_step)
        sl_range = np.arange(sl_lower_bound + sl_min_step, sl_upper_bound, sl_min_step)
        #
        if skipped_indexes:
            logger.warning(
                f'Redefinding a total of {len(skipped_indexes)} tp_n_sl\'s that did not reside within the bounds of their p-space \n'
                f'  ** these are selected at random, however, change the initial n and gamma ratios to something more appropriate to minimize this process**')
            for missing_index in skipped_indexes:
                tp_sl_record["take_profit"][missing_index] = random.choice(
                    [x for x in tp_range if x not in tp_sl_record["take_profit"]])
                tp_sl_record["stop_loss"][missing_index] = random.choice(
                    [x for x in sl_range if x not in tp_sl_record["stop_loss"]])

        return tp_sl_record

    def _initiate_metric_records(self, add_ids=None, initial_params_size=None):
        """
        Returns:
            object:

        """
        metrics_record_dtype = []
        if add_ids:
            metrics_record_dtype.append(('trial_id', np.int_))
        for metric_name in self.metrics_key_names:
            metrics_record_dtype.append((metric_name, 'U8'))
        self.metric_data_dtype = np.dtype(metrics_record_dtype)
        self.metrics_record = np.empty(
            self.parameters_lengths_dict["n_initial_combinations"] if not initial_params_size else initial_params_size,
            dtype=self.metric_data_dtype)

    # todo initial is done but everything for continuation missing
    def suggest_parameters(self):
        """
          List of Initial Params:
               Product of:
                   All Categorical Params
                   Use a grid-like approach to windows for indicators
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

        if not all(self.parameters_record):
            self.continuing = False
            # Get the lens and sizes of each parameter to determine number of combinations and create a numpy record
            self._initiate_parameters_records(add_ids=True)
        #
        if not any(self.metrics_record):
            if not all(self.parameters_lengths_dict):
                from Utilities.general_utilities import load_dict_from_file
                self.parameters_lengths_dict = load_dict_from_file(f'{self.misc_dir_path}/_parameters_lengths_dict')
            #
            self._initiate_metric_records(add_ids=True)

        '''Fill initial parameter space'''
        #
        timeframe_params = {}
        window_params = {}
        #
        #
        # todo TP_SL type parameters needs to add steps in case number of trends is more than 1
        if not self.continuing:
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
                min_step = self.parameter_windows[key_name]["min_step"]
                #
                values = np.linspace(start=lower_bound, stop=upper_bound, num=number_of_suggestions).astype(
                    type(min_step))
                assert number_of_suggestions == len(values)

                window_params[key_name] = values

                ...

            # Concat Timeframes and Windows parameters
            params = timeframe_params | window_params

            # TP_SL type parameters
            # todo we need to apply steps in case number of trends is more than 1
            if self.parameters_lengths_dict["using_reduced_tp_sl_space"]:
                #
                from Simulation_Handler.simulation_handler import Simulation_Handler
                simulation_handler = Simulation_Handler(self)
                simulation_handler.compute_bar_atr
                #
                number_of_suggestions = self.parameters_lengths_dict[f'tp_sl_length']
                #
                # todo we need to apply steps in case number of trends is more than 1
                tp_sl_record = self._compute_tp_n_sl_from_tp_sl_0
                #
                assert number_of_suggestions == len(tp_sl_record)
                params["tp_sl"] = [(tp, sl) for tp, sl in zip(tp_sl_record["take_profit"], tp_sl_record["stop_loss"])]
            #
            else:
                '''This means that we can run all the tp_sl combinations, doubtful we will hit this but here
                  in the unlikelihood it occurs'''
                #
                tp_upper_bound, tp_lower_bound, tp_min_step = self.parameter_windows[self.tp_keyname[0]]["upper_bound"], \
                                                              self.parameter_windows[self.tp_keyname[0]]["lower_bound"], \
                                                              self.parameter_windows[self.tp_keyname[0]]["min_step"]
                tp_number_of_suggestions = self.parameters_lengths_dict[f'{self.tp_keyname[0]}_length']
                #
                sl_upper_bound, sl_lower_bound, sl_min_step = self.parameter_windows[self.sl_keyname[0]]["upper_bound"], \
                                                              self.parameter_windows[self.sl_keyname[0]]["lower_bound"], \
                                                              self.parameter_windows[self.sl_keyname[0]]["min_step"]
                sl_number_of_suggestions = self.parameters_lengths_dict[f'{self.sl_keyname[0]}_length']
                #
                tp_values = np.linspace(start=tp_lower_bound, stop=tp_upper_bound, num=tp_number_of_suggestions).astype(
                    type(tp_min_step))
                sl_values = np.linspace(start=sl_lower_bound, stop=sl_upper_bound, num=sl_number_of_suggestions).astype(
                    type(sl_min_step))
                #
                from itertools import product
                tp_sl_combinations = list(set(product(tp_values, sl_values)))
                #
                params["tp_sl"] = [tp_sl for tp_sl in tp_sl_combinations]
            #
            self._compute_params_product_n_fill_record(params)
            self._save_initial_param_record()
            from Utilities.general_utilities import write_dictionary_to_file
            write_dictionary_to_file(f'{self.misc_dir_path}/_parameters_lengths_dict', self.parameters_lengths_dict)
            #
        #
        else:
            from Utilities.general_utilities import load_dict_from_file
            parameters_lengths_dict = load_dict_from_file(f'{self.misc_dir_path}/_parameters_lengths_dict')
            logger.info(f'{parameters_lengths_dict = }')
            logger.info(f'\n')
            logger.info(f'{self.parameters_lengths_dict = }')
            assert parameters_lengths_dict["n_total_combinations"] == self.parameters_lengths_dict[
                "n_total_combinations"]
            assert parameters_lengths_dict["n_initial_combinations"] == self.parameters_lengths_dict[
                "n_initial_combinations"]
        #
        logger.info(f'Total # of Combinations -> {self.parameters_lengths_dict["n_total_combinations"]}\n'
                    f'  * given current definitions for parameter space\n'
                    f'\n'
                    f'Initial Parameter Space Reduced to {self.parameters_lengths_dict["n_initial_combinations"]}')
        for key_name in self.key_names:
            if key_name not in self.tp_sl_keynames:
                logger.info(f'# of {key_name} paramters = {self.parameters_lengths_dict[f"{key_name}_length"]} ')
        logger.info(f'# of TP/SL combinations = {self.parameters_lengths_dict[f"tp_sl_length"]} ')

    def _save_initial_param_record(self):
        import pandas as pd
        df = pd.DataFrame(self.parameters_record).set_index('trial_id')
        #
        df.to_csv(
            self.path_of_initial_params_record) if self.compression_of_initial_params_record != 'gzip' else df.to_parquet(
            self.path_of_initial_params_record, compression='gzip')

    def _save_initial_computed_params(self):
        """Add to previously created file (or new if it does not exist), in a memory conscious way, the trial number,
        parameter combination, values, and combination stats"""

        import pandas as pd
        from Utilities.general_utilities import delete_non_filled_elements, rm_field_from_record
        #
        filled_metrics = delete_non_filled_elements(self.metrics_record)
        filled_params = np.take(self.parameters_record, filled_metrics["trial_id"])
        filled_params = rm_field_from_record(filled_params, 'trial_id')
        #
        import numpy.lib.recfunctions as rfn
        merged_array = rfn.merge_arrays([filled_params, filled_metrics], flatten=True, usemask=False)
        #
        df = pd.DataFrame(merged_array).set_index('trial_id')
        #
        df.to_csv(
            self.path_of_initial_metrics_record) if self.compression_of_initial_metrics_record != 'gzip' else df.to_parquet(
            self.path_of_initial_metrics_record, compression='gzip')

    # TODO CONTINUING SECTIONS
    def simulate(self, SAVE_EVERY_Nth_CHUNK=None):
        """
        In chunks/batches:
           1.  Simulate N parameters' indicators
           2.  Simulate N parameters' events
           3.  Compute Metrics
           4.  Save Results to file
        """

        from Simulation_Handler.simulation_handler import Simulation_Handler
        from Analysis_Handler.analysis_handler import Analysis_Handler
        simulation_handler = Simulation_Handler(self)
        analysis_handler = Analysis_Handler(self)
        #
        number_of_parameters = len(self.parameters_record)
        batch_size = self.batch_size
        N_chunks = int(np.ceil(number_of_parameters / batch_size))
        #
        highest_profit = -sys.maxsize
        best_parameters = None
        #
        initial_cash_total = self.runtime_settings["Simulation_Settings.Portfolio_Settings.init_cash"]
        stop_after_n_epoch = self.runtime_settings["Optimization_Settings.Initial_Search.stop_after_n_epoch"]
        #
        # todo!!!! do not simulate already_computed params
        for chunk_index in range(N_chunks):  # todo: would like to use ray to scale this (*keep track of index)
            if chunk_index == stop_after_n_epoch:
                break
            #
            start_time = perf_counter()
            logger.info(f'\n\n\n')
            logger.info(f'Epoch --> {chunk_index}, Trials Ran --> {chunk_index * self.batch_size} ')
            indexes = [chunk_index * self.batch_size, chunk_index * self.batch_size + self.batch_size]
            epoch_index_range_ = range(indexes[0], indexes[1], 1)
            epoch_params = np.take(self.parameters_record, epoch_index_range_)
            #
            long_entries, long_exits, short_entries, short_exits, \
            strategy_specific_kwargs = simulation_handler.simulate_signals(epoch_params)
            #
            pf, extra_sim_info = simulation_handler.simulate_events(long_entries, long_exits,
                                                                    short_entries, short_exits,
                                                                    strategy_specific_kwargs)
            #
            '''Reconstruct Metrics from Order Records and Save'''
            portfolio_combined = analysis_handler.compute_stats(pf, self.metrics_key_names, groupby=self.group_by)
            metrics_this_epoch = portfolio_combined.to_numpy()
            #
            # Fill metric record with new metric values
            for _index, overall_index in enumerate(epoch_index_range_):
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
            # Concat and save the parameter and metric records to file every Nth epoch
            if SAVE_EVERY_Nth_CHUNK:
                if chunk_index % SAVE_EVERY_Nth_CHUNK == 0:
                    self._save_initial_computed_params()
                    #
                    # TODO:
                    #  IDEA:
                    #   * Can you use ray inside of apply_function MMT?
                    #   * Set up ray clusters which means get this code ready for that
                    #   * Replace parameters that resulted in NaN each reset?
                    #   * After a large initial run chunk then run optuna (1 comb at a time without storage) on all or reduced space
                    #   * How to run ray server across multiple pc? and how fast could it run on AWS?
                    #   _
                    #  TASK:
                    #   * Need to add backtest module/logic (can use np.arrange)
                    #   * Fill, save and load params,  metrics records (convinient file format) -------------------------
                    #   * Do not compute or include in chunks, previously computed
                    #   * ADD EQUIPMENT HANDLER -------------------------------------------------
                    #
                    #
                    logger.info(f'Just saved to file during epoch {chunk_index}')
            #
            logger.info(f'Epoch {chunk_index} took {perf_counter() - start_time} seconds')
        exit()
