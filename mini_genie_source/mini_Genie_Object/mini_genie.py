#!/usr/bin/env python3.9
import os.path
import sys
import warnings
from datetime import datetime
from os import path, remove
from time import perf_counter

import numpy as np
import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger

from Analysis_Handler.analysis_handler import compute_stats_remote
from Equipment_Handler.equipment_handler import CHECKTEMPS
from Run_Time_Handler.equipment_settings import TEMP_DICT
from Utilities.general_utilities import rm_field_from_record, next_path, create_dirs, create_or_clean_directories, \
    flip_bool

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)


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

    def __init__(self, runtime_kwargs, args):

        """Constructor for mini_genie_trader"""

        self.parameters_record_length = None
        self.status = '__init__'
        from Utilities.general_utilities import print_dict
        self.print_dict = print_dict
        import flatdict
        self.runtime_settings = flatdict.FlatDict(runtime_kwargs, delimiter='.')

        # Define frequently used variables in first level
        self.study_name = self.runtime_settings["Simulation_Settings.study_name"]
        #
        self.batch_size = self.runtime_settings["Simulation_Settings.batch_size"]
        #
        self.user_pick = args.user_pick

        self.config_file_path = args.run_time_dictionary_path.rsplit('.', 1)[0]
        #
        self.parameter_windows = self.runtime_settings["Strategy_Settings.parameter_windows"]._values
        #
        self.key_names = tuple(self.parameter_windows.keys())
        #
        self.group_by = [f'custom_{key_name}' for key_name in self.key_names]
        #
        self.asset_names = self.runtime_settings["Data_Settings.data_files_names"]
        #
        self.metrics_key_names = self.runtime_settings['Simulation_Settings.Loss_Function.metrics']
        #
        self.optimization_start_date = self.runtime_settings['Simulation_Settings.optimization_period.start_date']
        self.optimization_end_date = self.runtime_settings['Simulation_Settings.optimization_period.end_date']
        #
        # Miscellaneous
        from datetime import datetime
        self.stop_sim_time = self.runtime_settings['Simulation_Settings.timer_limit'] + datetime.now()
        self.continuing = self.runtime_settings["Simulation_Settings.Continue"]
        self.run_mode = self.runtime_settings["Simulation_Settings.run_mode"]
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
        self.keynames_not_tp_sl = tuple(keyname for keyname in self.key_names if keyname not in self.tp_sl_keynames)
        #
        self.tp_sl_selection_space = self.runtime_settings[
            "Simulation_Settings.Initial_Search_Space.parameter_selection.tp_sl"]
        #
        self.parameters_record = [None]
        self.metrics_record = [None]
        self.parameters_lengths_dict = [None]
        #
        # Initial Actions
        #
        # Init Ray
        self.ray_init = ray.init(num_cpus=self.runtime_settings["RAY_SETTINGS.ray_init_num_cpus"],
                                 # object_store_memory=67 * 10 ** 9
                                 )
        #
        # Prepare directories and save file paths
        self._prepare_directory_paths_for_study()  # If path to study directory does not exist, self.continuing = False
        #
        self.number_of_parameters_ran = 0
        if not self.user_pick and self.continuing:
            # Load precomputed params, values, and stats if continuing study
            self._load_initial_params_n_precomputed_metrics()

    def _prepare_directory_paths_for_study(self):
        """
        Returns:
            object:

        """

        logger.debug('''Create Folders if needed''')
        studies_directory = 'Studies'
        study_dir_path = f'{studies_directory}/{self.study_name}'
        portfolio_dir_path = f'{study_dir_path}/Portfolio'
        reports_dir_path = f'{study_dir_path}/Reports'
        data_dir_path = f'Datas'
        misc_dir_path = f'{study_dir_path}/misc'
        #
        file_name_of_initial_params_record = self.runtime_settings[
            'Simulation_Settings.Initial_Search_Space.path_of_initial_params_record']
        file_name_of_initial_metrics_record = self.runtime_settings[
            'Simulation_Settings.Initial_Search_Space.path_of_initial_metrics_record']
        #
        file_name_of_backtest_results = self.runtime_settings[
            "Strategy_Settings.strategy_user_picked_params.output_file_name"]
        #
        user_defined_param_file = self.runtime_settings[
            "Strategy_Settings.strategy_user_picked_params.read_user_defined_param_file"]
        #
        from os import path
        if not path.exists(study_dir_path):
            self.continuing = False
        #
        create_dirs(studies_directory, data_dir_path)
        #
        create_or_clean_directories(study_dir_path, portfolio_dir_path, reports_dir_path, misc_dir_path,
                                    delete_content=flip_bool(self.continuing))

        self.study_dir_path = study_dir_path
        self.portfolio_dir_path = portfolio_dir_path
        self.reports_dir_path = reports_dir_path
        self.data_dir_path = data_dir_path
        self.misc_dir_path = misc_dir_path
        #
        self.path_of_initial_params_record = f'{self.study_dir_path}/{file_name_of_initial_params_record}'
        self.path_of_initial_metrics_record = f'{self.study_dir_path}/{file_name_of_initial_metrics_record}'
        self.file_name_of_backtest_results = f'{self.study_dir_path}/{file_name_of_backtest_results}'
        self.user_defined_param_file = f'{self.study_dir_path}/{user_defined_param_file}'
        self.path_of_saved_study_config_file = f'{self.study_dir_path}/{self.config_file_path}'

        #
        self.compression_of_initial_params_record = file_name_of_initial_params_record.rsplit('.', 1)[-1]
        self.compression_of_initial_params_record = file_name_of_initial_metrics_record.rsplit('.', 1)[-1]
        #
        if not self.continuing:
            if path.exists(self.path_of_initial_params_record):
                remove(self.path_of_initial_params_record)
            if path.exists(self.path_of_initial_metrics_record):
                remove(self.path_of_initial_metrics_record)
            if path.exists(self.path_of_saved_study_config_file):
                remove(self.path_of_saved_study_config_file)
            #
            import shutil
            shutil.copy2(self.config_file_path, self.path_of_saved_study_config_file)
            #

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
        #
        if os.path.exists(path_of_initial_params_record):
            logger.info(f'Loading parameters from file {path_of_initial_params_record}  ...')
            if self.compression_of_initial_params_record == 'csv':
                # parameter_df = pd.read_csv(path_of_initial_params_record)
                from dask import dataframe as dd
                parameter_df = dd.read_csv(path_of_initial_params_record).compute(scheduler='processes')
            else:
                parameter_df = pd.read_pickle(path_of_initial_params_record)
            #
            # Initiate_params_record
            logger.info(f'Initiating parameters record  ...')
            initial_params_size = len(parameter_df)

            # fixme read from mist dict instead
            # max_initial_combinations = self.runtime_settings[
            #     "Simulation_Settings.Initial_Search_Space.max_initial_combinations"]
            # assert max_initial_combinations >= initial_params_size
            #
            self._initiate_parameters_records(add_ids=True, initial_params_size=initial_params_size)
            #
            # Fill values
            logger.info(f'Filling record  ...')
            for key_name, values in parameter_df.items():
                self.parameters_record[key_name] = values
            #
            logger.info(f'Loaded parameters_record')
            #
            if os.path.exists(path_of_initial_metrics_record):
                logger.info(f'Loading metrics from file {path_of_initial_metrics_record}  ...')
                # column_names = list(self.key_names) + self.runtime_settings['Simulation_Settings.Loss_Function.metrics']
                # metrics_df = pd.read_csv(path_of_initial_metrics_record)
                from dask import dataframe as dd
                metrics_df = dd.read_csv(path_of_initial_params_record).compute(scheduler='processes')
                #
                # Determine number of parameter combinations ran
                self.number_of_parameters_ran = int(len(metrics_df) / len(self.asset_names))
                #
                # Initiate_metrics_record
                logger.info(f'Initiating metrics record  ...')
                self._initiate_metric_records(add_ids=True, params_size=initial_params_size * len(self.asset_names))
                #
                # Fill values
                logger.info(f'Filling record ...')
                dtype_names = list(self.metrics_record.dtype.names)
                for row in metrics_df[dtype_names].to_numpy():
                    # First row is 'trial_id'
                    trial_id = row[0]
                    # Second row is 'asset' name
                    asset_name = row[1]
                    ass_index = self.asset_names.index(asset_name)
                    self.metrics_record[trial_id + (initial_params_size * ass_index)] = tuple(row)
                logger.info(f'Loaded metrics_record')

            ''''''

            # # Clean Data
            # trial_ids_not_ran_mask = ~parameter_df["trial_id"].isin(metrics_df["trial_id"])
            # cleaned_parameter_df = parameter_df["trial_id"][trial_ids_not_ran_mask]
            # from Utilities.general_utilities import clean_params_record
            # self.parameters_record = clean_params_record(self.parameters_record)
            # initial_params_size = len(self.parameters_record)

            #

    def fetch_and_prepare_input_data(self):
        """
        Returns:
            object:

        """
        self.status = 'fetch_and_prepare_input_data'

        from Data_Handler.data_handler import Data_Handler
        data_processing = Data_Handler(self).fetch_data()  # Load symbols_data (attr)
        data_processing.break_up_olhc_data_from_symbols_data()  # splits ^ into open, low, high, close, *alt (attrs)

    def _define_backtest_parameters(self):
        n_initial_combinations, initial_param_combinations, parameter_df = None, None, pd.DataFrame()
        self.user_defined_param_file_bool = os.path.exists(self.user_defined_param_file)
        #
        # If parameter file given
        if self.user_defined_param_file_bool:
            # load parameters (will only use parameter columns), create empty record with same length, fill with set of parameters
            parameter_df = pd.read_csv(self.user_defined_param_file) \
                if self.user_defined_param_file.rsplit('.', 1)[-1] == 'csv' \
                else pd.read_pickle(self.user_defined_param_file)  # fixme looks ugly
            #
            parameter_df = parameter_df[list(self.key_names)]
            n_initial_combinations = len(parameter_df)
            #
        # if product of user picked parameters
        elif self.runtime_settings[f"Strategy_Settings.strategy_user_picked_params.compute_product"]:
            n_initial_combinations = np.product(
                [len(self.parameter_windows[f'{key_name}.values']) for key_name in self.key_names])
            #
            from itertools import product

            initial_param_combinations = list(
                set(
                    product(
                        *[self.parameter_windows[f'{key_name}.values'] for key_name in self.key_names]
                    )
                )
            )
        # If user picked parameter combinations
        else:
            n_initial_combinations = len(self.parameter_windows[f'{self.key_names[0]}.values'])
            for key_name in self.key_names[1:]:
                assert len(self.parameter_windows[f'{key_name}.values']) == n_initial_combinations
            initial_param_combinations = np.array(
                [(self.parameter_windows[f'{key_name}.values']) for key_name in self.key_names]).transpose()
        #
        initial_param_combinations = initial_param_combinations if parameter_df.empty else parameter_df
        return n_initial_combinations, initial_param_combinations

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

            def _compute_n_initial_combinations_carefully(dictionary):  # fixme: naming is horrible
                """

                Args:
                    dictionary:

                Returns:

                """
                n_reduced_lengths = [dictionary[f'{key_name}_length'] for key_name in self.key_names
                                     if self.parameter_windows[key_name][
                                         'type'].lower() not in self.ACCEPTED_TP_SL_TYPES]
                n_reduced_lengths.append(dictionary["tp_sl_length"])

                return np.product(n_reduced_lengths)

            def _compute_windows_lengths_now(dictionary, _keynames=None):
                """

                Args:
                    dictionary:
                    _keynames:

                Returns:
                    object:

                """
                _keynames = self.key_names if not _keynames else _keynames

                return np.product(
                    [dictionary[f'{key_name}_length'] for key_name in _keynames if
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
                                                                                    _keynames=temp_lengths_dict[
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
                tf_in_key_name = [tf.lower() for tf in self.parameter_windows[key_name]['values']]

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
                    parameters_record_dtype.append((key_name, 'i4'))
                elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                    parameters_record_dtype.append((key_name, 'f4'))
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
                    parameters_record_dtype.append((key_name, 'i4'))
                elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                    parameters_record_dtype.append((key_name, 'f4'))
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

                if not self.user_pick:
                    parameters_lengths_dict[f'tp_sl_length'] = parameters_lengths_dict[f'tp_sl_length'] * \
                                                               parameters_lengths_dict[f'{key_name}_length'] if \
                        parameters_lengths_dict[f'tp_sl_length'] else parameters_lengths_dict[f'{key_name}_length']

            else:
                logger.error(f'Parameter {key_name} is defined as type {self.parameter_windows[key_name]["type"]}'
                             f'but that type is not accepted ... yet ;)')
                sys.exit()

        if not self.user_pick:
            parameters_lengths_dict["all_tf_in_this_study"] = list(set(parameters_lengths_dict["all_tf_in_this_study"]))
            # Determine size of complete parameter space combinations with settings given as well reduced space
            parameters_lengths_dict["n_total_combinations"] = parameters_lengths_dict["n_initial_combinations"] = \
                np.product([parameters_lengths_dict[f'{key_name}_length'] for key_name in self.key_names])

            '''Get record dimensions'''
            if not initial_params_size:
                self.parameters_lengths_dict = _reduce_initial_parameter_space(parameters_lengths_dict,
                                                                               self.runtime_settings[
                                                                                   "Simulation_Settings.Initial_Search_Space.max_initial_combinations"])
            else:
                parameters_lengths_dict["n_initial_combinations"] = initial_params_size
                self.parameters_lengths_dict = parameters_lengths_dict

            if self.parameters_lengths_dict["n_initial_combinations"] > self.runtime_settings[
                "Simulation_Settings.Initial_Search_Space.max_initial_combinations"]:
                continuing_text = f' because we are continuing the study from file' if self.continuing else ' '
                logger.warning(
                    f'I know max_initial_combinations was set to '
                    f'{self.runtime_settings["Simulation_Settings.Initial_Search_Space.max_initial_combinations"]} '
                    f'but, I needed at least {self.parameters_lengths_dict["n_initial_combinations"]} '
                    f'initial combinations{continuing_text}'
                    f"\N{smiling face with smiling eyes}"
                )

            self.parameter_data_dtype = np.dtype(parameters_record_dtype)
            # FIXME exploring initiating it after elminating some parameter to reduce memory usage
            if self.continuing:
                self.parameters_record = np.empty(self.parameters_lengths_dict["n_initial_combinations"],
                                                  dtype=self.parameter_data_dtype)
            # #
        else:
            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
            self.parameter_windows = self.runtime_settings[
                f"Strategy_Settings.strategy_user_picked_params.parameter_windows"]
            #
            '''Prepare'''
            # Determine Parameter Combinations (returns either a list of tuple params or a df of params)
            n_initial_combinations, initial_param_combinations = self._define_backtest_parameters()
            #
            # Create empty parameter record
            self.parameter_data_dtype = np.dtype(parameters_record_dtype)
            self.parameters_record = np.empty(n_initial_combinations, dtype=self.parameter_data_dtype)
            #
            # Fill parameter records
            if not self.user_defined_param_file_bool:  # (List of param combination tuples)
                for index in range(len(initial_param_combinations)):
                    value = ((index,) + tuple(initial_param_combinations[index]))
                    self.parameters_record[index] = value
            else:
                # fixme: left here
                self.parameters_record["trial_id"] = np.arange(0, len(initial_param_combinations), 1)
                #
                for key_name, values in initial_param_combinations.items():
                    self.parameters_record[key_name] = values
            #
            logger.info(f'Total number of combinations to run backtest on -->  {len(self.parameters_record)}\n'
                        f'  on {len(self.asset_names)} assets')

            from Utilities.general_utilities import delete_non_filled_elements
            self.parameters_record = delete_non_filled_elements(self.parameters_record)

    def _compute_params_product_n_fill_record(self, params):
        from itertools import product
        logger.info(f'Computing Cartesian Product for Parameter Record')
        initial_param_combinations = list(
            set(
                product(
                    *[
                        # params[key_name] for key_name in self.key_names if key_name not in self.tp_sl_keynames
                        params[key_name] for key_name in self.keynames_not_tp_sl
                    ], params["tp_sl"]
                )
            )
        )

        logger.info(f'Allocating parameter_record')
        logger.info(f'{len(initial_param_combinations) = }')
        self.parameters_record = np.empty(len(initial_param_combinations), dtype=self.parameter_data_dtype)
        #
        logger.info(f"Shuffling Once \N{Face with Finger Covering Closed Lips}")
        initial_param_combinations = np.random.permutation(initial_param_combinations)
        # indexes = [(i,) for i in np.arange(0, len(initial_param_combinations))]
        indexes = np.arange(0, len(initial_param_combinations))
        # value = ([(i,) for i in indexes] + initial_param_combinations[:, :-1] + initial_param_combinations[:, -1])
        column_names = list(self.parameters_record.dtype.names)
        #
        logger.info(f'Filling Parameter Record')
        self.parameters_record["trial_id"] = indexes
        for col_index, key_name in enumerate(self.keynames_not_tp_sl):
            self.parameters_record[key_name] = initial_param_combinations[:, col_index]
        #
        tp_sl = initial_param_combinations[:, -1]
        self.parameters_record[self.tp_sl_keynames[0]] = list(list(zip(*tp_sl))[0])
        self.parameters_record[self.tp_sl_keynames[1]] = list(list(zip(*tp_sl))[1])
        #
        # for index in range(len(initial_param_combinations)):
        #     # logger.info(index)
        #     ipci = initial_param_combinations[index]
        #     value = ((index,) + ipci[:-1] + ipci[-1])
        #     self.parameters_record[index] = value

        logger.info(f'Clean Up Parameter Record')
        from Utilities.general_utilities import delete_non_filled_elements
        self.parameters_record = delete_non_filled_elements(self.parameters_record)
        #
        # fixme!!!   HOTFIX
        # fixme need to remove this, or create a setting for. this was a hot fix to eliminate ema1 parameters combinations
        #   that are less than ema2
        self.parameters_record = self.parameters_record[
            np.where(self.parameters_record["ema_1_windows"] <= self.parameters_record["ema_2_windows"])[0]]
        #
        self.parameters_record = self.parameters_record[
            np.where(self.parameters_record["Trend_filter_1_timeframes"] != self.parameters_record[
                "PEAK_and_ATR_timeframes"])[0]]
        #
        rng = np.random.default_rng()
        logger.info(f"Shuffling Once \N{Face with Finger Covering Closed Lips}")
        self.parameters_record = rng.permutation(self.parameters_record)
        logger.info(f"Shuffling Twice \N{Grinning Face with One Large and One Small Eye}")
        self.parameters_record = rng.permutation(self.parameters_record)
        logger.info(f"Shuffling a Third time \N{Hugging Face}")
        self.parameters_record = rng.permutation(self.parameters_record)
        #
        # value = ([(i,) for i in indexes] + initial_param_combinations[:, :-1] + initial_param_combinations[:, -1])
        logger.info(f'Sorting Index')
        self.parameters_record["trial_id"] = indexes[:len(self.parameters_record["trial_id"])]

    def _compute_bar_atr(self):
        """
        Returns:
            object:

        """
        from Simulation_Handler.compute_bar_atr import compute_bar_atr
        logger.info(f'I am here bar')
        self.bar_atr = compute_bar_atr(self)
        logger.info(f'I am here bar after')
        from pprint import pprint
        pprint(self.bar_atr)
        logger.info(f'I am here bar print done')

    @staticmethod
    def _fill_tp_sl_n_skip_out_of_bound_suggestions(tp_sl_record, tp_sl_0, n_ratios, gamma_ratios, tick_size,
                                                    tp_upper_bound, tp_lower_bound, sl_upper_bound, sl_lower_bound,
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
        tick_sizes = self.runtime_settings["Data_Settings.tick_size"]
        tick_size = max(tick_sizes) if not isinstance(tick_sizes, float or int) else tick_sizes
        #
        tp_upper_bound, tp_lower_bound, tp_min_step = self.parameter_windows[self.tp_keyname[0]]["upper_bound"], \
                                                      self.parameter_windows[self.tp_keyname[0]]["lower_bound"], \
                                                      self.parameter_windows[self.tp_keyname[0]]["min_step"]
        sl_upper_bound, sl_lower_bound, sl_min_step = self.parameter_windows[self.sl_keyname[0]]["upper_bound"], \
                                                      self.parameter_windows[self.sl_keyname[0]]["lower_bound"], \
                                                      self.parameter_windows[self.sl_keyname[0]]["min_step"]
        #

        dtype = 'i4' if isinstance(self.parameter_windows[self.tp_keyname[0]]['lower_bound'], int) else 'f4'
        tp_sl_record = np.zeros(self.parameters_lengths_dict["tp_sl_length"],
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
        # Deletes combinations with a take profit of 0, that makes no sense
        tp_sl_record = tp_sl_record[tp_sl_record["take_profit"] != 0]
        #

        #
        #   Fill missing indexes that weren't in bounds
        # tp_range = np.arange(tp_lower_bound + tp_min_step, tp_upper_bound, tp_min_step)
        # sl_range = np.arange(sl_lower_bound + sl_min_step, sl_upper_bound, sl_min_step)
        #
        # if skipped_indexes:
        #     logger.warning(
        #         f'Redefinding a total of {len(skipped_indexes)} tp_n_sl\'s that did not reside within the bounds of their p-space \n'
        #         f'  ** these are selected at random, however, change the initial n and gamma ratios to something more appropriate to minimize this process**')
        #     for missing_index in skipped_indexes:
        #         tp_sl_record["take_profit"][missing_index] = random.choice(
        #             [x for x in tp_range if x not in tp_sl_record["take_profit"]])
        #         tp_sl_record["stop_loss"][missing_index] = random.choice(
        #             [x for x in sl_range if x not in tp_sl_record["stop_loss"]])

        return tp_sl_record, skipped_indexes

    def _initiate_metric_records(self, add_ids=None, params_size=None):
        """
        Returns:
            object:

        """
        metrics_record_dtype = []
        if add_ids:
            metrics_record_dtype.append(('trial_id', 'i4'))
        #
        metrics_record_dtype.append(('asset', 'U8'))
        #
        for metric_name in self.metrics_key_names:
            if 'duration' not in metric_name.lower():
                metrics_record_dtype.append((metric_name, 'f4'))
            else:
                metrics_record_dtype.append((metric_name, 'timedelta64'))
        #
        self.metric_data_dtype = np.dtype(metrics_record_dtype)
        self.metrics_record = np.zeros(
            self.parameters_lengths_dict["n_initial_combinations"] if not params_size else params_size,
            dtype=self.metric_data_dtype)

    # todo initial is almost complete; missing steps in case number of trends setting is more than 1
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
           Run product of unique param values in each category, and remove out of bound tp and sl combinations.
        """

        if not all(self.parameters_record):
            self.continuing = False
            # Get the lens and sizes of each parameter to determine number of combinations and create a numpy record
            self._initiate_parameters_records(add_ids=True)

        '''Fill initial parameter space'''
        #
        timeframe_params = {}
        window_params = {}
        #
        # todo TP_SL type parameters needs to add steps in case number of trends is more than 1
        if not self.continuing:
            # Timeframe type parameters
            for key_name in self.timeframe_keynames:
                '''All Categorical Params'''
                number_of_suggestions = self.parameters_lengths_dict[f'{key_name}_length']
                #
                values = self.parameter_windows[key_name]["values"]
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
                logger.info(f'I am here before compute atr')
                simulation_handler.compute_bar_atr
                #
                number_of_suggestions = self.parameters_lengths_dict[f'tp_sl_length']
                #
                # todo we need to apply steps in case number of trends is more than 1
                tp_sl_record, skipped_indexes = self._compute_tp_n_sl_from_tp_sl_0
                #
                assert number_of_suggestions == len(tp_sl_record) + len(skipped_indexes)
                params["tp_sl"] = list(
                    set([(tp, sl) for tp, sl in zip(tp_sl_record["take_profit"], tp_sl_record["stop_loss"])]))
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
            if not len(params["tp_sl"]):
                logger.warning(
                    f'Hi there! I (genie) could not produce any TP and SL combinations that fit within your '
                    f'restrictions ... sometimes the issue is a result of the wrong tick size for the dataset or upper/lower TP and SL '
                    f'bound should be expanded')
                sys.exit()
            #
            self._compute_params_product_n_fill_record(params)
            self._save_record_to_file(self.parameters_record, self.path_of_initial_params_record,
                                      self.compression_of_initial_params_record)
            #
            # Clean up parameters_lengths_dict before saving
            del self.parameters_lengths_dict[f'{self.tp_keyname[0]}_length']
            del self.parameters_lengths_dict[f'{self.sl_keyname[0]}_length']
            #
            self.parameters_lengths_dict[f'tp_sl_length'] = len(params["tp_sl"])
            from Utilities.general_utilities import write_dictionary_to_file
            write_dictionary_to_file(f'{self.misc_dir_path}/_parameters_lengths_dict', self.parameters_lengths_dict)
            #
        #
        logger.info(f'Total # of Combinations: {self.parameters_lengths_dict["n_total_combinations"]} per asset\n'
                    f'  * given current definitions for parameter space')
        logger.info(f'Initial Parameter Space Reduced to {self.parameters_lengths_dict["n_initial_combinations"]}')
        s = '' if len(self.asset_names) == 1 else 's'
        logger.info(f'Running on {len(self.asset_names)} asset{s}')

    @staticmethod
    def _save_record_to_file(record, path_to_file, extension=None, write_mode='w'):
        logger.info(f"Saving record to {path_to_file}")
        if not extension:
            extension = path_to_file.rsplit('.', 1)[-1]
        import pandas as pd
        df = pd.DataFrame(record).set_index('trial_id')
        if path.exists(path_to_file) and write_mode == 'a':
            df.to_csv(path_to_file, mode=write_mode, header=False)
        else:
            df.to_csv(path_to_file) if extension == 'csv' else df.to_pickle(path_to_file)
        #

    def _save_computed_params_metrics(self, new_indexes=None):
        """Add to previously created file (or new if it does not exist), in a memory conscious way, the trial number,
        parameter combination, and combination stats"""

        from Utilities.general_utilities import rm_field_from_record
        #
        if new_indexes is None:
            metrics_elements_to_save = self.metrics_record[self.metrics_record["asset"] != '']
            logger.info(f'Trials completed: {len(metrics_elements_to_save)}')
            #
        else:
            metrics_elements_to_save = np.take(self.metrics_record, new_indexes)
            #
        filled_params = np.take(self.parameters_record, metrics_elements_to_save["trial_id"])
        filled_params = rm_field_from_record(filled_params, 'trial_id')
        #
        #
        import numpy.lib.recfunctions as rfn
        merged_array = rfn.merge_arrays([filled_params, metrics_elements_to_save], flatten=True, usemask=False)
        #
        file_path = self.path_of_initial_metrics_record if not self.user_pick else self.file_name_of_backtest_results
        self._save_record_to_file(merged_array, file_path, write_mode='a')

    # Will be removed in a future update and replaced by simulate_with_post_processing
    def simulate(self):
        """
        In chunks/batches:
           1.  Simulate N parameters' indicators
           2.  Simulate N parameters' events
           3.  Compute Metrics
           4.  Save Results to file
        """

        def _analyze_n_save(portfolio_metrics, params_rec, highest_profit_cash_, highest_profit_perc_, best_parameters_,
                            initial_cash_total_, epoch_n_, save_every_nth_chunk=None):
            '''Reconstruct Metrics from Order Records and Save'''
            tell_metrics_start_timer = perf_counter()
            #
            # params_rec = ray.get(params_rec_id)
            #
            # Used for Printing
            highest_profit_this_epoch = portfolio_metrics['Total Return [%]'].max()
            highest_cash_profit_this_epoch = highest_profit_this_epoch * initial_cash_total_ / 100
            best_parameters_this_epoch = portfolio_metrics['Total Return [%]'].idxmax()
            #
            if highest_cash_profit_this_epoch > highest_profit_cash_:
                highest_profit_cash_ = highest_cash_profit_this_epoch
                highest_profit_perc_ = highest_profit_this_epoch
                best_parameters_ = best_parameters_this_epoch
            #
            logger.info(
                f'Highest Profit so far: {round(highest_profit_cash_, 2):,}   \N{money-mouth face}\N{money bag}: '
                f'{round(highest_profit_perc_, 2)}% of a ${initial_cash_total_:,} account')
            logger.info(f'Best Param so far: {best_parameters_}  \N{money with wings}')
            #
            logger.info(f'  -> highest_profit_cash this epoch {highest_cash_profit_this_epoch:,}')
            logger.info(f'  -> best_param this epoch {best_parameters_this_epoch}')
            #
            # clean up porfolio
            portfolio_metrics.fillna(0.0)
            for name, values in portfolio_metrics.items():
                portfolio_metrics[name] = portfolio_metrics[name].astype(float).fillna(0.0)

            # portfolio.replace(pd.NaT, pd.Timedelta(seconds=0), inplace=True)
            #
            # Fill metric record with new metric values
            new_indexes = []
            for _index, param_record in enumerate(params_rec):
                trial_id = param_record["trial_id"]
                param_record = rm_field_from_record(param_record, 'trial_id')
                #
                for ass_index, asset in enumerate(self.asset_names):
                    param_tuple_ = tuple(param_record)
                    param_tuple_ = param_tuple_ + (asset,)
                    metrics_np = tuple(portfolio_metrics[:].loc[tuple(param_tuple_)])
                    new_index = trial_id + (self.parameters_record_length * ass_index)
                    new_indexes.append(new_index)
                    metrics_input = (trial_id, asset) + metrics_np
                    self.metrics_record[new_index] = metrics_input

            logger.info(f'Time to Analyze Metrics {perf_counter() - tell_metrics_start_timer}')
            #
            # Concat and save the parameter and metric records to file every Nth epoch
            if save_every_nth_chunk:
                if epoch_n_ % save_every_nth_chunk == 0:
                    logger.info(f"Saving epoch {epoch_n_}")
                    save_start_timer = perf_counter()
                    self._save_computed_params_metrics(new_indexes)
                    #
                    logger.info(f'Time to Save Records {perf_counter() - save_start_timer} during epoch {epoch_n_}')
            #
            return highest_profit_cash_, highest_profit_perc_, best_parameters_

        batch_size = self.batch_size
        # self.parameters_record_length = len(ray.get(self.parameters_record))
        self.parameters_record_length = len(self.parameters_record)
        #
        from Simulation_Handler.simulation_handler import Simulation_Handler
        # from Analysis_Handler.analysis_handler import Analysis_Handler`
        simulation_handler = Simulation_Handler(self)
        # analysis_handler = Analysis_Handler(self)
        #
        # simulation_handler_id, analysis_handler_id = put_objects_list_to_ray([simulation_handler, analysis_handler])
        # simulation_handler_id, analysis_handler_id = simulation_handler, analysis_handler
        #
        logger.info(f'I am here 1')
        highest_profit_cash = -sys.maxsize
        highest_profit_perc = -sys.maxsize
        best_parameters = None
        #
        logger.info(f'I am here 2')

        initial_cash_total = self.runtime_settings["Portfolio_Settings.init_cash"]
        stop_after_n_epoch = self.runtime_settings["Simulation_Settings.Initial_Search_Space.stop_after_n_epoch"]
        save_every_nth_chunk = self.runtime_settings["Simulation_Settings.save_every_nth_chunk"]
        #
        # If metrics record empty then initiate
        if not any(self.metrics_record):
            logger.info(f'I am here 3')

            self._initiate_metric_records(add_ids=True, params_size=self.parameters_record_length * len(
                self.asset_names))  # we need n_assets as many metric elements as there are trial.params
        else:
            logger.info(f'I am here 4')

            highest_profit_perc = np.max(self.metrics_record["Total Return [%]"])
            #
            logger.info(f'I am here 5')

            highest_profit_cash = highest_profit_perc * initial_cash_total / 100
            # self.parameters_record[]

        if not self.user_pick:
            logger.info(f'I am here 6')

            # Get an array of indexes remaining to compute
            from Utilities.general_utilities import fetch_non_filled_elements_indexes
            # Since metric_record is n_assets times bigger than parameters_record,and because metrics record just
            #   repeats every 1/n_assets of the array we only need the first portion it
            trials_ids_not_computed = fetch_non_filled_elements_indexes(
                self.metrics_record[:self.parameters_record_length])
            logger.info(f'I am here 7')

            # Take elements from parameter record that match with trials_ids_not_computed
            # params_to_compute = np.take(ray.get(self.parameters_record), trials_ids_not_computed)
            params_to_compute = np.take(self.parameters_record, trials_ids_not_computed)
        else:
            params_to_compute = self.parameters_record
        #
        logger.info(f'I am here 8')

        # Get max n_chunks given max batch_size
        n_chunks = int(np.floor(len(params_to_compute) / batch_size)) if batch_size < len(params_to_compute) else 1
        # Split arrays into n_chunks
        logger.info(f'I am here 9')

        chunks_of_params_left_to_compute = np.array_split(params_to_compute, n_chunks)
        #
        # from Utilities.general_utilities import put_objects_list_to_ray
        # chunks_ids_of_params_left_to_compute = put_objects_list_to_ray(chunks_of_params_left_to_compute)
        # for epoch_n, epoch_params_record_id in enumerate(chunks_of_params_left_to_compute):
        for epoch_n, epoch_params_record in enumerate(chunks_of_params_left_to_compute):
            if epoch_n == stop_after_n_epoch or datetime.now() >= self.stop_sim_time:
                break
            #
            logger.info(f'I am here 10')

            start_time = perf_counter()
            CHECKTEMPS(TEMP_DICT)
            #
            long_entries, long_exits, short_entries, short_exits, \
            strategy_specific_kwargs = simulation_handler.simulate_signals(epoch_params_record)
            #
            pf = simulation_handler.simulate_events(long_entries, long_exits,
                                                    short_entries, short_exits,
                                                    strategy_specific_kwargs)
            #
            # pf = vbt.Portfolio.load(
            #     f'{self.portfolio_dir_path}/{self.runtime_settings["Portfolio_Settings.saved_pf_optimization"]}')
            '''Reconstruct Metrics from Order Records and Save'''
            logger.info('Reconstructing Portfolio Stats')
            compute_stats_timer = perf_counter()
            func_calls = []
            split_metric_names = np.array_split(self.metrics_key_names, len(self.metrics_key_names) / 3)
            pf_id = ray.put(pf)
            for metric_chunk in split_metric_names:
                func_calls.append(compute_stats_remote.remote(pf_id, metric_chunk))
            #
            # Compute All metrics in Chunk, returns [*Dataframes]
            compute_stats_results = ray.get(func_calls)

            # Join all Metrics
            portfolio_stats = compute_stats_results[0].join(compute_stats_results[1:])

            ...
            #
            # portfolio_stats = compute_stats(pf, self.metrics_key_names)
            #
            logger.info(f'Time to Reconstruct Metrics {perf_counter() - compute_stats_timer}')
            #
            highest_profit_cash, highest_profit_perc, best_parameters = _analyze_n_save(portfolio_stats,
                                                                                        epoch_params_record,
                                                                                        highest_profit_cash,
                                                                                        highest_profit_perc,
                                                                                        best_parameters,
                                                                                        initial_cash_total, epoch_n,
                                                                                        save_every_nth_chunk=save_every_nth_chunk)
            #
            self.number_of_parameters_ran = self.number_of_parameters_ran + len(epoch_params_record)
            logger.info(f'Number of parameter combinations ran: {self.number_of_parameters_ran:,}')
            #
            logger.info(f'Epoch {epoch_n} took {perf_counter() - start_time} seconds')
            logger.info(f'\n\n')
        #
        # Do a final save !
        self._save_computed_params_metrics()

    def simulate_with_post_processing(self):
        """
        In chunks/batches:
           1.  Simulate N parameters' indicators
           2.  Simulate N parameters' events
           3.  Compute Metrics
           4.  Save Results to file
        """

        def _analyze_n_save(portfolio_current, params_rec, highest_profit_cash_, highest_profit_perc_, best_parameters_,
                            initial_cash_total_, epoch_n_, save_every_nth_chunk=None):
            '''Reconstruct Metrics from Order Records and Save'''

            logger.info('Reconstructing Portfolio Stats')
            compute_stats_timer = perf_counter()
            split_metric_names = np.array_split(self.metrics_key_names, len(self.metrics_key_names) / 3)
            pf_id = ray.put(portfolio_current)
            func_calls = [compute_stats_remote.remote(pf_id, metric_chunk) for metric_chunk in split_metric_names]
            #
            # Compute All metrics in Chunk, returns [*Dataframes]
            compute_stats_results = ray.get(func_calls)
            #
            # Join all Metrics
            portfolio_metrics = compute_stats_results[0].join(compute_stats_results[1:])

            logger.info(f'Time to Reconstruct Metrics {perf_counter() - compute_stats_timer}')
            #
            tell_metrics_start_timer = perf_counter()
            #
            # Used for Printing
            highest_profit_this_epoch = portfolio_metrics['Total Return [%]'].max()
            highest_cash_profit_this_epoch = highest_profit_this_epoch * initial_cash_total_ / 100
            best_parameters_this_epoch = portfolio_metrics['Total Return [%]'].idxmax()
            #
            if highest_cash_profit_this_epoch > highest_profit_cash_:
                highest_profit_cash_ = highest_cash_profit_this_epoch
                highest_profit_perc_ = highest_profit_this_epoch
                best_parameters_ = best_parameters_this_epoch
            #
            logger.info(
                f'Highest Profit so far: {highest_profit_cash_:,}   \N{money-mouth face}\N{money bag}: '
                f'{highest_profit_perc_} of a ${initial_cash_total_:,} account')
            logger.info(f'Best Param so far: {best_parameters_}  \N{money with wings}')
            #
            logger.info(f'  -> highest_profit_cash this epoch {highest_cash_profit_this_epoch:,}')
            logger.info(f'  -> best_param this epoch {best_parameters_this_epoch}')
            #
            # clean up porfolio
            portfolio_metrics.fillna(0.0)
            for name, values in portfolio_metrics.items():
                portfolio_metrics[name] = portfolio_metrics[name].astype(float).fillna(0.0)
            #
            # Fill metric record with new metric values
            new_indexes = []
            for _index, param_record in enumerate(params_rec):
                trial_id = param_record["trial_id"]
                param_record = rm_field_from_record(param_record, 'trial_id')
                #
                for ass_index, asset in enumerate(self.asset_names):
                    param_tuple_ = tuple(param_record)
                    param_tuple_ = param_tuple_ + (asset,)
                    metrics_np = tuple(portfolio_metrics[:].loc[tuple(param_tuple_)])
                    # todo: Keep record of indexes just added and append those to file ...
                    new_index = trial_id + (self.parameters_record_length * ass_index)
                    new_indexes.append(new_index)
                    # for i in metrics_np:
                    #     logger.info(f'{i} {metrics_np[i]} {type(i)}')
                    metrics_input = (trial_id, asset) + metrics_np
                    # logger.info(f'{metrics_input}')
                    self.metrics_record[new_index] = metrics_input

            logger.info(f'Time to Analyze Metrics {perf_counter() - tell_metrics_start_timer}')
            #
            # Save Portfolio after each epoch
            logger.info(f"Saving Portfolio for Post-Processing {epoch_n_}")
            save_start_timer = perf_counter()
            file_path = next_path(f'{self.portfolio_dir_path}/pf_%s.pickle')
            logger.info(f'{file_path = }')
            portfolio_current.save(file_path)
            logger.info(f'Time to Save Portfolio {perf_counter() - save_start_timer} during epoch {epoch_n_}')
            #
            # Save the parameter and metric records to file
            logger.info(f"Saving epoch {epoch_n_}")
            save_start_timer = perf_counter()
            self._save_computed_params_metrics(new_indexes)
            #
            logger.info(f'Time to Save Records {perf_counter() - save_start_timer} during epoch {epoch_n_}')
            #
            return highest_profit_cash_, highest_profit_perc_, best_parameters_

        batch_size = self.batch_size
        # self.parameters_record_length = len(ray.get(self.parameters_record))
        self.parameters_record_length = len(self.parameters_record)
        #
        from Simulation_Handler.simulation_handler import Simulation_Handler
        # from Analysis_Handler.analysis_handler import Analysis_Handler`
        simulation_handler = Simulation_Handler(self)
        # analysis_handler = Analysis_Handler(self)
        #
        # simulation_handler_id, analysis_handler_id = put_objects_list_to_ray([simulation_handler, analysis_handler])
        # simulation_handler_id, analysis_handler_id = simulation_handler, analysis_handler
        #
        highest_profit_cash = -sys.maxsize
        highest_profit_perc = -sys.maxsize
        best_parameters = None
        #
        initial_cash_total = self.runtime_settings["Portfolio_Settings.init_cash"]
        stop_after_n_epoch = self.runtime_settings["Simulation_Settings.Initial_Search_Space.stop_after_n_epoch"]
        save_every_nth_chunk = self.runtime_settings["Simulation_Settings.save_every_nth_chunk"]
        #
        # If metrics record empty then initiate
        if not any(self.metrics_record):
            self._initiate_metric_records(add_ids=True, params_size=self.parameters_record_length * len(
                self.asset_names))  # we need n_assets as many metric elements as there are trial.params
        else:
            highest_profit_perc = np.max(self.metrics_record["Total Return [%]"])
            #
            highest_profit_cash = highest_profit_perc * initial_cash_total / 100
            # self.parameters_record[]

        if not self.user_pick:
            # Get an array of indexes remaining to compute
            from Utilities.general_utilities import fetch_non_filled_elements_indexes
            # Since metric_record is n_assets times bigger than parameters_record,and because metrics record just
            #   repeats every 1/n_assets of the array we only need the first portion it
            trials_ids_not_computed = fetch_non_filled_elements_indexes(
                self.metrics_record[:self.parameters_record_length])

            # Take elements from parameter record that match with trials_ids_not_computed
            # params_to_compute = np.take(ray.get(self.parameters_record), trials_ids_not_computed)
            params_to_compute = np.take(self.parameters_record, trials_ids_not_computed)
        else:
            params_to_compute = self.parameters_record
        #

        # Get max n_chunks given max batch_size
        n_chunks = int(np.floor(len(params_to_compute) / batch_size)) if batch_size < len(params_to_compute) else 1
        # Split arrays into n_chunks
        chunks_of_params_left_to_compute = np.array_split(params_to_compute, n_chunks)
        #
        # from Utilities.general_utilities import put_objects_list_to_ray
        # chunks_ids_of_params_left_to_compute = put_objects_list_to_ray(chunks_of_params_left_to_compute)
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
            pf = simulation_handler.simulate_events(long_entries, long_exits,
                                                    short_entries, short_exits,
                                                    strategy_specific_kwargs)
            #
            '''Reconstruct Metrics from Order Records and Save'''
            #
            highest_profit_cash, highest_profit_perc, best_parameters = _analyze_n_save(
                pf,
                epoch_params_record,
                highest_profit_cash,
                highest_profit_perc,
                best_parameters,
                initial_cash_total, epoch_n,
                save_every_nth_chunk=save_every_nth_chunk)
            #
            self.number_of_parameters_ran = self.number_of_parameters_ran + len(epoch_params_record)
            logger.info(f'Number of parameter combinations ran: {self.number_of_parameters_ran:,}')
            #
            logger.info(f'Epoch {epoch_n} took {perf_counter() - start_time} seconds')
            logger.info(f'\n\n')
        #
        # Do a final save !
        self._save_computed_params_metrics()

    def prepare_backtest(self):
        """Simulate parameters passed by user; either explicitly or produced from settings"""
        ...
        self._initiate_parameters_records(add_ids=True)

    def _metric_record_to_tsv(self):
        original_file_path = self.path_of_initial_metrics_record if not self.user_pick else self.file_name_of_backtest_results
        #
        original_file_path_without_extension = original_file_path.rsplit('.', 1)[:-1]
        df = pd.DataFrame(self.metrics_record).set_index('trial_id')
        df.to_csv(f'{original_file_path_without_extension}.tsv', delimiter='\t')

    def metric_csv_file_to_tsv(self):
        original_file_path = self.path_of_initial_metrics_record if not self.user_pick else self.file_name_of_backtest_results
        #
        if os.path.exists(original_file_path):
            metrics_df = pd.read_csv(original_file_path)
            #
            logger.info(f'Loaded metrics csv')
            #
            original_file_path_without_extension = original_file_path.rsplit('.', 1)[0]
            #
            metrics_df.set_index('trial_id', inplace=True)
            metrics_df.to_csv(f'{original_file_path_without_extension}.tsv', sep='\t')
            #
            logger.info(f'Successfully created tsv file from csv file')
        #
        else:
            logger.info(f'Could not find {original_file_path} to convert to tsv')
