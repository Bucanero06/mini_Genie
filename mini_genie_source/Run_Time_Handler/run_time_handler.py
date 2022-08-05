#!/usr/bin/env python3
import argparse

from logger_tt import logger

GP_DEFAULT = False
UP_DEFAULT = False
POST_ANALYSIS_DEFAULT = False
TSV_DEFAULT = False
CONFIG_FILE_DEFAULT = False
EXAMPLE_CONFIG_PATH = "mini_genie_source/Run_Time_Handler/example_genie_settings.py"


# CONFIG_FILE_DEFAULT = "mmt_DAXUSD_config.py.Run_Time_Settings"
# CONFIG_FILE_DEFAULT = "mmt_debug.py.debug_settings"


class run_time_handler:
    """


    Args:
         ():
         ():

    Attributes:
         ():
         ():

    Note:

    """

    def __init__(self, run_function):
        """Constructor for run_time_handler"""
        from Utilities.general_utilities import print_dict
        self.print_dict = print_dict

        #
        parser = argparse.ArgumentParser(description="Help for mini-Genie Trader")
        general_group = parser.add_argument_group(description="Basic Usage")
        # expand_study_group = parser.add_argument_group(description="Expand Study Usage")
        #
        general_group.add_argument("-gp", help="Simulate using genie picked space based on user settings",
                                   dest="genie_pick",
                                   action='store_true', default=GP_DEFAULT)
        general_group.add_argument("-up", help="Simulate using solely the user picked space", dest="user_pick",
                                   action='store_true',
                                   default=UP_DEFAULT)
        general_group.add_argument("-pa", help="Calls Genie's post analysis module", dest="post_analysis",
                                   action='store_true',
                                   default=POST_ANALYSIS_DEFAULT)
        # default=True)
        general_group.add_argument("-tsv",
                                   help="Convert csv to tsv previously computed metric files. File will vary based on "
                                        "whether user or genie pick option was used.",
                                   dest="metrics_to_tsv", action='store_true', default=TSV_DEFAULT)
        general_group.add_argument("-c", "-config_file_path",
                                   help="Point to Run-Time-Parameters (a.k.a settings) dictionary path",
                                   dest="run_time_dictionary_path", action='store',
                                   default=CONFIG_FILE_DEFAULT
                                   )
        #
        general_group.add_argument("--example",
                                   help="Creates example Run-Time-Parameters (a.k.a settings) file in current "
                                        "directory",
                                   dest="create_example_file", action='store_true', default=False)
        # #
        # expand_study_group.add_argument("--expand_study_params",
        #                                 help="Provide path to parameter's file you want to add to current study "
        #                                      "(repeated params will be ignored) *not-available",
        #                                 dest="expand_study_params_path", action='store_true', default=False)
        # expand_study_group.add_argument("--expand_study_metrics",
        #                                 help="Provide path to metrics' file you want to add to current study "
        #                                      "(repeated params will be ignored) *not-available",
        #                                 dest="expand_study_metrics_path", action='store_true', default=False)

        self.parser = parser
        self.parser.set_defaults(func=run_function)
        self.args = self.parser.parse_args()
        #
        if self.args.create_example_file:
            self.create_example()
            exit()
        #
        elif self.args.run_time_dictionary_path:
            self.run_time_module_path, self.run_time_dictionary_name = self.args.run_time_dictionary_path.rsplit('.', 1)
            # Check that a command has been passed
            if not any([vars(self.args)[i] for i in vars(self.args) if i not in ['func', 'run_time_dictionary_path']]):
                logger.warning("No action requested, exiting ...")
                parser.print_help()
                exit()
            #
            self.fetch_run_time_settings()
        #
        else:
            logger.warning(
                "Please pass an existing genie configuration file using -c or create one with defaults using --example")
            parser.print_help()

            exit()
        #
        if self.args.post_analysis:
            # fetch paths to add-ons for Genie
            from genie_add_ons_paths import genie_add_ons_paths
            self.args.post_analysis_path = genie_add_ons_paths["post_analysis_main"]
        #

    def create_example(self):
        logger.info("Creating example_genie_settings.py")
        import shutil
        shutil.copy2(EXAMPLE_CONFIG_PATH, ".")

    @staticmethod
    def load_module_from_path(filename, object_name=None):
        module_path = filename.rsplit('.', 1)[0]
        module = module_path.replace("/", ".")

        from importlib import import_module
        mod = import_module(module)

        ###
        # import importlib.util
        # import sys
        # from os import path
        # #
        # logger.info(f"Loading Run_Time_Settings from file {filename}")
        # module_name = path.basename(module_path)
        # #
        # spec = importlib.util.spec_from_file_location(module_name, filename)
        # mod = importlib.util.module_from_spec(spec)
        # sys.modules[module_name] = mod
        # spec.loader.exec_module(mod)
        ###

        if object_name is not None:
            met = getattr(mod, object_name)
            return met
        else:
            return mod

    def fetch_run_time_settings(self):
        """
        Adds self.run_time_settings
        @return self.run_time_settings (to be used if needed)
        """

        run_time_settings = self.load_module_from_path(self.run_time_module_path,
                                                       object_name=self.run_time_dictionary_name)

        #
        strategy_path = run_time_settings["Portfolio_Settings"]["Simulator"]["Strategy"]

        strategy_settings_dict = self.load_module_from_path(strategy_path, object_name="Strategy_Settings")
        strategy_settings_dict["Strategy"] = f'{strategy_path}.{strategy_settings_dict["Strategy"]}'
        run_time_settings["Strategy_Settings"] = strategy_settings_dict
        #
        optimization_sim_path = run_time_settings["Portfolio_Settings"]["Simulator"]["optimization"]
        optimization_sim_module_path, optimization_sim_dictionary_name = optimization_sim_path.rsplit('.', 1)
        optimization_sim = self.load_module_from_path(optimization_sim_module_path,
                                                      object_name=optimization_sim_dictionary_name)
        #
        strategy_sim_path = run_time_settings["Strategy_Settings"]["Strategy"]
        strategy_sim_module_path, strategy_sim_dictionary_name = strategy_sim_path.rsplit('.', 1)
        strategy_sim = self.load_module_from_path(strategy_sim_module_path,
                                                  object_name=strategy_sim_dictionary_name)
        #
        run_time_settings["Strategy_Settings"]["Strategy"] = strategy_sim
        run_time_settings["Portfolio_Settings"]["Simulator"]["optimization"] = optimization_sim
        #
        self.run_time_settings = run_time_settings

        return self.run_time_settings

    def call_run_function(self):
        import pprint
        self.args.func(self.run_time_settings, self.args)
