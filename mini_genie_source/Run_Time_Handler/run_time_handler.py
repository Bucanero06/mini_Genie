#!/usr/bin/env python3
import argparse
from os import path

from logger_tt import logger


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
        #
        parser.add_argument("-gp", help="Simulate using genie picked space based on user settings", dest="genie_pick",
                            action='store_true', default=True)
        parser.add_argument("-up", help="Simulate using solely the user picked space", dest="user_pick",
                            action='store_true', default=False)
        parser.add_argument("-tsv",
                            help="Convert csv to tsv previously computed metric files. File will vary based on "
                                 "whether user or genie pick option was used.",
                            dest="metrics_to_tsv", action='store_true', default=False)
        parser.add_argument("-c",
                            help="Point to Run-Time-Parameters (a.k.a settings) dictionary path",
                            dest="run_time_dictionary_path", action='store',
                            # default=False
                            default='debugging_config.py.Run_Time_Settings'
                            )
        parser.add_argument("--example",
                            help="Creates example Run-Time-Parameters (a.k.a settings) file in current directory",
                            dest="create_example_file", action='store_true', default=False)

        self.parser = parser
        self.parser.set_defaults(func=run_function)
        self.args = self.parser.parse_args()
        self.run_time_module_path, self.run_time_dictionary_name = self.args.run_time_dictionary_path.rsplit('.', 1)
        #
        if self.args.create_example_file:
            self.create_example()
            exit()
        #
        elif self.args.run_time_dictionary_path and path.exists(self.run_time_module_path):
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
            exit()
        #

    def create_example(self):
        logger.info("Creating example_genie_settings.py")
        import shutil
        shutil.copy2("mini_genie_source/Run_Time_Handler/example_genie_settings.py",
                     "example_genie_settings.py")

    @staticmethod
    def load_module_from_path(filename, object_name=None):
        # import importlib.util
        # import sys
        # #
        # logger.info(f"Loading Run_Time_Settings from file {filename}")
        # module_name = path.basename(filename)
        # #
        # spec = importlib.util.spec_from_file_location(module_name, filename)
        # foo = importlib.util.module_from_spec(spec)
        # sys.modules[module_name] = foo
        # spec.loader.exec_module(foo)
        from importlib import import_module

        module_path = filename.rsplit('.', 1)[0]
        module = module_path.replace("/", ".")

        mod = import_module(module)

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

        # run_time_settings = foo.Run_Time_Settings
        # run_time_settings = self.load_module_from_path(self.args.run_time_dictionary_path).Run_Time_Settings

        run_time_settings = self.load_module_from_path(self.run_time_module_path,
                                                       object_name=self.run_time_dictionary_name)
        #
        backtest_sim_path = run_time_settings["Portfolio_Settings"]["Simulator"]["backtesting"]
        backtest_sim_module_path_module_path, backtest_sim_dictionary_name = backtest_sim_path.rsplit('.', 1)
        backtest_sim = self.load_module_from_path(backtest_sim_module_path_module_path,
                                                  object_name=backtest_sim_dictionary_name)
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
        run_time_settings["Portfolio_Settings"]["Simulator"]["backtesting"] = backtest_sim
        run_time_settings["Portfolio_Settings"]["Simulator"]["optimization"] = optimization_sim
        #
        self.run_time_settings = run_time_settings

        return self.run_time_settings

    def call_run_function(self):
        self.args.func(self.run_time_settings, self.args)