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
                            action='store_true', default=False)
        parser.add_argument("-up", help="Simulate using solely the user picked space", dest="user_pick",
                            action='store_true', default=False)
        parser.add_argument("-tsv",
                            help="Convert csv to tsv previously computed metric files. File will vary based on "
                                 "whether user or genie pick option was used.",
                            dest="metrics_to_tsv", action='store_true', default=False)
        parser.add_argument("-c",
                            help="Point to Run-Time-Parameters (a.k.a settings) file",
                            dest="run_time_file_name", action='store',
                            # default=False
                            default='debugging_runtime_parameters.py'
                            )
        parser.add_argument("--example",
                            help="Creates example Run-Time-Parameters (a.k.a settings) file in current directory",
                            dest="create_example_file", action='store_true', default=False)

        self.parser = parser
        self.parser.set_defaults(func=run_function)
        self.args = self.parser.parse_args()
        #
        if self.args.create_example_file:
            self.create_example()
            exit()
        #
        elif self.args.run_time_file_name and path.exists(self.args.run_time_file_name):
            # Check that a command has been passed
            if not any([vars(self.args)[i] for i in vars(self.args) if i not in ['func', 'run_time_file_name']]):
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

    def fetch_run_time_settings(self):
        """
        Adds self.run_time_settings
        @return self.run_time_settings (to be used if needed)
        """
        import importlib.util
        import sys
        #
        logger.info(f"Loading Run_Time_Settings from file {self.args.run_time_file_name}")
        module_name = path.basename(self.args.run_time_file_name)
        module_path = self.args.run_time_file_name
        #
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = foo
        spec.loader.exec_module(foo)

        self.run_time_settings = foo.Run_Time_Settings

        return self.run_time_settings

    def call_run_function(self):
        self.args.func(self.run_time_settings, self.args)
