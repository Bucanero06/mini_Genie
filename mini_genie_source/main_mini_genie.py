#!/usr/bin/env python3
import warnings
from logger_tt import setup_logging, logger

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def call_genie(run_time_settings, arg_parser_values):
    # Initiate the genie object
    from mini_genie_source.mini_Genie_Object.mini_genie import mini_genie_trader
    '''
    The genie object works as an operator, can act on itself through its methods, can be acted upon by other 
        operators, and must always return the latest state of genie_operator.
     '''

    genie_object = mini_genie_trader(runtime_kwargs=run_time_settings, args=arg_parser_values)

    if arg_parser_values.metrics_to_tsv:
        genie_object.metric_csv_file_to_tsv()

    # Load symbols_data, open, low, high, close to genie object.
    genie_object.fetch_and_prepare_input_data()

    if arg_parser_values.genie_pick:
        # todo update with new changes  in dev branch
        '''
         List of Initial Params:
              Product of:
                  All Categorical Params
                  Use a grid-like approach to windows for indicators
                  For TP and SL use the avg ATR for the 3 months prior to the optimization date window for every
                      timeframe (when possible do this separately for upwards, downwards and sideways, then use
                      these values separately during the strategy or average them for a single value) then:
                          1.  Using \bar{ATR}(TF), define multiple TP_0 and SL_0 for and scale with n ratios [ 0.5, 1, 1.5, 2, 2.5]
                          2.  Use (TP/SL) \gamma ratios like [ 1, 1.2, 1.5, 1.75, 2, 2.5, 3]
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.5, -> R=\bar{ATR}(TF='1h') * n=500
                                  ==> TP=750 & SL=-500)
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.0, -> R=\bar{ATR}(TF='1h') * n=500
                                  ==> TP=500 & SL=-500)
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=0.5, -> R=\bar{ATR}(TF='1h') * n=500
                                  ==> TP=500 & SL=-750
                                  
                              (e.g. -> \bar{ATR}(TF='1d')=2600, n=1 and \gamma=1, -> R=\bar{ATR}(TF='1h') * n=2600
                                  ==> TP=2600 & SL=-2600
                                  
            Run product of unique param values in each category, and remove out of bound tp and sl combinations.
          '''

        # Determine initial search space size and content
        #   Initiates:       genie_object._initiate_parameters_records
        #                    genie_object._initiate_metric_records
        #   Fills:           genie_object._initiate_parameters_records
        genie_object.suggest_parameters()

        if run_time_settings["Simulation_Settings"]["run_mode"] == "ludicrous":
            """
            # In chunks/batches:
            #    1.  Simulate N parameters' indicators
            #    2.  Simulate N parameters' events
            #    3.  Compute Metrics
            #    4.  Save Results to file
            """
            genie_object.simulate()
        # elif run_time_settings["Simulation_Settings"]["run_mode"] == "plaid_plus":
        #     #
        #     genie_object. -...

        else:
            logger.error("The given run_mode is not known, please refer to documentation for accepted inputs")
    #
    elif arg_parser_values.user_pick:
        genie_object.prepare_backtest()
        #
        genie_object.simulate()
    #
    from mini_genie_source.Utilities.general_utilities import is_empty_dir
    if arg_parser_values.post_analysis and not is_empty_dir(genie_object.portfolio_dir_path):
        from Utilities.general_utilities import Execute
        # self.args.post_analysis_path
        from os.path import abspath
        study_dir_abs_path = abspath(genie_object.study_dir_path)
        Execute(f'python {arg_parser_values.post_analysis_path} -s {study_dir_abs_path} ')
    #
    logger.info('All Done')


if __name__ == "__main__":
    from Run_Time_Handler.run_time_handler import run_time_handler

    setup_logging(full_context=1)

    #
    run_time_handler = run_time_handler(run_function=call_genie)
    run_time_handler.call_run_function()
