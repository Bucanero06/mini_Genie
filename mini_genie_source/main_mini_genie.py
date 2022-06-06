#!/usr/bin/env python3
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def call_genie(args):
    from logger_tt import setup_logging, logger
    setup_logging(full_context=1)
    if not any([vars(args)[i] for i in vars(args) if i != 'func']):
        logger.warning("No action requested, exiting ...")
        parser.print_help()
        exit()
    #
    genie_pick = args.genie_pick
    user_pick = args.user_pick
    metrics_to_tsv = args.metrics_to_tsv
    #
    from Configuration_Files.runtime_parameters import Run_Time_Settings
    from mini_Genie_Object.mini_genie import mini_genie_trader

    # Initiate the genie object
    '''
    The genie object works as an operator, can act on itself through its methods, can be acted upon by other 
        operators, and must always return the latest state of genie_operator.
     '''
    genie_object = mini_genie_trader(runtime_kwargs=Run_Time_Settings, user_pick=user_pick)

    if metrics_to_tsv:
        genie_object.metric_csv_file_to_tsv()

    # Load symbols_data, open, low, high, close to genie object.
    genie_object.fetch_and_prepare_input_data()

    if genie_pick:
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
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.0, -> R=\bar{ATR}(TF='1h')/n=500
                                  ==> TP=500 & SL=-500)
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=0.5, -> R=\bar{ATR}(TF='1h')/n=500
                                  ==> TP=500 & SL=-750
                                  
                              (e.g. -> \bar{ATR}(TF='1d')=2600, n=1 and \gamma=1, -> R=\bar{ATR}(TF='1h')/n=2600
                                  ==> TP=2600 & SL=-2600
                                  
            Run product of unique param values in each category, and remove out of bound tp and sl combinations.
          '''

        # Determine initial search space size and content
        #   Initiates:       genie_object._initiate_parameters_records
        #                    genie_object._initiate_metric_records
        #   Fills:           genie_object._initiate_parameters_records
        genie_object.suggest_parameters()

        # In chunks/batches:
        #    1.  Simulate N parameters' indicators
        #    2.  Simulate N parameters' events
        #    3.  Compute Metrics
        #    4.  Save Results to file
        genie_object.simulate()
    #
    elif user_pick:
        genie_object.prepare_backtest()
        #
        genie_object.simulate()
    #
    logger.info('All Done')


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description="Help for mini-Genie Trader")
    #
    parser.add_argument("-gp", help="Simulate using genie picked space based on user settings", dest="genie_pick",
                        action='store_true', default=True)
    parser.add_argument("-up", help="Simulate using solely the user picked space", dest="user_pick",
                        action='store_true', default=False)
    parser.add_argument("-tsv",
                        help="Will convert csv to tsv previously computed metric files. File will vary based on "
                             "whether user or genie pick option was used.",
                        dest="metrics_to_tsv", action='store_true', default=False)

    parser.set_defaults(func=call_genie)
    args = parser.parse_args()
    #
    # if not os.path.exists(".mini_genie"):
    #     print(
    #         "Wither re-clone repository or if only .mini_genie is missing, use $touch .mini_genie under \"mini_Genie\"")
    # if args.setup_bool:
    #     from Utilities.set_up_genie import set_up_mini_genie
    #
    #     set_up_mini_genie()
    #     parser.print_help()
    #     exit()

    # import pandas as pd
    # from datetime import datetime
    #
    # custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
    # df = pd.read_csv('Datas/DAX.csv',
    # df = pd.read_csv('Datas/DAX.csv',
    # df = pd.read_csv('Datas/DAX.csv',
    #                  parse_dates=['Datetime'],
    #                  date_parser=custom_date_parser)
    #
    # df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime("%m.%d.%Y %H:%M:%S")
    # df.set_index("Datetime",inplace=True)
    # df.to_csv('Datas/DAXC_E.csv')
    # print(df)
    # exit()
    #
    args.func(args)
    #
