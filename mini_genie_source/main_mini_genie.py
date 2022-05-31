import argparse
import warnings

from logger_tt import setup_logging, logger

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO:
#  TASK:
#   * Need to add backtest module/logic (can use np.arrange) -----------------------------
#   * Fill, save and load params,  metrics records (convinient file format) -------------------------
#   * Do not compute or include in chunks, previously computed params---------------------
#   * ADD EQUIPMENT HANDLER -------------------------------------------------

def call_genie(args):
    genie_pick = args.genie_pick
    user_pick = args.user_pick
    metrics_to_tsv = args.metrics_to_tsv
    #
    from mini_genie_source.Configuration_Files.runtime_parameters import Run_Time_Settings
    from mini_genie_source.mini_Genie_Object.mini_genie import mini_genie_trader

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
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.5, -> R=\bar{ATR}(TF='1h')/n=500
                                  ==> TP=750 & SL=-500)
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.0, -> R=\bar{ATR}(TF='1h')/n=500
                                  ==> TP=500 & SL=-500)
                              (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=0.5, -> R=\bar{ATR}(TF='1h')/n=500
                                  ==> TP=500 & SL=-750)
          Run product of unique param values in each category, use the best N params params as the starting seeds
              for Optimization ... next
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


if __name__ == "__main__":
    setup_logging(full_context=1)
    #
    parser = argparse.ArgumentParser(description="Help for ChargeMigration Interface")
    #
    parser.add_argument("-gp", help="Will simulate using genie picked space based on user settings", dest="genie_pick",
                        type=str, action='store', default=False)
    parser.add_argument("-up", help="Will simulate using solely the user picked space", dest="user_pick", type=str,
                        action='store', default=False)
    parser.add_argument("-tsv",
                        help="Will convert csv to tsv previously computed metric files. File will vary based on "
                             "whether user or genie pick option was used",
                        dest="metrics_to_tsv", type=str, action='store', default=False)
    #
    parser.set_defaults(func=call_genie)
    args = parser.parse_args()
    #
    if not any([vars(args)[i] for i in vars(args) if i != 'func']):
        logger.warning("No action requested, exiting ...")
        parser.print_help()
        exit()
    #
    args.func(args)
    #
    logger.info('All Done')
