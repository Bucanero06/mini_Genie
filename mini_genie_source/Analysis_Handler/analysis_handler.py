import numpy as np
from logger_tt import logger

from Configuration_Files.vbtmetricsdictionary import vbtmetricsdictionary


class Analysis_Handler:
    """


    Args:
         ():
         ():

    Attributes:
         ():
         ():

    Note:

    """

    def __init__(self, genie_object):
        """Constructor for Analysis_Handler"""

        self.genie_object = genie_object

    def print_dict(self, optional_object=None):
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    def compute_stats(self, Portfolio, only_these_stats=None, groupby=None):
        logger.debug('Compute Stats')
        if not only_these_stats:
            if not groupby:
                portfolio_combined = Portfolio.stats(agg_func=None).replace([np.inf, -np.inf], np.nan,
                                                                            inplace=False)
            else:
                portfolio_combined = Portfolio.stats(agg_func=None, group_by=groupby).replace([np.inf, -np.inf], np.nan,
                                                                                              inplace=False)
        else:
            logger.debug('Only computing called metrics found in setting Loss_Metrics')
            only_these_stats = [vbtmetricsdictionary[string] for string in only_these_stats]
            if not groupby:
                portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None).replace(
                    [np.inf, -np.inf], np.nan, inplace=False)
            else:
                portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None, group_by=groupby).replace(
                    [np.inf, -np.inf], np.nan, inplace=False)
        return portfolio_combined
