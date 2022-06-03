#!/usr/bin/env python3.9
import gc
from time import perf_counter

import numpy as np
import ray
from Configuration_Files.vbtmetricsdictionary import vbtmetricsdictionary
from logger_tt import logger


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

    @staticmethod
    def compute_stats(Portfolio, only_these_stats, groupby=None):
        logger.info('Reconstructing Portfolio Stats')
        print('Reconstructing Portfolio Stats')
        start_time = perf_counter()
        only_these_stats = [vbtmetricsdictionary[string] for string in only_these_stats]
        if groupby is None:
            portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None).replace(
                [np.inf, -np.inf], np.nan, inplace=False)
        else:
            portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None, group_by=groupby).replace(
                [np.inf, -np.inf], np.nan, inplace=False)
        logger.info(f'Time to Reconstruct Metrics {perf_counter() - start_time}')
        return portfolio_combined

    # @ray.remote
    # def compute_stats(self, Portfolio, only_these_stats=None, groupby=None):
    #     logger.info('Reconstructing Portfolio Stats')
    #     print('Reconstructing Portfolio Stats')
    #     start_time = perf_counter()
    #     if only_these_stats == None:
    #         if not groupby:
    #             portfolio_combined = Portfolio.stats(agg_func=None).replace([np.inf, -np.inf], np.nan,
    #                                                                         inplace=False)
    #         else:
    #             portfolio_combined = Portfolio.stats(agg_func=None, group_by=groupby).replace([np.inf, -np.inf], np.nan,
    #                                                                                           inplace=False)
    #     else:
    #         logger.debug('Only computing called metrics found in setting Loss_Metrics')
    #         only_these_stats = [vbtmetricsdictionary[string] for string in only_these_stats]
    #         if groupby == None:
    #             portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None).replace(
    #                 [np.inf, -np.inf], np.nan, inplace=False)
    #         else:
    #             portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None, group_by=groupby).replace(
    #                 [np.inf, -np.inf], np.nan, inplace=False)
    #     logger.info(f'Time to Reconstruct Metrics {perf_counter() - start_time}')
    #     print(f'Time to Reconstruct Metrics {perf_counter() - start_time}')
    #     return portfolio_combined


@ray.remote
def compute_stats(Portfolio, only_these_stats, groupby=None):
    gc.collect()
    only_these_stats = [vbtmetricsdictionary[string] for string in only_these_stats]
    if groupby is None:
        portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None).replace(
            [np.inf, -np.inf], np.nan, inplace=False)
    else:
        portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None, group_by=groupby).replace(
            [np.inf, -np.inf], np.nan, inplace=False)
    gc.collect()
    return portfolio_combined
