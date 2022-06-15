#!/usr/bin/env python3.9
import gc
from time import perf_counter

import numpy as np
import pandas as pd
import ray
import vectorbtpro as vbt
from logger_tt import logger

from Run_Time_Handler.vbtmetricsdictionary import vbtmetricsdictionary


@vbt.chunked(
    n_chunks=1,
    # size=vbt.LenSizer(arg_query='only_these_stats'),
    size=vbt.ShapeSizer(arg_query="pf_size", axis=0),
    # size=Porfolio.wrapper.columns.shape[0],
    arg_take_spec=dict(
        Portfolio=vbt.ChunkSlicer(),
        only_these_stats=None,
        groupby=None,
        pf_size=None,
    ),
    merge_func=lambda x: pd.concat(x).vbt.sort_index(),
    show_progress=True,
    # engine='ray'
    engine='ray',
    init_kwargs={
        'address': 'auto',
        # 'num_cpus': 28,
        # 'memory': 100 * 10 ** 9,
        # 'object_store_memory': 100 * 10 ** 9,
    },
)
def compute_stats_chunked(Portfolio, only_these_stats, groupby=None, pf_size=None):
    portfolio_combined = Portfolio.stats(metrics=only_these_stats, agg_func=None).replace(
        [np.inf, -np.inf], np.nan, inplace=False)
    return portfolio_combined


def compute_stats(Portfolio, metrics_key_names, groupby=None, ray_init_cpu_count=1):
    "Chunked_method"
    logger.info('Reconstructing Portfolio Stats')
    compute_stats_timer = perf_counter()
    metric_names = [vbtmetricsdictionary[string] for string in metrics_key_names]
    pf_size = Portfolio.wrapper.columns.shape[0]
    chunks_ = int(np.floor(pf_size / 2))  # int(np.floor(pf_size / ray_init_cpu_count * len(metric_names)))
    compute_stats_chunked.options.n_chunks = chunks_ if chunks_ >= 1 else 1
    compute_stats_chunked.options.size = pf_size
    portfolio_combined = compute_stats_chunked(Portfolio=Portfolio, only_these_stats=metric_names)
    logger.info(f'Time to Reconstruct Metrics {perf_counter() - compute_stats_timer}')
    assert portfolio_combined.shape == (Portfolio.wrapper.columns.shape[0], len(metrics_key_names))
    logger.info(f'{portfolio_combined.shape = }')
    logger.info(f'{Portfolio.wrapper.columns.shape[0] = }')
    logger.info(f'{len(metrics_key_names) = }')
    return portfolio_combined


@ray.remote
def compute_stats_remote(Portfolio, only_these_stats, groupby=None):
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
