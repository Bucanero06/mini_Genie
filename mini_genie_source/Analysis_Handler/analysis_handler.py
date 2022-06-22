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
    n_chunks=None,
    # size=vbt.LenSizer(arg_query='only_these_stats'),
    size=vbt.ShapeSizer(arg_query="pf_size", axis=0),
    # size=Porfolio.wrapper.columns.shape[0],
    arg_take_spec=dict(
        # pf=vbt.ChunkSlicer(),
        pf=vbt.ChunkSlicer,
        only_these_stats=None,
        pf_size=None,
    ),
    merge_func=lambda x: pd.concat(x, axis=0),  #
    show_progress=True,
    engine='ray',
    init_kwargs={
        # 'address': 'auto',
        # 'num_cpus': 28,
        # 'memory': 100 * 10 ** 9,
        # 'object_store_memory': 100 * 10 ** 9,
    },
)
def _compute_stats_chunked(pf, only_these_stats, pf_size):
    pf_combined = pf.stats(metrics=only_these_stats, agg_func=None).replace(
        [np.inf, -np.inf], np.nan, inplace=False)
    # ###################
    repeated_params_eg_keys = pf_combined["Total Return [%]"].keys()
    repeated = []
    not_repeated = []
    for param in repeated_params_eg_keys:
        if param not in not_repeated:
            not_repeated.append(param)
        else:
            repeated.append(param)
    #
    assert len(repeated) == 0
    # ###################
    return pf_combined


def compute_stats(Portfolio, metrics_key_names):
    # repeated = []
    # not_repeated = []
    # for param in Portfolio.wrapper.keys():
    #     if param not in not_repeated:
    #         not_repeated.append(param)
    #     else:
    #         repeated.append(param)
    #
    # logger.info(f'Repeated params {repeated}')
    # logger.info(f'Repeated params {len(repeated)}')
    #
    # exit()

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    "Chunked_method"
    logger.info('Reconstructing Portfolio Stats')
    compute_stats_timer = perf_counter()
    metric_names = [vbtmetricsdictionary[string] for string in metrics_key_names]
    pf_size = Portfolio.wrapper.columns.shape[0]
    chunks_ = int(np.floor(pf_size / 2))  # int(np.floor(pf_size / ray_init_cpu_count * len(metric_names)))
    chunks_ = 28
    _compute_stats_chunked.options.n_chunks = chunks_ if chunks_ >= 1 else 1
    _compute_stats_chunked.options.size = pf_size

    portfolio_combined = _compute_stats_chunked(pf=Portfolio, only_these_stats=metric_names, pf_size=pf_size)
    logger.info(f'Time to Reconstruct Metrics {perf_counter() - compute_stats_timer}')
    # ###################
    repeated_params_eg_keys = portfolio_combined["Total Return [%]"].keys()
    repeated = []
    not_repeated = []
    for param in repeated_params_eg_keys:
        if param not in not_repeated:
            not_repeated.append(param)
        else:
            repeated.append(param)
    logger.info(f'Repeated all params {len(repeated)}')
    exit()
    # ###################
    # ###################

    repeated = []
    not_repeated = []
    for porfolio_stats in portfolio_combined:
        repeated_params_eg_keys = porfolio_stats["Total Return [%]"].keys()
        for param in repeated_params_eg_keys:
            if param not in not_repeated:
                not_repeated.append(param)
            else:
                repeated.append(param)
    logger.info(f'Repeated [i] params {len(repeated)}')
    exit()
    # ###################
    #
    if not isinstance(portfolio_combined, pd.DataFrame):
        portfolio_combined = pd.concat(portfolio_combined)
    #
    portfolio_combined = portfolio_combined.sort_index()
    logger.info(f'{portfolio_combined.shape = } {Portfolio.wrapper.columns.shape[0], len(metrics_key_names)}')

    # ###################
    repeated_params_eg_keys = portfolio_combined["Total Return [%]"].keys()
    repeated = []
    not_repeated = []
    for param in repeated_params_eg_keys:
        if param not in not_repeated:
            not_repeated.append(param)
        else:
            repeated.append(param)
    logger.info(f'Repeated all params {len(repeated)}')
    exit()
    # ###################
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #
    # compute_stats_timer = perf_counter()
    # metric_names = [vbtmetricsdictionary[string] for string in metrics_key_names]
    # portfolio_combined = Portfolio.stats(metrics=metric_names, agg_func=None).replace(
    #     [np.inf, -np.inf], np.nan, inplace=False)
    # logger.info(f'Time to normal Reconstruct Metrics {perf_counter() - compute_stats_timer}')
    #
    # portfolio_combined = portfolio_combined.vbt.sort_index()
    # logger.info(f'normal {portfolio_combined.shape = } {Portfolio.wrapper.columns.shape[0], len(metrics_key_names)}')

    exit()
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
