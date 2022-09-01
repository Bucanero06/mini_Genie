#!/usr/bin/env python3
import gc
import warnings

from mini_genie_source.Utilities.general_utilities import get_objects_list_from_ray

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- ↓ Do not remove these libs ↓ -------------------------------------------------------------------------------------
from time import perf_counter
from logger_tt import logger


# --- ↑ Do not remove these libs ↑ -------------------------------------------------------------------------------------


class Simulation_Handler:
    """
    Takes care of simulations

    Args:
         ():
         ():

    Attributes:
         ():
         ():

    Note:
        this needs work lol
    """

    def __init__(self, genie_object):
        """Constructor for Simulation_Handler"""
        self.genie_object = genie_object
        from mini_genie_source.Utilities.general_utilities import print_dict
        self.print_dict = print_dict

    @property
    def compute_bar_atr(self) -> object:
        """

        Returns:
            object: 

        """
        from mini_genie_source.Simulation_Handler.compute_bar_atr import compute_bar_atr
        bar_atr = compute_bar_atr(self.genie_object)
        setattr(self.genie_object, "bar_atr", bar_atr)
        return self

    # @ray.remote
    def simulate_signals(self, parameters, ray_sim_n_cpus="auto"):
        """

        Args:
            ray_sim_n_cpus:
            parameters:

        Returns:
            object:

        """
        logger.info(f"Simulating Signals 🤙")
        gc.collect()
        data = [self.genie_object.optimization_open_data, self.genie_object.optimization_low_data,
                self.genie_object.optimization_high_data, self.genie_object.optimization_close_data]
        open_data, low_data, high_data, close_data = get_objects_list_from_ray(data)
        # open_data, low_data, \
        # high_data, close_data = self.genie_object.optimization_open_data, self.genie_object.optimization_low_data, \
        #                         self.genie_object.optimization_high_data, self.genie_object.optimization_close_data

        Start_Timer = perf_counter()
        long_entries, long_exits, \
        short_entries, short_exits, \
        strategy_specific_kwargs = self.genie_object.runtime_settings["Strategy_Settings.Strategy"](open_data, low_data,
                                                                                                    high_data,
                                                                                                    close_data,
                                                                                                    parameters,
                                                                                                    ray_sim_n_cpus)

        gc.collect()
        logger.info(f'Time to Prepare Entries and Exits Signals {perf_counter() - Start_Timer}')
        return long_entries, long_exits, short_entries, short_exits, strategy_specific_kwargs

    def simulate_events(self, long_entries: object, long_exits: object, short_entries: object, short_exits: object,
                        strategy_specific_kwargs: object) -> object:
        '''Run Portfolio Simulation

        Args:
            long_entries: 
            long_exits: 
            short_entries: 
            short_exits: 
            strategy_specific_kwargs: 

        Returns:
            object: 
        '''  # (2b)_n-1
        logger.info(f"Simulating Events 💥")
        gc.collect()
        data = [self.genie_object.optimization_open_data, self.genie_object.optimization_low_data,
                self.genie_object.optimization_high_data, self.genie_object.optimization_close_data,
                self.genie_object.optimization_spread_data
                ]
        open_data, low_data, high_data, close_data, spread_data = get_objects_list_from_ray(data)
        # open_data, low_data, \
        # high_data, close_data = self.genie_object.optimization_open_data, self.genie_object.optimization_low_data, \
        #                         self.genie_object.optimization_high_data, self.genie_object.optimization_close_data

        batch_size_ = int(long_entries.shape[1] / len(close_data.keys()))

        pf = self.genie_object.runtime_settings[
            "Portfolio_Settings.Simulator.optimization"](
            self.genie_object.runtime_settings,
            open_data, low_data, high_data, close_data,
            spread_data,
            long_entries, long_exits, short_entries,
            short_exits, strategy_specific_kwargs,
            batch_size_
        )

        # '''Save Portfolio after each epoch'''  # (3)_n-1
        # pf.save(
        #     f'{self.genie_object.portfolio_dir_path}/{self.genie_object.runtime_settings["Portfolio_Settings.saved_pf_optimization"]}')

        gc.collect()
        return pf

        ...
