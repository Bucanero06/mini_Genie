from time import perf_counter

from logger_tt import logger

from Utilities.general_utilities import get_objects_list_from_ray


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

    def print_dict(self, optional_object: object = None) -> object:
        """

        Args:
            optional_object: 

        Returns:
            object: 
        """
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    @property
    def compute_bar_atr(self) -> object:
        """

        Returns:
            object: 

        """
        from Simulation_Handler.compute_bar_atr import compute_bar_atr
        bar_atr = compute_bar_atr(self.genie_object)
        setattr(self.genie_object, "bar_atr", bar_atr)
        return self

    def simulate_signals(self, parameters):
        data = [self.genie_object.optimization_open_data, self.genie_object.optimization_low_data,
                self.genie_object.optimization_high_data, self.genie_object.optimization_close_data]
        open_data, low_data, high_data, close_data = get_objects_list_from_ray(data)

        Start_Timer = perf_counter()
        long_entries, long_exits, \
        short_entries, short_exits, \
        strategy_specific_kwargs = self.genie_object.runtime_settings["Strategy_Settings.Strategy"](open_data, low_data,
                                                                                                    high_data,
                                                                                                    close_data,
                                                                                                    parameters)

        logger.info(f'Time to Prepare Entries and Exits Signals {perf_counter() - Start_Timer}')
        # print(f'Time to Prepare Entries and Exits Signals {perf_counter() - Start_Timer}')

        return long_entries, long_exits, short_entries, short_exits, strategy_specific_kwargs

    def simulate_events(self, long_entries, long_exits, short_entries, short_exits, strategy_specific_kwargs):




















        ...
