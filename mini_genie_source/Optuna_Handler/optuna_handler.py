#!/usr/bin/env python3
class Optuna_Handler():
    """
    Multi-Objective Optimization:
        Uses reduced parameter space.
        Starts with prior trials loaded from metrics_record file
        Does not compute them in vbt batches but can be parallelized

    Args:
         (): 
         (): 

    Attributes:
         (): 
         (): 

    Note:
        
    """

    def __init__(self, optional_object):
        """Constructor for Optuna_Handler"""
        from Utilities.general_utilities import print_dict
        self.print_dict = print_dict
