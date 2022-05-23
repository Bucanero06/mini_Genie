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

    def print_dict(self, optional_object=None):
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    def simulate_atr(self):
        ...