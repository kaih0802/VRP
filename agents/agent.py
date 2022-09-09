from abc import abstractmethod
import numpy as np

class Agent():
    """Base class for agent"""

    @abstractmethod
    def __init__(self, env):
        """
        Setting contains a dictionary with the parameters
        """
    
    @abstractmethod
    def compute_delivery_to_crowdship(self, deliveries):
        pass

    @abstractmethod
    def compute_VRP(self, delivery_to_do, vehicles):
        pass

    @abstractmethod
    def learn_and_save(self):
        pass
    
    @abstractmethod
    def start_test(self):
        pass
