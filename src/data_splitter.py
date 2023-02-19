# Let's develop the data spliting based on strategy design
from abc import ABC, abstractmethod

# Base Strategy Class
class DataSplitStrategy(ABC):
    def standardize_ratios(self):
        pass
    
    @abstractmethod
    def split(self):
        pass