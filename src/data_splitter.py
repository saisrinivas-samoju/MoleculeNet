# Let's develop the data spliting based on strategy design
from abc import ABC, abstractmethod

# Base Strategy Class
class DataSplitStrategy(ABC):
    def standardize_ratios(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        train_ratio = abs(train_ratio)
        val_ratio = abs(val_ratio)
        test_ratio = abs(test_ratio)
        if train_ratio + val_ratio + test_ratio != 1:
            total_ratio = train_ratio + val_ratio + test_ratio
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        return train_ratio, val_ratio, test_ratio
    
    @abstractmethod
    def split(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32):
        """Split a dataset into train, validation and test sets"""
        pass