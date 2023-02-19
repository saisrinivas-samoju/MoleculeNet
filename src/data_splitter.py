# Let's develop the data spliting based on strategy design
from abc import ABC, abstractmethod
import random
import torch
from torch_geometric.loader import DataLoader

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
    
class SimpleSplit(DataSplitStrategy):
    def split(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32):
        train_ratio, val_ratio, test_ratio = self.standardize_ratios(train_ratio, val_ratio, test_ratio)
        train_dataset = dataset[:int(train_ratio * len(dataset))]
        val_dataset = dataset[int(train_ratio * len(dataset)):int((train_ratio + val_ratio) * len(dataset))]
        test_dataset = dataset[int((train_ratio + val_ratio) * len(dataset)):]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
class ShuffleSplit(DataSplitStrategy):
    def split(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, random_seed=42):
        train_ratio, val_ratio, test_ratio = self.standardize_ratios(train_ratio, val_ratio, test_ratio)
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        
        # Create a list of indices and shuffle them
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create dataset subsets
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
        
        return train_loader, val_loader, test_loader
    
# Context class that uses a strategy
class DataSplitter:
    def __init__(self, strategy=None):
        self.strategy = strategy or SimpleSplit()
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def split_data(self, dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, random_seed=42):
        if isinstance(self.strategy, ShuffleSplit):
            return self.strategy.split(dataset, train_ratio, val_ratio, test_ratio, batch_size, random_seed)
        return self.strategy.split(dataset, train_ratio, val_ratio, test_ratio, batch_size)