from typing import *
from src.data_loader import MoleculeDataset

class Preprocessor:
    """Preprocessing steps in the class:
    - Removing outliers based on the labels if the task is regression.
    - Oversampling/undersampling based on a threshold value if the task is classification."""
    def __init__(self, dataset: MoleculeDataset, task: Literal["regression", "classification"], balance_data: bool = False, balance_threshold: float = 0.2, balance_strategy: Literal["oversampling", "undersampling"] = "oversampling", remove_outliers: bool = False, outlier_threshold: float = 3.5):
        self.dataset = dataset
        self.task = task
        if self.task == "classification":
            self.balance_data = balance_data
            self.num_classes = self.dataset.num_classes
            balance_threshold = balance_threshold if balance_threshold < 0.5 else (1 - balance_threshold)
            # considering balanced threshold value is based on binary class division though there are multiple classes
            balance_threshold = balance_threshold * 2 / self.num_classes
            self.balance_threshold = balance_threshold
            self.balance_strategy = balance_strategy
        elif self.task=="regression":
            self.remove_outliers = remove_outliers
            self.outlier_threshold = outlier_threshold
    
    def remove_outliers(self):
        pass
    
    def get_balance_data(self):
        pass
    
    def preprocess(self): # main item
        pass
    
    def __call__(self):
        return self.preprocess()