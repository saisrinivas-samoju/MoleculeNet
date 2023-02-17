from typing import *
import numpy as np
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
        label_list = [data.y for data in self.dataset]
        mean = np.mean(label_list)
        std = np.std(label_list)
        lower_bound = mean - self.outlier_threshold * std
        upper_bound = mean + self.outlier_threshold * std
        filtered_dataset = [data for data in self.dataset if data.y > lower_bound and data.y < upper_bound]
        return filtered_dataset
    
    def get_balance_data(self):
        label_list = [data.y for data in self.dataset]
        unique, counts = np.unique(label_list, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        label_prop_dict = {label: counts_dict[label] / len(label_list) for label in unique}
        if self.balance_strategy == 'oversampling':
            idx_values_to_add = []
            for label in unique:
                if label_prop_dict[label] < self.balance_threshold:
                    diff_prop = self.balance_threshold - label_prop_dict[label]
                    num_samples_to_add = int(diff_prop * len(label_list))
                    class_data_idx_list = [idx for idx, data in enumerate(self.dataset) if data.y == label]
                    if num_samples_to_add > len(class_data_idx_list):
                        replace = True
                    else:
                        replace = False
                    random_idx_values = np.random.choice(range(len(class_data_idx_list)), size=num_samples_to_add, replace=replace)
                    idx_values_to_add.extend(random_idx_values)
            res_dataset = [self.dataset[idx] for idx in idx_values_to_add] + self.dataset.data_list
        elif self.balance_strategy == 'undersampling':
            idx_value_to_remove = []
            for label in unique:
                if label_prop_dict[label] > self.balance_threshold:
                    diff_prop = label_prop_dict[label] - self.balance_threshold
                    num_samples_to_remove = int(diff_prop * len(label_list))
                    class_data_idx_list = [idx for idx, data in enumerate(self.dataset) if data.y == label]
                    random_idx_values = np.random.choice(range(len(class_data_idx_list)), size=num_samples_to_remove, replace=False)
                    idx_value_to_remove.extend(random_idx_values)
            res_dataset = [self.dataset[idx] for idx in idx_value_to_remove]
        else:
            raise ValueError(f"Invalid balance strategy: {self.balance_strategy} | balance_strategy must be either 'oversampling' or 'undersampling'")
        return res_dataset
    
    def preprocess(self): # main item
        if self.task == "regression":
            if self.remove_outliers:
                processed_dataset = self.remove_outliers()
            else:
                return self.dataset
        elif self.task == "classification":
            if self.balance_data:
                processed_dataset = self.get_balance_data()
            else:
                return self.dataset
                
        elif self.task == "multi-label classification": # TODO: Add multi-label classification preprocessing
            return self.dataset
        else:
            raise ValueError(f"Invalid task: {self.task} | task must be either 'regression' or 'classification'")
        
        self.dataset.data_list = processed_dataset
        return self.dataset
    
    def __call__(self):
        return self.preprocess()