from src.data_loader import MoleculeDataset

class Preprocessor:
    """Preprocessing steps in the class:
    - Removing outliers based on the labels if the task is regression.
    - Oversampling/undersampling based on a threshold value if the task is classification."""
    def __init__(self):
        pass
    
    def remove_outliers(self):
        pass
    
    def get_balance_data(self):
        pass
    
    def preprocess(self): # main item
        pass
    
    def __call__(self):
        return self.preprocess()