from typing import *
from torch_geometric.data import Dataset, Data

class MoleculeDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self):
        pass
    
    @property
    def num_nodes(self) -> int:
        pass
    
    @property
    def feature_names(self):
        pass

    @property
    def has_labels(self):
        pass

    @property
    def data_source_type(self):
        pass

    @classmethod
    def from_moleculenet(cls):
        pass

    @classmethod
    def from_smiles_file(cls):
        pass

    @classmethod
    def from_compound_name_file(cls):
        pass

    @classmethod
    def from_smiles(cls):
        pass

    @classmethod
    def from_compound_names(cls):
        pass