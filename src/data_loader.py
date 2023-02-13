from typing import *
from torch_geometric.data import Dataset, Data

class MoleculeDataset(Dataset):
    def __init__(self, root: str = None, name: Literal["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBBP", "Tox21", "ToxCast", "SIDER", "ClinTox"] = None, filepath: str = None, smiles_colname: str = 'smiles', label_colname: str = 'label', compound_name_colname: str = 'compound_name', filetype: Literal['smiles', 'compound_name'] = None, smiles: Union[str, List[str]] = None, compound_names: Union[str, List[str]] = None):
        """
        Constructor for the data loader
        """
        self.root = root
        self.name = name
        self.filepath = filepath
        self.smiles_colname = smiles_colname
        self.label_colname = label_colname
        self.compound_name_colname = compound_name_colname
        self.smiles = smiles
        self.compound_names = compound_names
        if filetype == 'compound_name':
            self.is_compound_name_file = True
            self.is_smiles_file = False
        else:
            self.is_compound_name_file = False
            self.is_smiles_file = True

        self.__load_data()
    
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes/molecules in the dataset."""
        return len(self.data_list)
    
    @property
    def feature_names(self) -> List[str]:
        """Names of the node features."""
        return ["atomic_num", "chiral_tag", "total_degree", "formal_charge", "total_num_h", "radical_electrons", "hybridization", 
                "is_aromatic", "in_ring"]

    @property
    def has_labels(self) -> bool:
        """Check if the dataset has labels."""
        return all(hasattr(data, 'y') and data.y is not None for data in self.data_list)

    @property
    def data_source_type(self) -> str:
        """The type of data source used for this dataset."""
        if self.root is not None and self.name is not None:
            return "moleculenet"
        elif self.filepath is not None:
            if self.is_smiles_file:
                return "smiles_file"
            else:
                return "compound_name_file"
        elif self.smiles is not None:
            return "smiles"
        elif self.compound_names is not None:
            return "compound_names"
        else:
            return "unknown"

    @classmethod
    def from_moleculenet(cls, root: str, name: str) -> 'MoleculeDataset':
        """Factory method to create MoleculeDataset from MoleculeNet dataset."""
        return cls(root=root, name=name)

    @classmethod
    def from_smiles_file(cls, filepath: str, smiles_colname: str = 'smiles', label_colname: str = 'label') -> 'MoleculeDataset':
        """Factory method to create MoleculeDataset from a file containing SMILES strings."""
        return cls(filepath=filepath, smiles_colname=smiles_colname, label_colname=label_colname, filetype='smiles')

    @classmethod
    def from_compound_name_file(cls, filepath: str, compound_name_colname: str = 'compound_name', label_colname: str = 'label') -> 'MoleculeDataset':
        """Factory method to create MoleculeDataset from a file containing compound names."""
        return cls(filepath=filepath, compound_name_colname=compound_name_colname, label_colname=label_colname, filetype='compound_name')

    @classmethod
    def from_smiles(cls, smiles: Union[str, List[str]]) -> 'MoleculeDataset':
        """Factory method to create MoleculeDataset directly from SMILES string(s)."""
        return cls(smiles=smiles)

    @classmethod
    def from_compound_names(cls, compound_names: Union[str, List[str]]) -> 'MoleculeDataset':
        """Factory method to create MoleculeDataset directly from compound name(s)."""
        return cls(compound_names=compound_names)