from typing import *
import os
from tqdm import tqdm
import pandas as pd
import cirpy
import torch
from torch_geometric.utils import from_smiles
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import MoleculeNet

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
        
    def __load_data(self) -> None:
        if self.root is not None and self.name is not None:
            if self.label_colname is not None and self.filepath is not None and os.path.exists(self.filepath):
                try:
                    df = pd.read_csv(self.filepath)
                    label_cols = [col for col in df.columns if col != self.smiles_colname]
                    if len(label_cols) > 1 and self.label_colname in label_cols:
                        print(f"Multi-label dataset detected. Loading from CSV to extract specific column: {self.label_colname}")
                        data_list, has_labelled_data = self.__load_data_from_filepath()
                        self.data_list = data_list
                        return
                except:
                    pass
            
            try:
                print("Trying to load dataset from MoleculeNet")
                self.data_list = list(MoleculeNet(self.root, self.name))
                print("Dataset loaded from MoleculeNet")
                
                if self.label_colname is not None and len(self.data_list) > 0:
                    first_data = self.data_list[0]
                    if hasattr(first_data, 'y') and first_data.y is not None:
                        if isinstance(first_data.y, torch.Tensor):
                            y_shape = first_data.y.shape
                            if len(y_shape) > 0 and (y_shape[0] > 1 if len(y_shape) == 1 else y_shape[-1] > 1):
                                if self.filepath is not None and os.path.exists(self.filepath):
                                    try:
                                        df = pd.read_csv(self.filepath)
                                        label_cols = [col for col in df.columns if col != self.smiles_colname]
                                        if self.label_colname in label_cols:
                                            label_idx = label_cols.index(self.label_colname)
                                            for data in self.data_list:
                                                if hasattr(data, 'y') and data.y is not None:
                                                    if isinstance(data.y, torch.Tensor):
                                                        if len(data.y.shape) == 1 and data.y.numel() > 1:
                                                            if label_idx < data.y.numel():
                                                                data.y = data.y[label_idx].clone().detach()
                                                            else:
                                                                print(f"Warning: Label index {label_idx} out of range for data.y with {data.y.numel()} elements")
                                                        elif data.y.numel() == 1:
                                                            pass
                                    except Exception as e:
                                        print(f"Warning: Could not extract specific label column from CSV: {e}")
            except:
                print("Dataset not found in MoleculeNet")
                dirpath = os.path.join(self.root, self.name, "processed")
                os.makedirs(dirpath, exist_ok=True)
                data_filepath = os.path.join(dirpath, "data.pt")
                if os.path.exists(data_filepath):
                    print("Loading dataset from local cache")
                    self.data_list = torch.load(data_filepath, weights_only=False)
                elif self.filepath is not None:
                    print("Loading data from filepath")
                    data_list, has_labelled_data = self.__load_data_from_filepath()
                    self.data_list = data_list
                    if has_labelled_data:
                        print("Saving data to local cache")
                        torch.save(self.data_list, data_filepath)
                else:
                    raise ValueError(
                        "At least one of root and name, or filepath must be provided")
        elif self.filepath is not None:
            print("Loading dataset from filepath")
            data_list, has_labelled_data = self.__load_data_from_filepath()
            self.data_list = data_list
        elif self.smiles is not None:
            self.data_list = self.__load_data_from_smiles(self.smiles)
        elif self.compound_names is not None:
            if isinstance(self.compound_names, str):
                data = self.__load_from_compound_name(self.compound_names)
                if data is None:
                    self.data_list = []
                else:
                    self.data_list = [data]
            elif isinstance(self.compound_names, list):
                self.data_list = self.__load_from_compound_names(self.compound_names)
            else:
                raise ValueError("Invalid compound names type. Expected a string or list of strings.")
        else:
            raise ValueError(
                "At least one of root and name, or filepath must be provided")
            
    def __load_data_from_filepath(self) -> Tuple[List[Data], bool]:
        if self.is_smiles_file:
            print("Loading dataset from smiles file")
            data_list, has_labelled_data = self.__load_data_from_smiles_file(
                self.filepath)
        elif self.is_compound_name_file:
            print("Loading dataset from compound name file")
            data_list, has_labelled_data = self.__load_from_compound_name_file(
                self.filepath)
        else:
            raise ValueError("Invalid file extension. Expected .csv or .txt.")
        return data_list, has_labelled_data
    
    def __load_data_from_smiles_file(self, filepath: str) -> Tuple[List[Data], bool]:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            smiles_list = df[self.smiles_colname].tolist()
            # label_list = df[self.label_colname].tolist()
            if isinstance(self.label_colname, str):
                label_list = df[self.label_colname].tolist()
                has_labelled_data = True
            elif isinstance(self.label_colname, list):
                label_list = df[self.label_colname].values.tolist()
                has_labelled_data = True
            elif self.label_colname is None:
                has_labelled_data = False
            else:
                raise ValueError("Invalid label column name. Expected a string or list of strings or None.")
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as file:
                smiles_list = file.readlines()
                label_list = [None] * len(smiles_list)
                has_labelled_data = False
        else:
            raise ValueError("Invalid file extension. Expected .csv or .txt.")
        data_list = self.__load_data_from_smiles(smiles_list, label_list)
        return data_list, has_labelled_data
    
    def __load_from_compound_name_file(self, filepath: str) -> Tuple[List[Data], bool]:
        has_labelled_data = False
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            compound_names = df[self.compound_name_colname].tolist()
            smiles_list = self.__get_smiles_from_compound_names(compound_names)
            if self.label_colname is not None:
                label_list = df[self.label_colname].tolist()
                label_list = [label for i, label in enumerate(label_list) if smiles_list[i] is not None]
                smiles_list = [smiles for smiles in smiles_list if smiles is not None]
                data_list = self.__load_data_from_smiles(smiles_list, label_list)
                has_labelled_data = True
            else:
                smiles_list = [smiles for smiles in smiles_list if smiles is not None]
                data_list = self.__load_data_from_smiles(smiles_list)
            return data_list, has_labelled_data
        elif filepath.endswith('.txt'):
            with open(filepath, 'r') as file:
                compound_names = file.readlines()
                data_list = self.__load_from_compound_names(compound_names)
                return data_list, has_labelled_data
        else:
            raise ValueError("Invalid file extension. Expected .csv or .txt.")
        
    def __load_from_compound_names(self, compound_names: List[str], labels: Union[int, float, List[int], List[float]]=None) -> List[Union[Data, None]]:
        smiles_list = [self.__get_smiles_from_compound_name(name) for name in compound_names]
        if labels is None:
            smiles_list = [smiles for smiles in smiles_list if smiles is not None]
            labels = [None] * len(smiles_list)
        else:
            labels = [label for i, label in enumerate(labels) if smiles_list[i] is not None]
            smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        data_list = self.__load_data_from_smiles(smiles_list, labels)
        return data_list
        
    def __load_from_compound_name(self, compound_name: str, label: Union[int, float]=None) -> Union[Data, None]:
        smiles = self.__get_smiles_from_compound_name(compound_name)
        if smiles is None:
            return None
        return self.__load_data_from_smiles(smiles, label)
        
    def __get_smiles_from_compound_names(self, compound_names: List[str]) -> List[Union[str, None]]:
        return [self.__get_smiles_from_compound_name(name) for name in tqdm(compound_names, desc="Resolving smiles to compound names")]
    
    def __get_smiles_from_compound_name(self, compound_name: str) -> Union[str, None]:
        smiles = cirpy.resolve(input=compound_name, representation="smiles")
        return smiles
        
    def __load_data_from_smiles(self, smiles: Union[str, List[str]], label: Union[int, float, List[int], List[float], None] = None) -> List[Data]:
        if isinstance(smiles, str):
            data_list = [self.__load_data_from_smiles_str(smiles, label)]
        elif isinstance(smiles, list):
            if label is None:
                label = [None] * len(smiles)
            
            data_list = [self.__load_data_from_smiles_str(s, l) for s, l in tqdm(zip(smiles, label), desc="Loading smiles")]
        else:
            raise ValueError(
                "Invalid input type. Expected a string or list of strings.")
        data_list = [data for data in data_list if data is not None]
        return data_list
    
    def __load_data_from_smiles_str(self, smiles: str, label: Union[int, float, None] = None) -> Union[Data, None]:
        """Convert SMILES to an RDKit molecule object"""
        try:
            data = from_smiles(smiles)
            if label is not None:
                data.y = label
        except:
            return None
        return data

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