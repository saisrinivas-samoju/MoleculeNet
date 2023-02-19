from src.data_loader import MoleculeDataset

def get_data_info(dataset: MoleculeDataset):
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of Node features: {dataset.num_features}")
    print(f"Number of Edge features: {dataset.num_edge_features}")
    print(f"Feature names: {dataset.feature_names}")