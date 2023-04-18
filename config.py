from constants import DatasetName, LayerType

# User Configuration - Set dataset and target here
DATASET_NAME: DatasetName = 'ESOL'
TARGET_COLUMN = 'ESOL predicted log solubility in mols per litre'
LAYER_TYPE: LayerType = 'gcn'  # Optional: if None, uses default from dataset_config
