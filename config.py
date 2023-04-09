from typing import *

root = 'datasets/processed'
name: Literal['ESOL', "FreeSolv", "Lipo", "HIV"] = "ESOL"
name = f"{name}-made"

filepath = 'datasets/csv_files/delaney-processed.csv'
# filepath = f'datasets/csv_files/SAMPL.csv'
# filepath = f"datasets/csv_files/Lipophilicity.csv"
# filepath = f"datasets/csv_files/HIV.csv"

smiles_colname = 'smiles'

label_colname = 'ESOL predicted log solubility in mols per litre'
# label_colname = 'expt'
# label_colname = 'exp'
# label_colname = 'HIV_active'

task_type: Literal['regression', 'classification'] = 'regression'

# Model Config
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
HIDDEN_DIM = 64
# LAYER_TYPE = 'gcn'
LAYER_TYPE = 'gat'
DROPOUT_RATE = 0.2

# Training config
SEED = 42
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 0
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5

# Model Saving Config
MODEL_DIR = 'models' ### Directory for saving models
MODEL_NAME = f"{name}_{LAYER_TYPE}" ### Model name with dataset and architecture info