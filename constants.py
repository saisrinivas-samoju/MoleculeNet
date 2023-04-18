from typing import Literal

# Global training config (never duplicated)
SEED = 42
MODEL_DIR = 'models'
ROOT = 'datasets/processed'
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 0

# Global model config (can be overridden per target if needed)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_DIM = 64
DROPOUT_RATE = 0.2

# Type hints for type safety
DatasetName = Literal[
    'ESOL',
    'FreeSolv',
    'Lipophilicity',
    'HIV',
    'BBBP',
    'BACE',
    'SIDER',
    'ClinTox'
]

# Target column names mapped to their datasets
# This provides type safety and shows all available targets per dataset
DatasetTargets = {
    'ESOL': Literal['ESOL predicted log solubility in mols per litre'],
    'FreeSolv': Literal['expt'],
    'Lipophilicity': Literal['exp'],
    'HIV': Literal['HIV_active'],
    'BBBP': Literal['p_np'],
    'BACE': Literal['Class'],
    'SIDER': Literal[
        'Hepatobiliary disorders',
        'Metabolism and nutrition disorders',
        'Product issues',
        'Eye disorders',
        'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders',
        'Social circumstances',
        'Immune system disorders',
        'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders',
        'Surgical and medical procedures',
        'Vascular disorders',
        'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders',
        'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders',
        'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders',
        'Cardiac disorders',
        'Nervous system disorders',
        'Injury, poisoning and procedural complications'
    ],
    'ClinTox': Literal['FDA_APPROVED', 'CT_TOX']
}

# Layer type
LayerType = Literal['gcn', 'gat']

# Task type
TaskType = Literal['regression', 'classification', 'multi-label classification']

