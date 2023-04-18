from constants import SEED, MODEL_DIR, ROOT, EARLY_STOPPING_PATIENCE, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, WEIGHT_DECAY, BATCH_SIZE, LEARNING_RATE, EPOCHS, HIDDEN_DIM, DROPOUT_RATE

DATASET_CONFIG = {
    'ESOL': {
        'filepath': 'datasets/csv_files/delaney-processed.csv',
        'smiles_colname': 'smiles',
        'task_type': 'regression',
        'targets': {
            'ESOL predicted log solubility in mols per litre': {
                'task_type': 'regression',
                'default_layer_type': 'gcn',
            }
        }
    },
    'FreeSolv': {
        'filepath': 'datasets/csv_files/SAMPL.csv',
        'smiles_colname': 'smiles',
        'task_type': 'regression',
        'targets': {
            'expt': {
                'task_type': 'regression',
                'default_layer_type': 'gcn',
            }
        }
    },
    'Lipophilicity': {
        'filepath': 'datasets/csv_files/Lipophilicity.csv',
        'smiles_colname': 'smiles',
        'task_type': 'regression',
        'targets': {
            'exp': {
                'task_type': 'regression',
                'default_layer_type': 'gcn',
            }
        }
    },
    'HIV': {
        'filepath': 'datasets/csv_files/HIV.csv',
        'smiles_colname': 'smiles',
        'task_type': 'classification',
        'targets': {
            'HIV_active': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            }
        }
    },
    'BBBP': {
        'filepath': 'datasets/csv_files/BBBP.csv',
        'smiles_colname': 'smiles',
        'task_type': 'classification',
        'targets': {
            'p_np': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            }
        }
    },
    'BACE': {
        'filepath': 'datasets/csv_files/bace.csv',
        'smiles_colname': 'mol',
        'task_type': 'classification',
        'targets': {
            'Class': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            }
        }
    },
    'SIDER': {
        'filepath': 'datasets/csv_files/sider.csv',
        'smiles_colname': 'smiles',
        'task_type': 'multi-label classification',
        'targets': {
            'Hepatobiliary disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Metabolism and nutrition disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Product issues': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Eye disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Investigations': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Musculoskeletal and connective tissue disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Gastrointestinal disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Social circumstances': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Immune system disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Reproductive system and breast disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'General disorders and administration site conditions': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Endocrine disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Surgical and medical procedures': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Vascular disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Blood and lymphatic system disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Skin and subcutaneous tissue disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Congenital, familial and genetic disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Infections and infestations': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Respiratory, thoracic and mediastinal disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Psychiatric disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Renal and urinary disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Pregnancy, puerperium and perinatal conditions': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Ear and labyrinth disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Cardiac disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Nervous system disorders': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            },
            'Injury, poisoning and procedural complications': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            }
        }
    },
    'ClinTox': {
        'filepath': 'datasets/csv_files/clintox.csv',
        'smiles_colname': 'smiles',
        'task_type': 'multi-label classification',
        'targets': {
            'FDA_APPROVED': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
                'seed': SEED,
                'model_dirpath': MODEL_DIR,
                'processed_data_dirpath': ROOT,
                'training_params': {
                    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                    'lr_scheduler_factor': LR_SCHEDULER_FACTOR,
                    'lr_scheduler_patience': LR_SCHEDULER_PATIENCE,
                    'weight_decay': WEIGHT_DECAY,
                },
                'model_config': {
                    'batch_size': BATCH_SIZE,
                    'lr': LEARNING_RATE,
                    'epochs': EPOCHS,
                    'hidden_dim': HIDDEN_DIM,
                    'dropout_rate': DROPOUT_RATE,
                }
            },
            'CT_TOX': {
                'task_type': 'classification',
                'default_layer_type': 'gcn',
            }
        }
    }
}

