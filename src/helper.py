import re
import os
import json
import torch
from typing import Dict, Any, Literal, Optional
from dataset_config import DATASET_CONFIG
from constants import SEED, MODEL_DIR, ROOT, EARLY_STOPPING_PATIENCE, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, WEIGHT_DECAY, BATCH_SIZE, LEARNING_RATE, EPOCHS, HIDDEN_DIM, DROPOUT_RATE


def sanitize_model_name(dataset_name: str, target_column: str, layer_type: str) -> str:

    dataset_name = dataset_name.lower()
    target_column = target_column.lower()
    layer_type = layer_type.lower()

    dataset_name = dataset_name.replace(' ', '_')
    target_column = target_column.replace(' ', '_')

    dataset_name = re.sub(r'[^a-z0-9_]', '', dataset_name)
    target_column = re.sub(r'[^a-z0-9_]', '', target_column)
    layer_type = re.sub(r'[^a-z0-9_]', '', layer_type)

    model_name = f"{dataset_name}_{target_column}_{layer_type}"

    model_name = re.sub(r'_+', '_', model_name)

    model_name = model_name.strip('_')

    return model_name


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_CONFIG:
        available_datasets = ', '.join(DATASET_CONFIG.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")

    return DATASET_CONFIG[dataset_name]


def get_full_config(dataset_name: str, target_column: str, layer_type: Optional[str] = None) -> Dict[str, Any]:

    dataset_config = get_dataset_config(dataset_name)

    if target_column not in dataset_config['targets']:
        available_targets = ', '.join(dataset_config['targets'].keys())
        raise ValueError(
            f"Target '{target_column}' not found for dataset '{dataset_name}'. Available targets: {available_targets}")

    target_config = dataset_config['targets'][target_column]

    final_layer_type = layer_type or target_config.get(
        'default_layer_type', 'gcn')

    model_name = sanitize_model_name(
        dataset_name, target_column, final_layer_type)

    full_config = {

        'dataset_name': dataset_name,
        'target_column': target_column,
        'filepath': dataset_config['filepath'],
        'smiles_colname': dataset_config['smiles_colname'],
        'task_type': target_config['task_type'],


        'layer_type': final_layer_type,
        'model_name': model_name,


        'seed': target_config.get('seed', SEED),
        'model_dirpath': target_config.get('model_dirpath', MODEL_DIR),
        'processed_data_dirpath': target_config.get('processed_data_dirpath', ROOT),


        'training_params': {
            'early_stopping_patience': target_config.get('training_params', {}).get('early_stopping_patience', EARLY_STOPPING_PATIENCE),
            'lr_scheduler_factor': target_config.get('training_params', {}).get('lr_scheduler_factor', LR_SCHEDULER_FACTOR),
            'lr_scheduler_patience': target_config.get('training_params', {}).get('lr_scheduler_patience', LR_SCHEDULER_PATIENCE),
            'weight_decay': target_config.get('training_params', {}).get('weight_decay', WEIGHT_DECAY),
        },


        'model_config': {
            'batch_size': target_config.get('model_config', {}).get('batch_size', BATCH_SIZE),
            'lr': target_config.get('model_config', {}).get('lr', LEARNING_RATE),
            'epochs': target_config.get('model_config', {}).get('epochs', EPOCHS),
            'hidden_dim': target_config.get('model_config', {}).get('hidden_dim', HIDDEN_DIM),
            'dropout_rate': target_config.get('model_config', {}).get('dropout_rate', DROPOUT_RATE),
        }
    }

    return full_config


def update_model_registry(registry_path: str = 'model_registry.json', model_dir: str = MODEL_DIR) -> None:

    if not os.path.exists(registry_path):
        print(
            f"Warning: Registry file {registry_path} not found. Nothing to update.")
        return

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    if 'models' not in registry:
        print("Warning: Registry file does not have 'models' key. Nothing to update.")
        return

    model_id_to_entry = {}
    for entry in registry['models']:
        model_id_to_entry[entry['id']] = entry

    id_mapping = {
        'esol_solubility': ('ESOL', 'ESOL predicted log solubility in mols per litre'),
        'freesolvation': ('FreeSolv', 'expt'),
        'lipophilicity': ('Lipophilicity', 'exp'),
    }

    def get_model_id(dataset_name, target_column):

        for custom_id, (ds, tgt) in id_mapping.items():
            if ds == dataset_name and tgt == target_column:
                return custom_id

        return sanitize_model_name(dataset_name, target_column, 'gcn').replace('_gcn', '')

    for dataset_name, dataset_config in DATASET_CONFIG.items():
        for target_column, target_config in dataset_config['targets'].items():
            task_type = target_config['task_type']

            model_id = get_model_id(dataset_name, target_column)

            entry = model_id_to_entry.get(model_id)
            if entry is None:

                continue

            existing_models = []
            for layer_type in ['gcn', 'gat']:
                model_name = sanitize_model_name(
                    dataset_name, target_column, layer_type)
                model_file = os.path.join(model_dir, f'{model_name}_full.pt')
                if os.path.exists(model_file):
                    existing_models.append((layer_type, model_file))

            if not existing_models:
                entry['model_file'] = None
                entry['enabled'] = False
                continue

            if len(existing_models) == 1:
                layer_type, model_file = existing_models[0]
                entry['model_file'] = model_file
                entry['enabled'] = True
            else:

                best_layer_type = None
                best_model_file = None
                best_metric = None

                for layer_type, model_file in existing_models:
                    try:

                        checkpoint = torch.load(model_file, map_location='cpu')
                        metrics = checkpoint.get('metrics', {})

                        if task_type == 'regression':

                            metric_value = metrics.get(
                                'test_RMSE', metrics.get('RMSE', float('inf')))
                            is_better = best_metric is None or metric_value < best_metric
                        else:

                            metric_value = metrics.get(
                                'test_F1_macro', metrics.get('F1_macro', -float('inf')))
                            is_better = best_metric is None or metric_value > best_metric

                        if is_better:
                            best_metric = metric_value
                            best_layer_type = layer_type
                            best_model_file = model_file
                        elif metric_value == best_metric:

                            if layer_type == 'gat' and best_layer_type == 'gcn':
                                best_layer_type = layer_type
                                best_model_file = model_file
                    except Exception as e:
                        print(
                            f"Warning: Could not load metrics from {model_file}: {e}")
                        continue

                if best_model_file is not None:
                    entry['model_file'] = best_model_file
                    entry['enabled'] = True
                else:
                    entry['model_file'] = None
                    entry['enabled'] = False

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=4)

    print(f"Model registry updated: {registry_path}")
