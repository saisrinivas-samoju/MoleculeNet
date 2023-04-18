from config import DATASET_NAME, TARGET_COLUMN, LAYER_TYPE
from src.helper import get_full_config
from src.data_loader import MoleculeDataset
from src.data_preprocessor import Preprocessor
from src.data_splitter import DataSplitter, ShuffleSplit
from src.model_architecture import MoleculeNetRegressor, MoleculeNetClassifier
from src.train import setup_training, plot_training_history
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from src.evaluate import *
from src.predict import predict_molecule, predict_molecules
from src.model_utils import save_model, load_model
from constants import DatasetName, LayerType, DatasetTargets
import matplotlib.pyplot as plt
import os
import pandas as pd


def run_pipeline(dataset_name: DatasetName, target_column: DatasetTargets, layer_type: str = LayerType):

    config = get_full_config(dataset_name, target_column, layer_type)

    filepath = config['filepath']
    smiles_colname = config['smiles_colname']
    label_colname = config['target_column']
    task_type = config['task_type']
    root = config['processed_data_dirpath']
    model_name = config['model_name']
    model_dirpath = config['model_dirpath']
    seed = config['seed']
    layer_type = config['layer_type']

    training_params = config['training_params']
    early_stopping_patience = training_params['early_stopping_patience']
    lr_scheduler_factor = training_params['lr_scheduler_factor']
    lr_scheduler_patience = training_params['lr_scheduler_patience']
    weight_decay = training_params['weight_decay']

    model_config = config['model_config']
    batch_size = model_config['batch_size']
    learning_rate = model_config['lr']
    epochs = model_config['epochs']
    hidden_dim = model_config['hidden_dim']
    dropout_rate = model_config['dropout_rate']

    dataset = MoleculeDataset(root=root, name=dataset_name, filepath=filepath,
                              smiles_colname=smiles_colname, label_colname=label_colname)

    preprocessor = Preprocessor(dataset, task_type)
    dataset = preprocessor.preprocess()

    splitter = DataSplitter()
    splitter.set_strategy(ShuffleSplit())
    train_loader, val_loader, test_loader = splitter.split_data(
        dataset, batch_size=batch_size, random_seed=seed)

    sample_data = dataset[0]
    num_features = sample_data.num_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if task_type == 'regression':
        model = MoleculeNetRegressor(
            num_features=num_features, hidden_dim=hidden_dim, layer_type=layer_type, dropout_rate=dropout_rate)
    elif task_type == 'classification':

        num_classes = dataset.num_classes if hasattr(
            dataset, 'num_classes') else 2
        model = MoleculeNetClassifier(num_features=num_features, hidden_dim=hidden_dim, layer_type=layer_type,
                                      dropout_rate=dropout_rate, num_classes=num_classes)

    model = model.to(device)

    print("Training model...")
    model, optimizer, history, best_metrics = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=epochs,
        patience=early_stopping_patience,
        verbose=True,
        task_type=task_type
    )

    if task_type == 'regression':
        print(
            f"Best validation metrics: RMSE={best_metrics['RMSE']:.4f}, R²={best_metrics['R2']:.4f}")
    if task_type == 'classification':

        print("Best validation metrics:")
        print(f"  Accuracy: {best_metrics['Accuracy']:.4f}")
        print(f"  F1 (macro): {best_metrics['F1_macro']:.4f}")
        if 'F1' in best_metrics:
            print(f"  Precision: {best_metrics['Precision']:.4f}")
            print(f"  Recall: {best_metrics['Recall']:.4f}")
        else:
            print(
                f"  Precision (macro): {best_metrics['Precision_macro']:.4f}")
            print(f"  Recall (macro): {best_metrics['Recall_macro']:.4f}")

    plot_training_history(history, task_type)

    print("\nEvaluating model on test data...")
    test_metrics, test_predictions, test_actual = evaluate_model(
        model, test_loader, device, task_type)

    print("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    if task_type == 'regression':

        plt.figure(figsize=(10, 6))
        plt.scatter(test_actual, test_predictions, alpha=0.5)

        min_val = min(min(test_actual), min(test_predictions))
        max_val = max(max(test_actual), max(test_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Test Set: Predicted vs Actual Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    elif task_type == 'classification':

        print("\nGenerating confusion matrix for test data...")

        class_names = dataset.class_names if hasattr(
            dataset, 'class_names') else None

        plot_confusion_matrix(
            test_actual, test_predictions, class_names=class_names)

        print("\nDetailed evaluation with visualization:")
        evaluate_and_visualize(model, test_loader, device,
                               task_type, class_names=class_names)

    print("\nSaving model...")

    if task_type == 'regression':

        model_info = {
            'num_features': num_features,
            'hidden_dim': hidden_dim,
            'layer_type': layer_type,
            'dropout_rate': dropout_rate,
            'dataset': dataset_name,
            'target_column': target_column,
            'task_type': task_type
        }
    elif task_type == 'classification':
        model_info = {
            'num_features': num_features,
            'hidden_dim': hidden_dim,
            'layer_type': layer_type,
            'dropout_rate': dropout_rate,
            'num_classes': num_classes,
            'dataset': dataset_name,
            'target_column': target_column,
            'task_type': task_type
        }

    combined_metrics = {
        **best_metrics,
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }

    save_model(
        model=model,
        optimizer=optimizer,
        model_info=model_info,
        metrics=combined_metrics,
        model_path=model_dirpath,
        model_name=model_name
    )

    print("\nDemonstrating model loading...")

    model_file = os.path.join(model_dirpath, f'{model_name}_full.pt')

    if os.path.exists(model_file):

        print(f"Loading model from {model_file}")
        loaded_model, loaded_model_info, loaded_metrics = load_model(
            model_file, device, task_type)

        print("\nEvaluating loaded model on test data...")
        loaded_test_metrics, loaded_predictions, loaded_actual = evaluate_model(
            loaded_model, test_loader, device, task_type)

        print("Loaded model test metrics:")
        for metric_name, metric_value in loaded_test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        print("\nVerifying metrics match with original model:")
        for metric_name in test_metrics.keys():
            original = test_metrics[metric_name]
            loaded = loaded_test_metrics[metric_name]
            difference = abs(original - loaded)
            print(
                f"  {metric_name}: {original:.6f} vs {loaded:.6f}, diff: {difference:.8f}")
    else:
        print(f"Model file {model_file} not found. Run training first.")

    single_molecule = dataset[0]

    if hasattr(single_molecule, 'smiles'):
        smiles = single_molecule.smiles
        print(f"SMILES: {smiles}")

        if task_type == 'regression':

            print("\nDemonstrating prediction from SMILES string directly:")
            smiles_prediction = predict_molecule(
                model, smiles, device, task_type)
            print(
                f"Prediction from SMILES: {smiles_prediction:.4f} log(mol/L)")
        elif task_type == 'classification':

            print("\nDemonstrating prediction from SMILES string directly:")
            pred_result = predict_molecule(model, smiles, device, task_type)

            if pred_result is None or (isinstance(pred_result, tuple) and (pred_result[0] is None or pred_result[1] is None)):
                print(f"Prediction from SMILES: Failed (could not process SMILES)")
            else:
                class_label, class_prob = pred_result

                if model.num_classes == 2:
                    print(
                        f"Prediction from SMILES: Class {class_label} with probability {class_prob:.4f}")
                else:
                    print(f"Prediction from SMILES: Class {class_label}")
                    print(
                        f"Class probabilities: {[f'{p:.4f}' for p in class_prob]}")
    else:

        print("SMILES not available, using data object directly")

    if task_type == 'regression':
        prediction = predict_molecule(
            model, single_molecule, device, task_type)

        if hasattr(single_molecule, 'y') and single_molecule.y is not None:

            if isinstance(single_molecule.y, torch.Tensor):
                actual_value = single_molecule.y.item()
            else:
                actual_value = single_molecule.y
            print(f"Actual solubility: {actual_value:.4f} log(mol/L)")
        print(f"Predicted solubility: {prediction:.4f} log(mol/L)")

        print("\nPredicting solubility for multiple molecules (first 10 compounds)...")

        molecules_to_predict = dataset.data_list[:10]
        smiles_list = []

        for mol in molecules_to_predict:
            if hasattr(mol, 'smiles'):
                smiles_list.append(mol.smiles)

        if not smiles_list and molecules_to_predict:
            print("SMILES not available for molecules, using data objects directly")
            predictions, _ = predict_molecules(
                model, molecules_to_predict, device, task_type)
        else:
            print(
                f"Predicting using SMILES strings for {len(smiles_list)} molecules")
            predictions, _ = predict_molecules(
                model, smiles_list, device, task_type)

        results = []
        for i, (mol, pred) in enumerate(zip(molecules_to_predict, predictions)):
            mol_smiles = mol.smiles if hasattr(mol, 'smiles') else "N/A"
            if hasattr(mol, 'y') and mol.y is not None:

                if isinstance(mol.y, torch.Tensor):
                    actual = mol.y.item()
                else:
                    actual = mol.y
            else:
                actual = "N/A"

            results.append({
                'Index': i,
                'SMILES': mol_smiles[:30] + '...' if len(mol_smiles) > 30 else mol_smiles,
                'Predicted': f"{pred:.4f}" if pred is not None else "Failed",
                'Actual': f"{actual:.4f}" if actual != "N/A" else "N/A"
            })

        if results:
            results_df = pd.DataFrame(results)
            print("\nPrediction results:")
            print(results_df.to_string(index=False))
        else:
            print("No predictions to display")

    elif task_type == 'classification':
        pred_result = predict_molecule(
            model, single_molecule, device, task_type)

        if pred_result is None or (isinstance(pred_result, tuple) and (pred_result[0] is None or pred_result[1] is None)):
            print("Prediction failed: Could not process molecule")
        else:
            class_label, class_prob = pred_result

            if hasattr(single_molecule, 'y') and single_molecule.y is not None:

                if isinstance(single_molecule.y, int):
                    actual_class = single_molecule.y

                else:
                    actual_class = int(single_molecule.y.item())
                print(f"Actual class: {actual_class}")

                if hasattr(dataset, 'class_names') and dataset.class_names:
                    actual_name = dataset.class_names[actual_class]
                    pred_name = dataset.class_names[class_label] if class_label is not None else "N/A"
                    print(f"Actual class name: {actual_name}")
                    print(f"Predicted class name: {pred_name}")

            if model.num_classes == 2:
                print(
                    f"Predicted class: {class_label} with probability {class_prob:.4f}")
            else:
                print(f"Predicted class: {class_label}")
                print(
                    f"Class probabilities: {[f'{p:.4f}' for p in class_prob]}")

        print("\nPredicting classes for multiple molecules (first 10 compounds)...")

        molecules_to_predict = dataset.data_list[:10]
        smiles_list = []

        for mol in molecules_to_predict:
            if hasattr(mol, 'smiles'):
                smiles_list.append(mol.smiles)

        if not smiles_list and molecules_to_predict:
            print("SMILES not available for molecules, using data objects directly")
            predictions, probabilities, _ = predict_molecules(
                model, molecules_to_predict, device, task_type)
        else:
            print(
                f"Predicting using SMILES strings for {len(smiles_list)} molecules")
            predictions, probabilities, _ = predict_molecules(
                model, smiles_list, device, task_type)

        results = []
        for i, (mol, pred, prob) in enumerate(zip(molecules_to_predict, predictions, probabilities)):
            mol_smiles = mol.smiles if hasattr(mol, 'smiles') else "N/A"
            actual = int(mol.y) if hasattr(
                mol, 'y') and mol.y is not None else "N/A"

            if prob is None:
                prob_str = "Failed"
            elif model.num_classes == 2:
                prob_str = f"{prob:.4f}"
            else:

                top_k = min(3, model.num_classes)
                if isinstance(prob, list):

                    sorted_indices = np.argsort(prob)[::-1][:top_k]
                    prob_str = ", ".join(
                        [f"C{idx}:{prob[idx]:.3f}" for idx in sorted_indices])
                else:
                    prob_str = "N/A"

            if hasattr(dataset, 'class_names') and dataset.class_names:
                actual_name = dataset.class_names[actual] if actual != "N/A" else "N/A"
                pred_name = dataset.class_names[pred] if pred is not None else "N/A"
                class_info = f"{pred}({pred_name})" if pred is not None else "Failed"
                actual_info = f"{actual}({actual_name})"
            else:
                class_info = str(pred) if pred is not None else "Failed"
                actual_info = str(actual)

            results.append({
                'Index': i,
                'SMILES': mol_smiles[:30] + '...' if len(mol_smiles) > 30 else mol_smiles,
                'Predicted Class': class_info,
                'Probability': prob_str,
                'Actual Class': actual_info
            })

        if results:
            results_df = pd.DataFrame(results)
            print("\nPrediction results:")
            print(results_df.to_string(index=False))
        else:
            print("No predictions to display")

    return {
        'model': model,
        'dataset': dataset,
        'best_metrics': best_metrics,
        'test_metrics': test_metrics,
        'history': history,
        'config': config
    }


if __name__ == "__main__":

    run_pipeline(DATASET_NAME, TARGET_COLUMN, LAYER_TYPE)

    from src.helper import update_model_registry
    update_model_registry()
