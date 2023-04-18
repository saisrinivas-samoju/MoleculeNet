import os
from typing import *
import torch
from src.model_architecture import MoleculeNetRegressor, MoleculeNetClassifier

def save_model(model, optimizer, model_info, metrics, model_path, model_name) -> None:

    os.makedirs(model_path, exist_ok=True)

    model_file = os.path.join(model_path, f'{model_name}_full.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_info': model_info,
        'metrics': metrics
    }, model_file)

    print(f"Model saved to {model_file}")


def load_model(model_path, device, task_type: Literal['classification', 'regression']) -> tuple:

    checkpoint = torch.load(model_path, map_location=device)

    model_info = checkpoint['model_info']

    if task_type == 'regression':

        model = MoleculeNetRegressor(
            num_features=model_info['num_features'],
            hidden_dim=model_info['hidden_dim'],
            layer_type=model_info['layer_type'],
            dropout_rate=model_info['dropout_rate']
        )
    elif task_type == 'classification':

        model = MoleculeNetClassifier(
            num_features=model_info['num_features'],
            hidden_dim=model_info['hidden_dim'],
            layer_type=model_info['layer_type'],
            dropout_rate=model_info['dropout_rate'],
            num_classes=model_info.get('num_classes', 2)
        )
    else:
        ValueError(f"InvalidTaskType: {task_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print("Model hyperparameters:")
    for param, value in model_info.items():
        print(f"  {param}: {value}")

    print("Model metrics:")
    for metric, value in checkpoint['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    return model, model_info, checkpoint['metrics']
