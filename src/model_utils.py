import os
from typing import *
import torch
from config import task_type
from src.model_architecture import MoleculeNetRegressor, MoleculeNetClassifier

def save_model(model, optimizer, model_info, metrics, model_path, model_name) -> None:
    """
    Save model with metadata
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    optimizer : torch.optim.Optimizer
        The optimizer used for training
    model_info : dict
        Dictionary containing model hyperparameters
    metrics : dict
        Dictionary containing evaluation metrics
    model_path : str
        Directory to save model
    model_name : str
        Name of the model file
    """
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Create full path
    model_file = os.path.join(model_path, f'{model_name}_full.pt')
    
    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_info': model_info,
        'metrics': metrics
    }, model_file)
    
    print(f"Model saved to {model_file}")

def load_model(model_path, device, task_type: Literal['classification', 'regression']=task_type) -> tuple:
    """
    Load model with metadata
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    device : torch.device
        Device to load the model on
        
    Returns:
    --------
    model : torch.nn.Module
        The loaded model
    model_info : dict
        Dictionary containing model hyperparameters
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model info
    model_info = checkpoint['model_info']
    
    if task_type=='regression':
        # Create model with same architecture
        model = MoleculeNetRegressor(
            num_features=model_info['num_features'],
            hidden_dim=model_info['hidden_dim'],
            layer_type=model_info['layer_type'],
            dropout_rate=model_info['dropout_rate']
        )
    elif task_type=='classification':
        # Create model with same architecture
        model = MoleculeNetClassifier(
            num_features=model_info['num_features'],
            hidden_dim=model_info['hidden_dim'],
            layer_type=model_info['layer_type'],
            dropout_rate=model_info['dropout_rate'],
            num_classes=model_info.get('num_classes', 2)
        )
    else:
        ValueError(f"InvalidTaskType: {task_type}")
        
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    print(f"Model loaded from {model_path}")
    print("Model hyperparameters:")
    for param, value in model_info.items():
        print(f"  {param}: {value}")
        
    print("Model metrics:")
    for metric, value in checkpoint['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return model, model_info, checkpoint['metrics'] 