import os
from typing import *
import torch
from src.model_architecture import MoleculeNetRegressor, MoleculeNetClassifier

def save_model(model, optimizer, model_info, metrics, model_path, model_name, task_type: Literal['classification', 'regression']) -> None:
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
    pass

def load_model(model_path, device, task_type: Literal['classification', 'regression']) -> tuple:
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
    pass