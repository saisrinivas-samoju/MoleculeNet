import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
import torch

def evaluate_model(model, loader, device):
    """
    Evaluate a model on a dataset
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    loader : torch_geometric.loader.DataLoader
        DataLoader for the dataset
    device : torch.device
        Device to use for evaluation
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    predictions : np.ndarray
        Array of predictions
    actual : np.ndarray
        Array of actual values
    """
    pass

# only for classification items
### Adding a new function for confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues', normalize=False):
    """
    Plot confusion matrix for classification results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        List of class names for axis labels
    figsize : tuple, optional
        Figure size (width, height)
    cmap : str, optional
        Colormap for the plot
    normalize : bool, optional
        Whether to normalize confusion matrix values to percentages
    """
    pass

# only for classification items
def evaluate_and_visualize(model, loader, device, class_names=None):
    """
    Evaluate model and visualize results including confusion matrix
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    loader : torch_geometric.loader.DataLoader
        DataLoader for the dataset
    device : torch.device
        Device to use for evaluation
    class_names : list, optional
        Names of classes for visualization
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    pass