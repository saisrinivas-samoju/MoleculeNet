import math
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
import torch

def _evaluate_model_reg(model, loader, device):
    model.eval()
    predictions = []
    actual = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Ensure correct data types
            x = data.x.float() # Convert features to float
            edge_index = data.edge_index.long() # Convert edge indices to long
            batch = data.batch.long() # Convert batch indices to long
            
            output = model(x, edge_index, batch) # Use type-converted tensors
            pred = output.cpu().numpy()
            target = data.y.view(-1, 1).float().cpu().numpy() # Ensure target is float
            
            predictions.extend(pred)
            actual.extend(target)
    
    predictions = np.array(predictions).flatten()
    actual = np.array(actual).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }, predictions, actual
    
def _evaluate_model_clf(model, loader, device):
    model.eval()
    predictions = []
    prediction_probs = []
    actual = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # Ensure correct data types
            x = data.x.float() # Convert features to float
            edge_index = data.edge_index.long() # Convert edge indices to long
            batch = data.batch.long() # Convert batch indices to long
            
            output = model(x, edge_index, batch) # Use type-converted tensors
            
            # Handle binary vs multi-class classification
            if model.num_classes == 2:
                # For binary classification
                pred_prob = output[:, 0].cpu().numpy()
                pred_label = (pred_prob > 0.5).astype(int)
                target = data.y.cpu().numpy()
            else:
                # For multi-class classification
                pred_prob = output[:, 0].cpu().numpy()
                pred_label = np.argmax(pred_prob, axis=1)
                target = data.y.cpu().numpy()
            
            predictions.extend(pred_label)
            prediction_probs.extend(pred_prob)
            actual.extend(target)
    
    predictions = np.array(predictions).flatten()
    actual = np.array(actual).flatten()
    prediction_probs = np.array(prediction_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(actual, predictions)
    
    # Handle potential warnings for binary case
    if model.num_classes == 2:
        precision_macro = precision_score(actual, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(actual, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(actual, predictions, average='macro', zero_division=0)
        
        # Also calculate binary metrics
        precision = precision_score(actual, predictions, zero_division=0)
        recall = recall_score(actual, predictions, zero_division=0)
        f1 = f1_score(actual, predictions, zero_division=0)
        
        # Calculate cross-entropy loss
        epsilon = 1e-15
        prediction_probs = np.clip(prediction_probs, epsilon, 1 - epsilon)
        loss = log_loss(actual, prediction_probs)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Precision_macro': precision_macro,
            'Recall_macro': recall_macro,
            'F1_macro': f1_macro,
            'Loss': loss
        }
    else:
        # Multi-class metrics
        precision_macro = precision_score(actual, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(actual, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(actual, predictions, average='macro', zero_division=0)
        
        # For multi-class, also add weighted versions
        precision_weighted = precision_score(actual, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(actual, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(actual, predictions, average='weighted', zero_division=0)
        
        # Calculate cross-entropy loss
        loss = log_loss(actual, prediction_probs)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision_macro': precision_macro,
            'Recall_macro': recall_macro,
            'F1_macro': f1_macro,
            'Precision_weighted': precision_weighted,
            'Recall_weighted': recall_weighted,
            'F1_weighted': f1_weighted,
            'Loss': loss
        }
    
    return metrics, predictions, actual

def evaluate_model(model, loader, device, task_type: Literal['classification', 'regression']):
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
    if task_type=='classification':
        return _evaluate_model_clf(model, loader, device)
    elif task_type=='regression':
        return _evaluate_model_reg(model, loader, device)
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")
    
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
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(i) for i in range(len(unique_classes))]
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        title = 'Normalized Confusion Matrix (%)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    
    # Add labels and title
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

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
    # Evaluate model
    metrics, predictions, actual = evaluate_model(model, loader, device)
    
    # Print metrics
    print("Classification Report:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(actual, predictions, class_names=class_names)
    
    return metrics