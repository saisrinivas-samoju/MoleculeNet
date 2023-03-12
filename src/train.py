import time
from typing import Literal
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluate import evaluate_model
from config import task_type, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE

def train_epoch(model, optimizer, train_loader, criterion, device, task_type:Literal['classification', 'regression']=task_type):
    """
    Train model for one epoch
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    train_loader : torch_geometric.loader.DataLoader
        DataLoader for the training dataset
    criterion : torch.nn.Module
        Loss function
    device : torch.device
        Device to use for training
    task_type: Literal['classificatin', 'regression']
    Returns:
    --------
    epoch_loss : float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in train_loader:
        # Move data to device
        data = data.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Ensure correct data types
        x = data.x.float()
        edge_index = data.edge_index.long()
        batch = data.batch.long()
        
        # Forward pass
        output = model(x, edge_index, batch)
        if task_type == 'regression':
            target = data.y.view(-1, 1).float()
            
            # Calculate loss
            loss = criterion(output, target)
        elif task_type == 'classification':
            target = data.y.long()
            # Calculate loss
            if model.num_classes == 2:
                target = target.float()
                # Make sure dimensions match by using the minimum size
                min_size = min(output.size(0), target.size(0))
                output_selected = output[:min_size, 0]  # Take first min_size elements, column 0
                target_selected = target[:min_size]      # Take first min_size elements
                
                loss = criterion(output_selected, target_selected)
            else:
                # For multi-class, also ensure sizes match
                min_size = min(output.size(0), target.size(0))
                output_selected = output[:min_size]
                target_selected = target[:min_size]
                
                loss = criterion(output_selected, target_selected)
        else:
            raise Exception(f"InvalidTaskType: {task_type}")
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
    
    epoch_loss = total_loss / max(num_batches, 1)
    return epoch_loss

def _train_reg_model(model, optimizer, train_loader, val_loader, device, num_epochs=100, 
                patience=10, criterion=None, scheduler=None, verbose=True):
    # Set up loss function if not provided
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = None
    counter = 0
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'epoch_times': []
    }
    
    # Main training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
        
        # Evaluate on validation set
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        val_loss = val_metrics['MSE']  # Use MSE as the validation loss
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            ### Get current learning rate before stepping the scheduler
            prev_lr = optimizer.param_groups[0]['lr'] ### Store previous learning rate
            
            scheduler.step(val_loss)
            
            ### Get updated learning rate after stepping the scheduler
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0] ### Use recommended get_last_lr() method
            else:
                current_lr = optimizer.param_groups[0]['lr'] ### Fallback for schedulers without get_last_lr()
            
            ### Print notification if learning rate changed
            if prev_lr != current_lr and verbose:
                print(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}") ### Manual learning rate change notification
        
        # Record history
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val RMSE: {val_metrics['RMSE']:.6f} | "
                  f"Val R²: {val_metrics['R2']:.6f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics.copy()
            counter = 0
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_metrics

def _train_clf_model(model, optimizer, train_loader, val_loader, device, num_epochs=100, 
                patience=10, criterion=None, scheduler=None, verbose=True):
    # Set up loss function if not provided
    if criterion is None:
        if model.num_classes == 2:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking variables
    best_val_f1 = 0.0
    best_model_state = None
    best_metrics = None
    counter = 0
    
    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'epoch_times': []
    }
    
    # Main training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, optimizer, train_loader, criterion, device)
        
        # Evaluate on validation set
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        val_loss = val_metrics['Loss']
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]['lr']
            
            scheduler.step(1.0 - val_metrics['F1_macro'])
            
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            if prev_lr != current_lr and verbose:
                print(f"Learning rate changed from {prev_lr:.6f} to {current_lr:.6f}")
        
        # Record history
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['epoch_times'].append(epoch_time)
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val F1 (macro): {val_metrics['F1_macro']:.6f} | "
                  f"Val Accuracy: {val_metrics['Accuracy']:.6f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Check if this is the best model so far
        if val_metrics['F1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['F1_macro']
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics.copy()
            counter = 0
        else:
            counter += 1
            
        # Early stopping
        if counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, best_metrics

def train_model(model, optimizer, train_loader, val_loader, device, num_epochs=100, 
                patience=10, criterion=None, scheduler=None, verbose=True, task_type:Literal['classification','regression']=task_type):
    """
    Train a model with early stopping
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    train_loader : torch_geometric.loader.DataLoader
        DataLoader for the training dataset
    val_loader : torch_geometric.loader.DataLoader
        DataLoader for the validation dataset
    device : torch.device
        Device to use for training
    num_epochs : int
        Maximum number of epochs to train for
    patience : int
        Number of epochs to wait for improvement before early stopping
    criterion : torch.nn.Module
        Loss function (defaults to MSE Loss)
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    history : dict
        Dictionary containing training history
    best_metrics : dict
        Dictionary containing best validation metrics
    """
    if task_type=='classification':
        return _train_clf_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=num_epochs, patience=patience, criterion=criterion, scheduler=scheduler, verbose=verbose)
    elif task_type=='regression':
        return _train_reg_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, device=device, num_epochs=num_epochs, patience=patience, criterion=criterion, scheduler=scheduler, verbose=verbose)
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")
    
def plot_training_history(history, task_type:Literal['classification','regression']=task_type):
    plt.figure(figsize=(12, 5))
    """
    Plot training and validation loss over epochs
    
    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    """
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    if task_type=='regression':
        metrics = ['RMSE', 'R2']
        for metric in metrics:
            values = [m[metric] for m in history['val_metrics']]
            plt.plot(values, label=f'Validation {metric}')
    elif task_type=='classification':
        metrics = ['F1_macro', 'Accuracy', 'Precision_macro', 'Recall_macro']
        for metric in metrics:
            if any(metric in m for m in history['val_metrics']):
                values = [m[metric] for m in history['val_metrics']]
                plt.plot(values, label=f'Validation {metric}')  
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")
        
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def setup_training(model, train_loader, val_loader, device, learning_rate=0.001,
                  weight_decay=0, num_epochs=100, patience=10, verbose=True):
    """
    Set up and run the complete training process
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    train_loader : torch_geometric.loader.DataLoader
        DataLoader for the training dataset
    val_loader : torch_geometric.loader.DataLoader
        DataLoader for the validation dataset
    device : torch.device
        Device to use for training
    learning_rate : float
        Learning rate for the optimizer
    weight_decay : float
        Weight decay (L2 penalty) for the optimizer
    num_epochs : int
        Maximum number of epochs to train for
    patience : int
        Number of epochs to wait for improvement before early stopping
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    optimizer : torch.optim.Optimizer
        The optimizer used for training
    history : dict
        Dictionary containing training history
    metrics : dict
        Dictionary containing best validation metrics
    """
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE
    )
    
    # Train the model
    model, history, metrics = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        scheduler=scheduler,
        verbose=verbose
    )
    
    # Return the trained model and training info
    return model, optimizer, history, metrics