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

def train_epoch(model, optimizer, train_loader, criterion, device, task_type:Literal['classification', 'regression']):
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