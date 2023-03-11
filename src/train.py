import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluate import evaluate_model

def train_epoch(model, optimizer, train_loader, criterion, device):
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
        target = data.y.view(-1, 1).float()
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
    
    epoch_loss = total_loss / max(num_batches, 1)
    return epoch_loss