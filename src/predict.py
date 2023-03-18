from typing import *
import numpy as np
import torch
from torch_geometric.utils import from_smiles
from config import task_type

def predict_molecule(model, data, device, task_type: Literal['regression', 'classification']=task_type):
    """
    Predict a property for a single molecule
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to use for prediction
    data : torch_geometric.data.Data or str
        The molecule data or SMILES string
    device : torch.device
        Device to use for prediction
        
    Returns:
    --------
    prediction : int or list
        The predicted class (for binary) or class probabilities (for multi-class)
    prob : float or list
        The prediction probability or probabilities
    """
    model.eval()
    with torch.no_grad():
        # Convert SMILES to data object if a string is provided
        if isinstance(data, str):
            try:
                data = from_smiles(data)
            except Exception as e:
                print(f"Error converting SMILES to graph: {e}")
                if task_type=='regression':
                    return None
                elif task_type=='classification':
                    return None, None
        
        # Create a batch with just this molecule
        data = data.to(device)
        
        # Our model expects batched data, so we need to add batch information
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        
        # Ensure correct data types
        x = data.x.float()  # Convert features to float
        edge_index = data.edge_index.long()  # Convert edge indices to long
        batch = data.batch.long()  # Convert batch indices to long
        
        if task_type=='regression':
            prediction = model(x, edge_index, batch)  # Use type-converted tensors
            return prediction.item()
        elif task_type=='classification':
            output = model(x, edge_index, batch)  # Use type-converted tensors
            
            # Handle binary vs multi-class classification
            if model.num_classes == 2:
                # Binary classification - get probability for positive class
                probs = output.cpu().numpy()
                # Get probability for positive class (second column)
                prob = probs[0, 1] if len(probs.shape) > 1 else probs[0]
                prediction = 1 if prob >= 0.5 else 0
                return prediction, prob
            else:
                # Multi-class classification
                probs = output.cpu().numpy()
                prediction = int(np.argmax(probs))
                return prediction, probs.tolist()[0]
        else:
            raise ValueError(f"InvalidTaskType: {task_type}")

def predict_molecules(model, data_list, device, task_type: Literal['regression', 'classification']=task_type):
    """
    Predict properties for multiple molecules
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to use for prediction
    data_list : list
        List of Data objects or SMILES strings
    device : torch.device
        Device to use for prediction
        
    Returns:
    --------
    predictions : list
        List of predicted classes
    probabilities : list
        List of prediction probabilities
    smiles_list : list
        List of SMILES strings if available
    """
    predictions = []
    smiles_list = []
    probabilities = []
    
    for item in data_list:
        # Store SMILES if available
        if isinstance(item, str):
            smiles = item
            smiles_list.append(smiles)
        elif hasattr(item, 'smiles'):
            smiles = item.smiles
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)
        
        if task_type=='regression':
            # Make prediction
            pred_val = predict_molecule(model, item, device)
            predictions.append(pred_val)
        elif task_type=='classification':
            # Make prediction
            pred_class, pred_prob = predict_molecule(model, item, device)
            predictions.append(pred_class)
            probabilities.append(pred_prob)
        else:
            raise ValueError(f"InvalidTaskType: {task_type}")
            
    if task_type=='regression':
        return predictions, smiles_list
    elif task_type=='classification':
        return predictions, probabilities, smiles_list 
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")