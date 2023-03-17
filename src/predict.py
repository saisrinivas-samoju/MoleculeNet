import torch
from torch_geometric.utils import from_smiles

def predict_molecule(model, data, device):
    model.eval()
    with torch.no_grad():
        # Convert SMILES to data object if a string is provided
        if isinstance(data, str):
            try:
                data = from_smiles(data)
            except Exception as e:
                print(f"Error converting SMILES to graph: {e}")
                return None
        
        # Create a batch with just this molecule
        data = data.to(device)
        
        # Our model expects batched data, so we need to add batch information
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        
        # Ensure correct data types
        x = data.x.float()  # Convert features to float
        edge_index = data.edge_index.long()  # Convert edge indices to long
        batch = data.batch.long()  # Convert batch indices to long
        
        prediction = model(x, edge_index, batch)  # Use type-converted tensors
        return prediction.item()

def predict_molecules(model, data_list, device):
    predictions = []
    smiles_list = []
    
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
        
        # Make prediction
        pred_val = predict_molecule(model, item, device)
        predictions.append(pred_val)
    
    return predictions, smiles_list 