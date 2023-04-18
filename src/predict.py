from typing import *
import numpy as np
import torch
from torch_geometric.utils import from_smiles


def predict_molecule(model, data, device, task_type: Literal['regression', 'classification']):
    model.eval()
    with torch.no_grad():

        if isinstance(data, str):
            try:
                data = from_smiles(data)
            except Exception as e:
                print(f"Error converting SMILES to graph: {e}")
                if task_type == 'regression':
                    return None
                elif task_type == 'classification':
                    return None, None

        if not hasattr(data, 'x') or data.x is None or data.x.size(0) == 0:
            print(f"Error: Empty graph generated from SMILES. Graph has 0 nodes.")
            if task_type == 'regression':
                return None
            elif task_type == 'classification':
                return None, None

        data = data.to(device)

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.size(
                0), dtype=torch.long, device=device)

        x = data.x.float()
        edge_index = data.edge_index.long()
        batch = data.batch.long()

        if x.size(0) == 0 or batch.size(0) == 0:
            print(
                f"Error: Empty graph detected before model call. x.shape: {x.shape}, batch.shape: {batch.shape}")
            if task_type == 'regression':
                return None
            elif task_type == 'classification':
                return None, None

        if task_type == 'regression':
            prediction = model(x, edge_index, batch)
            return prediction.item()
        elif task_type == 'classification':
            output = model(x, edge_index, batch)

            if model.num_classes == 2:

                output_squeezed = output.squeeze(-1)

                if output_squeezed.numel() == 0:
                    raise ValueError(
                        f"Model output is empty. Output shape: {output.shape}, Squeezed shape: {output_squeezed.shape}")

                prob_np = output_squeezed.cpu().numpy()
                if prob_np.size == 0:
                    raise ValueError(
                        f"Model output has 0 elements after processing. Output shape: {output.shape}")

                prob = float(prob_np.item() if prob_np.ndim ==
                             0 else prob_np[0])
                prediction = 1 if prob >= 0.5 else 0
                return prediction, prob
            else:

                output_np = output.cpu().numpy()

                if output_np.size == 0:
                    raise ValueError(
                        f"Model output is empty. Output shape: {output.shape}")

                probs = np.exp(output_np)

                if probs.ndim == 2 and probs.shape[0] == 1:
                    probs = probs[0]
                elif probs.ndim == 1:
                    pass
                else:
                    raise ValueError(
                        f"Unexpected output shape for multi-class: {probs.shape}")

                prediction = int(np.argmax(probs))

                prob = float(np.max(probs))
                return prediction, prob
        else:
            raise ValueError(f"InvalidTaskType: {task_type}")


def predict_molecules(model, data_list, device, task_type: Literal['regression', 'classification']):
    predictions = []
    smiles_list = []
    probabilities = []

    for item in data_list:

        if isinstance(item, str):
            smiles = item
            smiles_list.append(smiles)
        elif hasattr(item, 'smiles'):
            smiles = item.smiles
            smiles_list.append(smiles)
        else:
            smiles_list.append(None)

        if task_type == 'regression':

            pred_val = predict_molecule(model, item, device, task_type)
            predictions.append(pred_val)
        elif task_type == 'classification':

            pred_result = predict_molecule(model, item, device, task_type)

            if pred_result is None or (isinstance(pred_result, tuple) and (pred_result[0] is None or pred_result[1] is None)):

                predictions.append(None)
                probabilities.append(None)
            else:
                pred_class, pred_prob = pred_result
                predictions.append(pred_class)
                probabilities.append(pred_prob)
        else:
            raise ValueError(f"InvalidTaskType: {task_type}")

    if task_type == 'regression':
        return predictions, smiles_list
    elif task_type == 'classification':
        return predictions, probabilities, smiles_list
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")
