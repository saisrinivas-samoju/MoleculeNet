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
            x = data.x.float()
            edge_index = data.edge_index.long()
            batch = data.batch.long()

            output = model(x, edge_index, batch)
            pred = output.cpu().numpy()
            target = data.y.view(-1, 1).float().cpu().numpy()

            predictions.extend(pred)
            actual.extend(target)

    predictions = np.array(predictions).flatten()
    actual = np.array(actual).flatten()

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

            x = data.x.float()
            edge_index = data.edge_index.long()
            batch = data.batch.long()

            output = model(x, edge_index, batch)

            if model.num_classes == 2:

                pred_prob = output.squeeze(-1).cpu().numpy()
                pred_label = (pred_prob > 0.5).astype(int)
            else:

                pred_prob = np.exp(output.cpu().numpy())
                pred_label = np.argmax(pred_prob, axis=1)

            target = data.y.cpu().numpy()

            if target.ndim > 1:

                if target.shape[0] == len(pred_label):

                    target = target[:, 0] if target.shape[1] > 1 else target.flatten()[
                        :len(pred_label)]
                else:

                    target_flat = target.flatten()
                    target = target_flat[:len(pred_label)]
            else:

                target = target.flatten()
                if len(target) != len(pred_label):

                    target = target[:len(pred_label)]

            pred_prob = pred_prob.flatten()
            pred_label = pred_label.flatten()

            min_len = min(len(pred_label), len(target), len(pred_prob))
            if min_len > 0:
                pred_label = pred_label[:min_len]
                pred_prob = pred_prob[:min_len]
                target = target[:min_len]

                predictions.extend(pred_label.tolist())
                prediction_probs.extend(pred_prob.tolist())
                actual.extend(target.tolist())

    predictions = np.array(predictions).flatten()
    actual = np.array(actual).flatten()
    prediction_probs = np.array(prediction_probs)

    min_len = min(len(predictions), len(actual))
    if len(predictions) != len(actual):
        print(
            f"Warning: Mismatch in lengths - predictions: {len(predictions)}, actual: {len(actual)}. Truncating to {min_len}.")
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        if prediction_probs.ndim == 1:
            prediction_probs = prediction_probs[:min_len]
        else:
            prediction_probs = prediction_probs[:min_len]

    accuracy = accuracy_score(actual, predictions)

    if model.num_classes == 2:
        precision_macro = precision_score(
            actual, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(
            actual, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(actual, predictions,
                            average='macro', zero_division=0)

        precision = precision_score(actual, predictions, zero_division=0)
        recall = recall_score(actual, predictions, zero_division=0)
        f1 = f1_score(actual, predictions, zero_division=0)

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

        precision_macro = precision_score(
            actual, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(
            actual, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(actual, predictions,
                            average='macro', zero_division=0)

        precision_weighted = precision_score(
            actual, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(
            actual, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(actual, predictions,
                               average='weighted', zero_division=0)

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
    if task_type == 'classification':
        return _evaluate_model_clf(model, loader, device)
    elif task_type == 'regression':
        return _evaluate_model_reg(model, loader, device)
    else:
        raise ValueError(f"InvalidTaskType: {task_type}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues', normalize=False):

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(i) for i in range(len(unique_classes))]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
        title = 'Normalized Confusion Matrix (%)'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)

    plt.tight_layout()

    plt.show()


def evaluate_and_visualize(model, loader, device, task_type: Literal['classification', 'regression'], class_names=None):
    if task_type != 'classification':
        raise ValueError(
            f"evaluate_and_visualize is only for classification tasks, got task_type='{task_type}'")

    metrics, predictions, actual = evaluate_model(
        model, loader, device, task_type)

    print("Classification Report:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(actual, predictions, class_names=class_names)

    return metrics
