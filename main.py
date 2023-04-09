# Imports
from config import *
from src.data_loader import MoleculeDataset
from src.data_preprocessor import Preprocessor
from src.data_splitter import DataSplitter, ShuffleSplit
from src.model_architecture import MoleculeNetRegressor, MoleculeNetClassifier
from src.train import setup_training, plot_training_history
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from src.evaluate import *
from src.predict import predict_molecule, predict_molecules
from src.model_utils import save_model, load_model
import matplotlib.pyplot as plt
import os
import pandas as pd ### For displaying prediction results

# Data Loading
dataset = MoleculeDataset(root=root, name=name, filepath=filepath, smiles_colname=smiles_colname, label_colname=label_colname)

# Explore data


# Data Preprocessing
preprocessor = Preprocessor(dataset, task_type)
dataset = preprocessor.preprocess()

# Train Test Validation Splitting
splitter = DataSplitter()
splitter.set_strategy(ShuffleSplit())
train_loader, val_loader, test_loader = splitter.split_data(dataset, batch_size=BATCH_SIZE, random_seed=SEED)

# Model Building
# Get the number of features from the first data point in the dataset
sample_data = dataset[0]
num_features = sample_data.num_features

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if task_type=='regression':
    model = MoleculeNetRegressor(num_features=num_features, hidden_dim=HIDDEN_DIM, layer_type=LAYER_TYPE, dropout_rate=DROPOUT_RATE)
elif task_type=='classification':
    # Get number of classes from dataset
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else 2  # Default to binary if not specified
    model = MoleculeNetClassifier(num_features=num_features, hidden_dim=HIDDEN_DIM, layer_type=LAYER_TYPE, 
                                dropout_rate=DROPOUT_RATE, num_classes=num_classes)

model = model.to(device)

### TODO: Model Training # train.py
print("Training model...")
model, optimizer, history, best_metrics = setup_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    num_epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    verbose=True
)

if task_type=='regression':
    print(f"Best validation metrics: RMSE={best_metrics['RMSE']:.4f}, R²={best_metrics['R2']:.4f}")
if task_type=='classification':
    # Display classification metrics after training
    print("Best validation metrics:")
    print(f"  Accuracy: {best_metrics['Accuracy']:.4f}")
    print(f"  F1 (macro): {best_metrics['F1_macro']:.4f}")
    if 'F1' in best_metrics:  # For binary classification
        print(f"  Precision: {best_metrics['Precision']:.4f}")
        print(f"  Recall: {best_metrics['Recall']:.4f}")
    else:  # For multi-class
        print(f"  Precision (macro): {best_metrics['Precision_macro']:.4f}")
        print(f"  Recall (macro): {best_metrics['Recall_macro']:.4f}")

# Visualization of training progress
plot_training_history(history)

### TODO: Model Evaluation on the test data # if it doesn't fall under train.py, evaluate.py
print("\nEvaluating model on test data...")
test_metrics, test_predictions, test_actual = evaluate_model(model, test_loader, device)

# Print test metrics
print("Test metrics:")
for metric_name, metric_value in test_metrics.items():
    print(f"  {metric_name}: {metric_value:.4f}")

if task_type=='regression':
    # Visualize test predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_actual, test_predictions, alpha=0.5)

    # Add diagonal line representing perfect predictions
    min_val = min(min(test_actual), min(test_predictions))
    max_val = max(max(test_actual), max(test_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Set: Predicted vs Actual Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
elif task_type=='classification':
    # Visualize classification results with confusion matrix
    print("\nGenerating confusion matrix for test data...")

    # Get class names if available
    class_names = dataset.class_names if hasattr(dataset, 'class_names') else None

    # Plot confusion matrix
    plot_confusion_matrix(test_actual, test_predictions, class_names=class_names)

    # You can also use the combined function for evaluation and visualization
    print("\nDetailed evaluation with visualization:")
    evaluate_and_visualize(model, test_loader, device, class_names=class_names)

### TODO: Save the model # Maybe a separate file for saving and dumping
# Save the trained model
print("\nSaving model...")

if task_type=='regression':
    # Create a dictionary with model info
    model_info = {
        'num_features': num_features,
        'hidden_dim': HIDDEN_DIM,
        'layer_type': LAYER_TYPE,
        'dropout_rate': DROPOUT_RATE,
        'dataset': name,
        'task_type': task_type
    }
elif task_type=='classification':
    model_info = {
        'num_features': num_features,
        'hidden_dim': HIDDEN_DIM,
        'layer_type': LAYER_TYPE,
        'dropout_rate': DROPOUT_RATE,
        'num_classes': num_classes,
        'dataset': name,
        'task_type': task_type
    }

# Combine validation and test metrics
combined_metrics = {
    **best_metrics,
    **{f'test_{k}': v for k, v in test_metrics.items()}
}

# Save the model with all information
save_model(
    model=model,
    optimizer=optimizer,
    model_info=model_info,
    metrics=combined_metrics,
    model_path=MODEL_DIR,
    model_name=MODEL_NAME
)

### TODO: Load the model 
print("\nDemonstrating model loading...")

# Construct the full path to the saved model file
model_file = os.path.join(MODEL_DIR, f'{MODEL_NAME}_full.pt')

# Check if the saved model file exists
if os.path.exists(model_file):
    # Load the model from the saved file
    print(f"Loading model from {model_file}")
    loaded_model, loaded_model_info, loaded_metrics = load_model(model_file, device)
    
    # Verify the loaded model works by evaluating on test data
    print("\nEvaluating loaded model on test data...")
    loaded_test_metrics, loaded_predictions, loaded_actual = evaluate_model(loaded_model, test_loader, device)
    
    # Print loaded model metrics
    print("Loaded model test metrics:")
    for metric_name, metric_value in loaded_test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Compare with the original model metrics to verify loading worked correctly
    print("\nVerifying metrics match with original model:")
    for metric_name in test_metrics.keys():
        original = test_metrics[metric_name]
        loaded = loaded_test_metrics[metric_name]
        difference = abs(original - loaded)
        print(f"  {metric_name}: {original:.6f} vs {loaded:.6f}, diff: {difference:.8f}")
else:
    print(f"Model file {model_file} not found. Run training first.")

# ### TODO: Predict single data point (dataset[0]) using the model & Print # predict.py
# print("\nPredicting solubility for a single molecule (dataset[0])...") ### Added header for single molecule prediction

# Get the first molecule from the dataset
single_molecule = dataset[0] ### Get the first molecule

# Get SMILES representation if available
if hasattr(single_molecule, 'smiles'):
    smiles = single_molecule.smiles
    print(f"SMILES: {smiles}") ### Print SMILES if available
    
    if task_type=='regression':
        # Demonstrate prediction directly from SMILES
        print("\nDemonstrating prediction from SMILES string directly:") ### Added header for SMILES prediction
        smiles_prediction = predict_molecule(model, smiles, device) ### Predict using SMILES string
        print(f"Prediction from SMILES: {smiles_prediction:.4f} log(mol/L)") ### Print prediction from SMILES
    elif task_type=='classification':
        # Demonstrate prediction directly from SMILES
        print("\nDemonstrating prediction from SMILES string directly:")
        class_label, class_prob = predict_molecule(model, smiles, device)  ### Get both class and probability
        
        # Format the output based on binary or multi-class prediction
        if model.num_classes == 2:
            print(f"Prediction from SMILES: Class {class_label} with probability {class_prob:.4f}")
        else:
            print(f"Prediction from SMILES: Class {class_label}")
            print(f"Class probabilities: {[f'{p:.4f}' for p in class_prob]}")
else:
    # If SMILES not available, use the data object
    print("SMILES not available, using data object directly") ### Print message about missing SMILES
    
# Predict molecule property using the trained model and data object
if task_type=='regression':
    prediction = predict_molecule(model, single_molecule, device) ### Make prediction using trained model
    
    # Get actual value if available
    actual_value = single_molecule.y

    # Print results
    if hasattr(single_molecule, 'y') and single_molecule.y is not None:
        print(f"Actual solubility: {actual_value:.4f} log(mol/L)") ### Print actual value if available
    print(f"Predicted solubility: {prediction:.4f} log(mol/L)") ### Print predicted value

    ### TODO: Predict multiple data points (dataset[0:10]) using the model & Print # predict.py
    print("\nPredicting solubility for multiple molecules (first 10 compounds)...") ### Added header for multiple prediction

    # Extract SMILES from the first 10 molecules
    molecules_to_predict = dataset.data_list[:10] ### Get first 10 molecules
    smiles_list = [] ### Initialize SMILES list

    # Extract SMILES strings when available
    for mol in molecules_to_predict:
        if hasattr(mol, 'smiles'):
            smiles_list.append(mol.smiles) ### Add SMILES to list if available

    # If we couldn't get SMILES strings, use the data objects directly
    if not smiles_list and molecules_to_predict:
        print("SMILES not available for molecules, using data objects directly") ### Print message if SMILES not available
        predictions, _ = predict_molecules(model, molecules_to_predict, device) ### Predict using data objects
    else:
        print(f"Predicting using SMILES strings for {len(smiles_list)} molecules") ### Print message for SMILES prediction
        predictions, _ = predict_molecules(model, smiles_list, device) ### Predict using SMILES strings

    # Create a DataFrame to display predictions
    results = [] ### Initialize results list
    for i, (mol, pred) in enumerate(zip(molecules_to_predict, predictions)):
        mol_smiles = mol.smiles if hasattr(mol, 'smiles') else "N/A" ### Get SMILES if available
        actual = mol.y if hasattr(mol, 'y') and mol.y is not None else "N/A" ### Get actual value if available
        
        results.append({
            'Index': i, 
            'SMILES': mol_smiles[:30] + '...' if len(mol_smiles) > 30 else mol_smiles, ### Truncate long SMILES
            'Predicted': f"{pred:.4f}" if pred is not None else "Failed", ### Format prediction
            'Actual': f"{actual:.4f}" if actual != "N/A" else "N/A" ### Format actual value if available
        }) ### Add result to list

    # Display predictions
    if results:
        results_df = pd.DataFrame(results) ### Create DataFrame from results
        print("\nPrediction results:") ### Print header
        print(results_df.to_string(index=False)) ### Print results table
    else:
        print("No predictions to display") ### Print message if no results
    
elif task_type=='classification':
    class_label, class_prob = predict_molecule(model, single_molecule, device)  ### Get both class and probability
    
    # Get actual value if available
    if hasattr(single_molecule, 'y') and single_molecule.y is not None:
        # Handle case where y is already an int
        if isinstance(single_molecule.y, int):
            actual_class = single_molecule.y
        # Handle case where y is a tensor
        else:
            actual_class = int(single_molecule.y.item())
        print(f"Actual class: {actual_class}")
        
        # Print class names if available
        if hasattr(dataset, 'class_names') and dataset.class_names:
            actual_name = dataset.class_names[actual_class]
            pred_name = dataset.class_names[class_label]
            print(f"Actual class name: {actual_name}")
            print(f"Predicted class name: {pred_name}")

    # Format the output based on binary or multi-class prediction
    if model.num_classes == 2:
        print(f"Predicted class: {class_label} with probability {class_prob:.4f}")
    else:
        print(f"Predicted class: {class_label}")
        print(f"Class probabilities: {[f'{p:.4f}' for p in class_prob]}")

    ### TODO: Predict multiple data points (dataset[0:10]) using the model & Print # predict.py
    print("\nPredicting classes for multiple molecules (first 10 compounds)...")  ### Updated header for classification

    # Extract SMILES from the first 10 molecules
    molecules_to_predict = dataset.data_list[:10]
    smiles_list = []

    # Extract SMILES strings when available
    for mol in molecules_to_predict:
        if hasattr(mol, 'smiles'):
            smiles_list.append(mol.smiles)

    # If we couldn't get SMILES strings, use the data objects directly
    if not smiles_list and molecules_to_predict:
        print("SMILES not available for molecules, using data objects directly")
        predictions, probabilities, _ = predict_molecules(model, molecules_to_predict, device)  ### Get both predictions and probabilities
    else:
        print(f"Predicting using SMILES strings for {len(smiles_list)} molecules")
        predictions, probabilities, _ = predict_molecules(model, smiles_list, device)  ### Get both predictions and probabilities

    # Create a DataFrame to display predictions
    results = []
    for i, (mol, pred, prob) in enumerate(zip(molecules_to_predict, predictions, probabilities)):
        mol_smiles = mol.smiles if hasattr(mol, 'smiles') else "N/A"
        actual = int(mol.y) if hasattr(mol, 'y') and mol.y is not None else "N/A"
        
        # Format probability output based on binary or multi-class
        if model.num_classes == 2:
            prob_str = f"{prob:.4f}"
        else:
            # For multi-class, show top 3 probabilities or fewer if num_classes < 3
            top_k = min(3, model.num_classes)
            if isinstance(prob, list):
                # Sort probabilities and get indices of top k
                sorted_indices = np.argsort(prob)[::-1][:top_k]
                prob_str = ", ".join([f"C{idx}:{prob[idx]:.3f}" for idx in sorted_indices])
            else:
                prob_str = "N/A"
        
        # Add class names if available
        if hasattr(dataset, 'class_names') and dataset.class_names:
            actual_name = dataset.class_names[actual] if actual != "N/A" else "N/A"
            pred_name = dataset.class_names[pred] if pred is not None else "N/A"
            class_info = f"{pred}({pred_name})"
            actual_info = f"{actual}({actual_name})"
        else:
            class_info = str(pred)
            actual_info = str(actual)
        
        results.append({
            'Index': i, 
            'SMILES': mol_smiles[:30] + '...' if len(mol_smiles) > 30 else mol_smiles,
            'Predicted Class': class_info,
            'Probability': prob_str,
            'Actual Class': actual_info
        })

    # Display predictions
    if results:
        results_df = pd.DataFrame(results)
        print("\nPrediction results:")
        print(results_df.to_string(index=False))
    else:
        print("No predictions to display")