# Project Documentation

Here's how to use the different parts of this molecular property prediction system.

## Data Loading

Load molecular data from CSV files. The system expects SMILES strings and property values.

=== "Basic Usage"
    ```python
    from src.data_loader import MoleculeDataset
    
    # Load data from a CSV file
    dataset = MoleculeDataset(
        root='datasets/processed',
        name='ESOL',
        filepath='datasets/csv_files/delaney-processed.csv',
        smiles_colname='smiles',
        label_colname='ESOL predicted log solubility in mols per litre'
    )
    
    print(f"Loaded {len(dataset)} molecules")
    ```

=== "Available Datasets"
    Here are the datasets you can use:
    
    | Dataset | Property | Type | Size |
    | ------- | -------- | ---- | ---- |
    | ESOL | Aqueous Solubility | Regression | 1,128 |
    | FreeSolv | Hydration Free Energy | Regression | 642 |
    | Lipophilicity | Octanol/Water Distribution | Regression | 4,200 |
    | BBBP | Blood-Brain Barrier Penetration | Classification | 2,039 |
    | BACE | β-secretase Inhibition | Classification | 1,513 |
    | HIV | HIV Replication Inhibition | Classification | 41,127 |
    | SIDER | Side Effect | Classification | 1,427 |
    | ClinTox | Clinical Toxicity | Classification | 1,478 |

The `MoleculeDataset` class does a few things:

- Loads molecule data from CSV files
- Stores SMILES strings and their property values
- Converts molecules into PyTorch Geometric Data objects
- Lets you access molecules by index and check how many you have

## Data Preprocessing

This turns raw molecular data into graphs that neural networks can work with.

=== "Preprocessing Steps"
    ```python
    from src.data_preprocessor import Preprocessor
    
    # Initialize preprocessor
    preprocessor = Preprocessor(dataset, task_type='regression')
    
    # Preprocess the dataset
    processed_dataset = preprocessor.preprocess()
    ```

=== "Feature Engineering"
    The preprocessor pulls out different features:
    
    - **Atom Features**: What element it is, chirality, hybridization, whether it's in a ring, etc.
    - **Bond Features**: Single, double, triple bonds, conjugation, ring membership
    - **Molecular Graphs**: Nodes are atoms, edges are bonds, all with their features attached

The preprocessing pipeline:

1. Takes SMILES strings
2. Converts them to molecular graphs
3. Extracts features for each atom and bond
4. Creates PyTorch Geometric Data objects

## Data Splitting

Split your data into training, validation, and test sets.

=== "Basic Splitting"
    ```python
    from src.data_splitter import DataSplitter, ShuffleSplit
    
    # Initialize splitter with a strategy
    splitter = DataSplitter()
    splitter.set_strategy(ShuffleSplit())
    
    # Split the data
    train_loader, val_loader, test_loader = splitter.split_data(
        processed_dataset, 
        batch_size=32, 
        random_seed=42
    )
    ```

=== "Available Strategies"
    You can use different ways to split:
    
    - **ShuffleSplit**: Random splitting (default)
    - **ScaffoldSplit**: Split based on molecular scaffolds
    - **TemporalSplit**: Split based on time if your data has that info

The data splitter makes sure:

- Data gets distributed properly across train, val, and test
- Batch sizes stay consistent for training
- You can reproduce splits with a random seed

## Model Architecture

You can use regression or classification models depending on what you're trying to predict.

=== "Regression Model"
    ```python
    from src.model_architecture import MoleculeNetRegressor
    import torch
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoleculeNetRegressor(
        num_features=dataset[0].num_features,
        hidden_dim=64,
        layer_type='gcn',
        dropout_rate=0.2
    )
    model = model.to(device)
    ```

=== "Classification Model"
    ```python
    from src.model_architecture import MoleculeNetClassifier
    import torch
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoleculeNetClassifier(
        num_features=dataset[0].num_features,
        hidden_dim=64,
        layer_type='gcn',
        dropout_rate=0.2,
        num_classes=2  # Binary classification
    )
    model = model.to(device)
    ```

=== "Available Architectures"
    You can use different graph neural network types:
    
    - **GCN**: Graph Convolutional Network (default)
    - **GAT**: Graph Attention Network
    - **GraphSAGE**: Graph SAmple and aggreGatE

The model classes:

- Take molecular graphs as input
- Process them through graph layers
- Output predictions (numbers for regression, class labels for classification)

## Model Training

Train your models with early stopping and learning rate scheduling built in.

=== "Basic Training"
    ```python
    from src.train import setup_training, plot_training_history
    
    # Train the model
    model, optimizer, history, best_metrics = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=100,
        patience=10,
        task_type='regression'
    )
    
    # Plot training progress
    plot_training_history(history, task_type='regression')
    
    print(f"Best validation metrics: RMSE={best_metrics['RMSE']:.4f}, R²={best_metrics['R2']:.4f}")
    ```

=== "Advanced Options"
    ```python
    # Train with additional options
    model, optimizer, history, best_metrics = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,  # L2 regularization
        num_epochs=100,
        patience=10,
        scheduler_factor=0.5,  # Learning rate reduction factor
        scheduler_patience=5,  # Epochs before reducing learning rate
        verbose=True,  # Print progress
        task_type='regression'
    )
    ```

The training module gives you:

- Early stopping so you don't overfit
- Learning rate scheduling to help convergence
- Training history so you can see what happened
- Plots of training progress

## Model Evaluation

See how well your model performs on test data.

=== "Basic Evaluation"
    ```python
    from src.evaluate import evaluate_model
    import matplotlib.pyplot as plt
    
    # Evaluate on test data
    test_metrics, test_predictions, test_actual = evaluate_model(
        model, test_loader, device, task_type='regression'
    )
    
    # Print metrics
    print("Test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Visualize predictions vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(test_actual, test_predictions, alpha=0.5)
    plt.plot([min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Set: Predicted vs Actual Values')
    plt.grid(True, alpha=0.3)
    plt.show()
    ```

The evaluation module calculates:

- **For Regression**: RMSE, MAE, R², Pearson correlation
- **For Classification**: Accuracy, precision, recall, F1-score, ROC-AUC

## Model Saving and Loading

Save trained models so you can use them later or deploy them.

=== "Saving a Model"
    ```python
    from src.model_utils import save_model
    import os
    
    # Ensure model directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create model info dictionary
    model_info = {
        'num_features': dataset[0].num_features,
        'hidden_dim': 64,
        'layer_type': 'gcn',
        'dataset': 'ESOL',
        'target_column': 'ESOL predicted log solubility in mols per litre',
        'task_type': 'regression'
    }
    
    # Save the model
    save_model(
        model=model,
        optimizer=optimizer,
        model_info=model_info,
        metrics=test_metrics,
        model_path='models',
        model_name='esol_gcn'
    )
    ```

=== "Loading a Model"
    ```python
    from src.model_utils import load_model
    
    # Load the model
    loaded_model, loaded_info, loaded_metrics = load_model(
        'models/esol_gcn_full.pt', device, task_type='regression'
    )
    
    print(f"Loaded model info: {loaded_info}")
    print(f"Loaded model metrics: {loaded_metrics}")
    ```

The model utilities let you:

- Save models with all their metadata
- Load models back easily
- Store model architecture, weights, and performance metrics together
- Seperate model files from their configuration

## Making Predictions

Make predictions for new molecules using SMILES strings.

=== "Single Molecule Prediction"
    ```python
    from src.predict import predict_molecule
    
    # Predict property for a single molecule using SMILES
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    
    # For regression
    prediction = predict_molecule(model, smiles, device, task_type='regression')
    print(f"Predicted solubility: {prediction:.4f} log(mol/L)")
    
    # For classification
    class_label, class_prob = predict_molecule(model, smiles, device, task_type='classification')
    print(f"Predicted class: {class_label} with probability {class_prob:.4f}")
    ```

=== "Multiple Molecule Predictions"
    ```python
    from src.predict import predict_molecules
    import pandas as pd
    
    # List of SMILES strings
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    # Predict properties for multiple molecules
    # For regression
    predictions, _ = predict_molecules(model, smiles_list, device, task_type='regression')
    
    # For classification
    predictions, probabilities, _ = predict_molecules(model, smiles_list, device, task_type='classification')
    
    # Create results dataframe
    results = pd.DataFrame({
        'SMILES': smiles_list,
        'Predicted': [f"{pred:.4f}" for pred in predictions]
    })
    
    print(results)
    ```

The prediction module lets you:

- Predict from SMILES strings directly
- Handle batches of molecules
- Work with both regression and classification models

## Web Interface

There's a web interface you can use to make predictions without writing code.

=== "Starting the Server"
    ```bash
    # Start the FastAPI server
    uvicorn app:app --reload
    
    # The web interface will be at http://localhost:8000
    ```

=== "Using the API"
    ```python
    import requests
    
    # Make a prediction via API
    response = requests.post(
        'http://localhost:8000/molecule-net/predict',
        json={
            'query': 'aspirin',  # Can be SMILES or compound name
            'model_ids': ['esol_solubility']  # Optional, uses all models if not specified
        }
    )
    
    result = response.json()
    print(result)
    ```

The web interface provides:

- A browser based UI for making predictions
- Support for SMILES strings or compound names (resolves names to SMILES)
- 3D molecular visualization
- Model registry to manage multiple trained models
- API endpoints for programmatic access

## Model Registry

The model registry keeps track of all your trained models and makes them available through the web interface.

=== "Model Registry Structure"
    The `model_registry.json` file contains information about each model:
    
    - Model ID and name
    - File path to the saved model
    - Dataset and task type
    - UI configuration for the web interface
    - Property definitions and interpretations

=== "Updating the Registry"
    ```python
    from src.helper import update_model_registry
    
    # After training a model, update the registry
    update_model_registry()
    ```

The model registry:

- Tracks all available models in one place
- Configures how models appear in the web interface
- Defines what properties each model predicts
- Groups models by category for easier selection
