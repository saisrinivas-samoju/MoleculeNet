# Project Documentation

This documentation provides detailed information about the Molecular Property Prediction framework components and how to use them.

## Data Loading

The data loading component is responsible for importing molecular data from various sources, particularly CSV files containing SMILES strings and molecular properties.

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
    The framework supports several built-in datasets:
    
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

The `MoleculeDataset` class handles:

- Loading molecule data from CSV files
- Storing SMILES strings and associated property values
- Converting molecules to PyTorch Geometric Data objects
- Providing standard dataset functionality (indexing, length, etc.)

## Data Preprocessing

The preprocessing component transforms raw molecular data into a format suitable for graph neural networks.

=== "Preprocessing Steps"
    ```python
    from src.data_preprocessor import Preprocessor
    
    # Initialize preprocessor
    preprocessor = Preprocessor(dataset, task_type='regression')
    
    # Preprocess the dataset
    processed_dataset = preprocessor.preprocess()
    ```

=== "Feature Engineering"
    The preprocessor extracts comprehensive molecular features:
    
    - **Atom Features**: Atomic number, chirality, hybridization, aromaticity, etc.
    - **Bond Features**: Bond type, conjugation, ring status, etc.
    - **Molecular Graphs**: Nodes (atoms) and edges (bonds) with associated features

The preprocessing pipeline:

1. Takes the raw SMILES strings
2. Converts them to molecular graphs
3. Extracts relevant features for each atom and bond
4. Creates PyTorch Geometric Data objects

## Data Splitting

The data splitting component handles division of datasets into training, validation, and test sets.

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
    The framework supports multiple splitting strategies:
    
    - **ShuffleSplit**: Random splitting (default)
    - **ScaffoldSplit**: Split based on molecular scaffolds
    - **TemporalSplit**: Split based on temporal information (if available)

The data splitter ensures:

- Proper distribution of data across training, validation, and test sets
- Consistent batch sizes for efficient training
- Reproducibility through random seed setting

## Model Architecture

The model architecture component provides neural network models specialized for molecular property prediction.

=== "Model Initialization"
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

=== "Available Architectures"
    The framework supports several types of graph neural networks:
    
    - **GCN**: Graph Convolutional Network (default)
    - **GAT**: Graph Attention Network
    - **GraphSAGE**: Graph SAmple and aggreGatE

The `MoleculeNetRegressor` class:

- Implements a graph neural network for molecular property prediction
- Processes molecular graphs as input
- Supports different types of graph layer architectures
- Produces property predictions as output

## Model Training

The training component handles the process of training models on molecular data.

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
        patience=10
    )
    
    # Plot training progress
    plot_training_history(history)
    
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
        verbose=True  # Print progress
    )
    ```

The training module provides:

- Automatic early stopping to prevent overfitting
- Learning rate scheduling for better convergence
- Comprehensive training history tracking
- Visualization of training progress

## Model Evaluation

The evaluation component assesses model performance on test data.

=== "Basic Evaluation"
    ```python
    from src.evaluate import evaluate_model
    import matplotlib.pyplot as plt
    
    # Evaluate on test data
    test_metrics, test_predictions, test_actual = evaluate_model(model, test_loader, device)
    
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

The evaluation module calculates various metrics:

- **For Regression**: RMSE, MAE, R², Pearson correlation
- **For Classification**: Accuracy, precision, recall, F1-score, ROC-AUC

## Model Saving and Loading

The model utilities component allows saving and loading trained models.

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
    loaded_model, loaded_info, loaded_metrics = load_model('models/esol_gcn_full.pt', device)
    
    print(f"Loaded model info: {loaded_info}")
    print(f"Loaded model metrics: {loaded_metrics}")
    ```

The model utilities provide:

- Comprehensive model saving with metadata
- Convenient loading for later use or deployment
- Storage of model architecture, weights, and performance metrics

## Making Predictions

The prediction component allows making predictions for new molecules.

=== "Single Molecule Prediction"
    ```python
    from src.predict import predict_molecule
    
    # Predict property for a single molecule using SMILES
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    prediction = predict_molecule(model, smiles, device)
    
    print(f"Predicted solubility: {prediction:.4f} log(mol/L)")
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
    predictions, _ = predict_molecules(model, smiles_list, device)
    
    # Create results dataframe
    results = pd.DataFrame({
        'SMILES': smiles_list,
        'Predicted': [f"{pred:.4f}" for pred in predictions]
    })
    
    print(results)
    ```

The prediction module offers:

- Direct prediction from SMILES strings
- Batch predictions for multiple molecules
- Support for both raw SMILES and preprocessed molecular graphs 