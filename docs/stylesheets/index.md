# Molecular Property Prediction

<div class="grid">
  <div class="card">
    <strong>Powerful Prediction</strong>
    <p>Accurately predict molecular properties including solubility, toxicity, and binding affinity from molecular structures.</p>
  </div>

  <div class="card">
    <strong>Graph Neural Networks</strong>
    <p>State-of-the-art graph neural networks for capturing complex molecular relationships and structural information.</p>
  </div>

  <div class="card">
    <strong>Comprehensive Evaluation</strong>
    <p>Detailed metrics and visualizations for model performance assessment with RMSE, R², and more.</p>
  </div>

  <div class="card">
    <strong>Versatile Dataset Support</strong>
    <p>Support for multiple molecular datasets including ESOL, FreeSolv, Lipophilicity, HIV, BACE, BBBP, SIDER, and ClinTox.</p>
  </div>
</div>

## Project Overview

The Molecular Property Prediction framework is designed to help data scientists, researchers, and business professionals predict various properties of chemical compounds based on their molecular structures. This project leverages graph neural networks to capture the complex relationships between atoms and bonds in molecules.

Our framework supports a wide range of molecular property prediction tasks:

- **ESOL**: Predicting aqueous solubility
- **FreeSolv**: Hydration free energy prediction
- **Lipophilicity**: Octanol/water distribution coefficient prediction
- **HIV**: HIV replication inhibition prediction
- **BACE**: β-secretase inhibition prediction
- **BBBP**: Blood-brain barrier penetration prediction
- **SIDER**: Side effect prediction
- **ClinTox**: Clinical toxicity prediction

!!! info "Key Features"

    - Flexible data loading from CSV files with SMILES strings
    - Automated molecular graph construction
    - Customizable graph neural network architectures (GCN, GAT, etc.)
    - Built-in model evaluation and visualization tools
    - Efficient model saving and loading
    - Easy-to-use API for both training and prediction

## Quick Start

=== "Installation"

    ```bash
    # Clone the repository
    git clone https://github.com/yourusername/molecular-property-prediction.git
    cd molecular-property-prediction
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Or using pip directly
    pip install cirpy deepchem mango mlflow pandas plotly rdkit seaborn torch torch-geometric
    ```

=== "Basic Example"

    ```python
    from src.data_loader import MoleculeDataset
    from src.data_preprocessor import Preprocessor
    from src.data_splitter import DataSplitter, ShuffleSplit
    from src.model_architecture import MoleculeNetRegressor
    from src.train import setup_training
    from src.predict import predict_molecule
    import torch
    
    # Load dataset
    dataset = MoleculeDataset(root='datasets/processed', 
                             name='ESOL',
                             filepath='datasets/csv_files/delaney-processed.csv',
                             smiles_colname='smiles',
                             label_colname='ESOL predicted log solubility in mols per litre')
    
    # Preprocess data
    preprocessor = Preprocessor(dataset, task_type='regression')
    processed_dataset = preprocessor.preprocess()
    
    # Split data
    splitter = DataSplitter()
    splitter.set_strategy(ShuffleSplit())
    train_loader, val_loader, test_loader = splitter.split_data(processed_dataset, batch_size=32)
    
    # Initialize and train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoleculeNetRegressor(num_features=processed_dataset[0].num_features, hidden_dim=64, layer_type='gcn')
    model = model.to(device)
    
    # Train model
    model, optimizer, history, best_metrics = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Make prediction for a new molecule
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    prediction = predict_molecule(model, smiles, device)
    print(f"Predicted solubility: {prediction:.4f} log(mol/L)")
    ```

=== "Framework Architecture"

    ```mermaid
    graph TD
        A[Data Loading] --> B[Data Preprocessing]
        B --> C[Model Training]
        C --> D[Model Evaluation]
        D --> E[Prediction]
        
        F[SMILES Encoding] --> B
        G[Molecular Graph Construction] --> B
        H[Feature Extraction] --> B
        
        C --> I[Model Saving]
        I --> J[Model Loading]
        J --> E
    ```

## Project Components

The framework consists of several key components designed to work together seamlessly:

1. **Data Loading**: The `MoleculeDataset` class handles loading molecular data from CSV files containing SMILES strings and property values.

2. **Data Preprocessing**: The `Preprocessor` class converts SMILES strings into molecular graphs with appropriate features, handling the complex task of representing molecules in a machine-learning-ready format.

3. **Model Architecture**: The `MoleculeNetRegressor` class implements graph neural networks specifically designed for molecular property prediction.

4. **Training**: The training module provides tools for model training with early stopping, learning rate scheduling, and comprehensive history tracking.

5. **Evaluation**: Evaluation tools calculate important metrics like RMSE and R² to assess model performance.

6. **Prediction**: The prediction module allows making predictions for new molecules, supporting both individual SMILES strings and batches.

## Documentation Structure

- **Installation**: Detailed setup instructions
- **User Guide**: Step-by-step guides for each component
- **API Reference**: Complete API documentation
- **Examples**: Practical examples and use cases

[Explore the Documentation](documentation.md){ .md-button .md-button--primary } 