# Molecular Property Prediction

<div class="grid">
  <div class="card">
    <strong>Predict Properties</strong>
    <p>Predict molecular properties like solubility, toxicity, and binding affinity from molecular structures.</p>
  </div>

  <div class="card">
    <strong>Graph Neural Networks</strong>
    <p>Uses graph neural networks to understand how atoms and bonds relate to each other in molecules.</p>
  </div>

  <div class="card">
    <strong>Model Evaluation</strong>
    <p>See how well your models perform with metrics like RMSE, R², accuracy, and F1 scores.</p>
  </div>

  <div class="card">
    <strong>Multiple Datasets</strong>
    <p>Works with ESOL, FreeSolv, Lipophilicity, HIV, BACE, BBBP, SIDER, and ClinTox datasets.</p>
  </div>
</div>

## Project Overview

This project helps you predict properties of chemical compounds by looking at their molecular structures. It uses graph neural networks to figure out how atoms and bonds connect and what that means for the molecule's behavior.

You can work with different types of prediction tasks:

- **ESOL**: How well something dissolves in water
- **FreeSolv**: Hydration free energy
- **Lipophilicity**: How a molecule distributes between oil and water
- **HIV**: Whether something stops HIV from replicating
- **BACE**: Whether something inhibits beta secretase
- **BBBP**: Whether something can cross the blood brain barrier
- **SIDER**: What side effects something might have
- **ClinTox**: Whether something is toxic in clinical settings

!!! info "What You Can Do"

    - Load data from CSV files with SMILES strings
    - Turn molecules into graphs automatically
    - Use different graph neural network types (GCN, GAT, etc.)
    - Evaluate models and see how they perform
    - Save and load trained models
    - Acces models through a web interface
    - Use a web interface to make predictions
    - Train models for both regression and classification tasks

## Quick Start

=== "Installation"

    ```bash
    # Clone the repository
    git clone https://github.com/saisrinivas-samoju/MoleculeNet.git
    cd MoleculeNet
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Or install directly
    pip install cirpy deepchem mango mlflow pandas plotly rdkit seaborn torch torch-geometric fastapi uvicorn
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
        device=device,
        task_type='regression'
    )
    
    # Make prediction for a new molecule
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    prediction = predict_molecule(model, smiles, device, task_type='regression')
    print(f"Predicted solubility: {prediction:.4f} log(mol/L)")
    ```

=== "Web Interface"

    ```bash
    # Start the web server
    uvicorn app:app --reload
    
    # Then open http://localhost:8000 in your browser
    # You can enter SMILES strings or compound names to get predictions
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
        I --> J[Model Registry]
        J --> K[Web API]
        K --> E
    ```

## Main Parts

The project has a few key pieces that work together:

1. **Data Loading**: The `MoleculeDataset` class loads molecular data from CSV files. Just point it at a file with SMILES strings and property values.

2. **Data Preprocessing**: The `Preprocessor` class turns SMILES strings into molecular graphs. It figures out which atoms connect to which and what features matter.

3. **Model Architecture**: You can use `MoleculeNetRegressor` for regression tasks or `MoleculeNetClassifier` for classification. Both support different graph layer types.

4. **Training**: The training code handles early stopping, learning rate scheduling, and tracks everything so you can see how training went.

5. **Evaluation**: Check model performance with metrics. For regression you get RMSE, R², and correlation. For classification you get accuracy, precision, recall, F1, and ROC-AUC.

6. **Prediction**: Make predictions from SMILES strings. Works for single molecules or batches.

7. **Web Interface**: A FastAPI web app lets you make predictions through a browser. It loads models from a registry and can handle multiple models at once.

8. **Model Registry**: Keep track of all your trained models in one place. The web interface uses this to know which models are available.

## Documentation Structure

- **Installation**: How to set things up
- **User Guide**: How to use each part
- **API Reference**: What the functions do
- **Examples**: Real examples you can try

[Explore the Documentation](documentation.md){ .md-button .md-button--primary }
