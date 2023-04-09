from fastapi import FastAPI, HTTPException, Request
import torch
from src.model_utils import load_model
from src.predict import predict_molecule
from pydantic import BaseModel, Field
import os
import json
from typing import Dict, List, Any, Optional, Union
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import cirpy
from fastapi.routing import APIRouter
from contextlib import asynccontextmanager

# Global dictionary to store loaded models
loaded_models = {}
model_registry = {}

# Load model registry
def load_registry():
    try:
        with open('model_registry.json', 'r') as f:
            registry = json.load(f)
        return registry
    except Exception as e:
        print(f"Error loading model registry: {str(e)}")
        return {"models": []}

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models from registry
    global loaded_models, model_registry
    
    try:
        # Load model registry
        registry_data = load_registry()
        model_registry = registry_data
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load enabled models
        for model_config in registry_data.get('models', []):
            if model_config.get('enabled', False):
                model_id = model_config['id']
                model_file = model_config['model_file']
                
                try:
                    if os.path.exists(model_file):
                        model, model_info, _ = load_model(model_file, device)
                        loaded_models[model_id] = {
                            'model': model,
                            'config': model_config,
                            'device': device
                        }
                        print(f"Model '{model_id}' loaded successfully from {model_file}")
                    else:
                        print(f"Warning: Model file {model_file} not found for model '{model_id}'")
                except Exception as e:
                    print(f"Error loading model '{model_id}': {str(e)}")
        
        if not loaded_models:
            print("Warning: No models were loaded successfully")
            
    except Exception as e:
        print(f"Error during startup: {str(e)}")
    
    yield  # This is where FastAPI runs
    
    # Shutdown: Cleanup if needed
    loaded_models.clear()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Molecular Property Prediction",
    description="API for predicting molecular properties from SMILES strings or compound names",
    version="1.0.0",
    lifespan=lifespan
)

# Create API router with prefix
router = APIRouter(prefix="/molecule-net")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Define unified input model
class MoleculeQuery(BaseModel):
    query: str
    model_ids: Optional[List[str]] = Field(None, description="List of model IDs to use for prediction. If not provided, all available models will be used.")
    
# Define property prediction result
class PropertyPrediction(BaseModel):
    property_id: str
    property_name: str
    value: float
    unit: str
    
# Define model prediction result
class ModelPredictionResult(BaseModel):
    model_id: str
    model_name: str
    properties: List[PropertyPrediction]
    
# Define overall prediction response
class PredictionResponse(BaseModel):
    smiles: str
    input_query: str
    predictions: List[ModelPredictionResult]

# Main UI route - both at root and at /molecule-net
@app.get("/", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Get available models and their configurations
@router.get("/models", response_class=JSONResponse)
async def get_models():
    return model_registry

# Unified prediction endpoint
@router.post("/predict")
async def predict_molecule_properties(molecule: MoleculeQuery):
    global loaded_models
    
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models are loaded. Please check the server configuration.")
    
    try:
        query = molecule.query.strip()
        
        # Determine if the input is already a valid SMILES or needs resolution
        # Try to resolve with cirpy regardless, since even a valid SMILES will return itself
        resolved_smiles = cirpy.resolve(query, "smiles")
        
        if not resolved_smiles:
            raise HTTPException(status_code=422, detail=f"Could not resolve '{query}' to a valid SMILES string")
        
        # Determine which models to use
        models_to_use = {}
        if molecule.model_ids:
            # Use only the specified models
            for model_id in molecule.model_ids:
                if model_id in loaded_models:
                    models_to_use[model_id] = loaded_models[model_id]
                else:
                    # Skip unavailable models but don't fail
                    continue
        else:
            # Use all available models
            models_to_use = loaded_models
        
        if not models_to_use:
            raise HTTPException(status_code=400, detail="No valid models specified for prediction")
        
        # Prepare response
        all_predictions = []
        
        # Process each model
        for model_id, model_data in models_to_use.items():
            model = model_data['model']
            config = model_data['config']
            device = model_data['device']
            
            # Make prediction
            prediction_value = predict_molecule(model, resolved_smiles, device)
            
            # Extract property predictions for this model
            property_predictions = []
            
            # For each property defined in the model config, extract the prediction
            for prop in config['properties']:
                # For now, we only support single property models
                # In a more complex implementation, we'd handle multiple outputs from a single model
                property_predictions.append(
                    PropertyPrediction(
                        property_id=prop['id'],
                        property_name=prop['name'],
                        value=float(prediction_value),
                        unit=prop['unit']
                    )
                )
            
            # Add model prediction to the response
            all_predictions.append(
                ModelPredictionResult(
                    model_id=model_id,
                    model_name=config['name'],
                    properties=property_predictions
                )
            )
        
        # Return aggregated results
        return PredictionResponse(
            smiles=resolved_smiles,
            input_query=query,
            predictions=all_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing input: {str(e)}")

# Legacy endpoint for backward compatibility (can be removed later)
@router.post("/predict-legacy", response_model=None, include_in_schema=False)
async def legacy_predict(molecule: MoleculeQuery):
    """Legacy endpoint for backward compatibility with v1 API"""
    # Get the solubility model result first
    response = await predict_molecule_properties(molecule)
    
    # Find the solubility prediction
    solubility_value = None
    for model_result in response.predictions:
        for prop in model_result.properties:
            if prop.property_id == "solubility":
                solubility_value = prop.value
                break
        if solubility_value:
            break
    
    if not solubility_value:
        raise HTTPException(status_code=404, detail="Solubility prediction not available")
    
    # Return in the old format
    return {
        "smiles": response.smiles,
        "input_query": response.input_query,
        "predicted_solubility": solubility_value,
        "units": "log(mol/L)"
    }

# Include the router in the main app
app.include_router(router)

# Legacy root endpoint for backward compatibility
@app.post("/predict", include_in_schema=False)
async def legacy_root_predict(molecule: MoleculeQuery):
    return await legacy_predict(molecule)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 