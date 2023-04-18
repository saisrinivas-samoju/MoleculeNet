from fastapi import FastAPI, HTTPException, Request
import torch
from src.model_utils import load_model
from src.predict import predict_molecule
from src.visualize import visualize_3d_html
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
                        # Load checkpoint to extract task_type from model_info (as saved in main.py)
                        checkpoint = torch.load(model_file, map_location=device)
                        model_info = checkpoint.get('model_info', {})
                        task_type = model_info['task_type']  # Extract from saved model_info, no default
                        
                        # Now load the model with the correct task_type
                        model, model_info, _ = load_model(model_file, device, task_type)
                        loaded_models[model_id] = {
                            'model': model,
                            'config': model_config,
                            'device': device,
                            'task_type': task_type
                        }
                        print(f"Model '{model_id}' loaded successfully from {model_file} (task_type: {task_type})")
                    else:
                        print(f"Warning: Model file {model_file} not found for model '{model_id}'")
                except KeyError as e:
                    print(f"Error loading model '{model_id}': Missing 'task_type' in model_info. {str(e)}")
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

# Get 3D molecular visualization HTML
@router.get("/visualize-3d")
async def get_3d_visualization(smiles: str):
    """
    Generate 3D molecular visualization HTML for a given SMILES string.
    
    Parameters:
    -----------
    smiles : str
        SMILES string of the molecule to visualize
    
    Returns:
    --------
    JSONResponse
        JSON object with 'html' field containing the visualization HTML
    """
    try:
        html = visualize_3d_html(smiles)
        return JSONResponse(content={"html": html})
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error generating visualization: {str(e)}")

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
        try:
            resolved_smiles = cirpy.resolve(query, "smiles")
        except Exception as resolve_error:
            # Check if it's a network/connection error
            error_str = str(resolve_error).lower()
            if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'unreachable', 'dns']):
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not resolve '{query}' to SMILES: Internet connection required for compound name resolution. Please use SMILES string directly (e.g., 'CCO' for ethanol) or ensure internet is available."
                )
            # Other resolution errors
            raise HTTPException(
                status_code=422,
                detail=f"Could not resolve '{query}' to a valid SMILES string: {str(resolve_error)}"
            )
        
        # Check if resolution failed or returned empty string
        if not resolved_smiles or not resolved_smiles.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Could not resolve '{query}' to a valid SMILES string. If using a compound name, internet connection is required. Alternatively, use SMILES string directly (e.g., 'CCO' for ethanol, 'CC(=O)OC1=CC=CC=C1C(=O)O' for aspirin)."
            )
        
        # Ensure resolved SMILES is not empty
        resolved_smiles = resolved_smiles.strip()
        
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
            try:
                model = model_data['model']
                config = model_data['config']
                device = model_data['device']
                task_type = model_data.get('task_type', 'regression')  # Get task_type from loaded model data
                
                print(f"[DEBUG] Processing model '{model_id}' (task_type: {task_type})")
                
                # Make prediction
                try:
                    prediction_result = predict_molecule(model, resolved_smiles, device, task_type)
                    # Safe formatting - handle None values
                    pred_repr = repr(prediction_result) if prediction_result is not None else "None"
                    print(f"[DEBUG] Model '{model_id}' prediction_result type: {type(prediction_result)}, value: {pred_repr}")
                except Exception as pred_error:
                    # Safely format error message to avoid format string issues
                    error_msg = str(pred_error).replace('{', '{{').replace('}', '}}')  # Escape braces in error messages
                    print(f"[ERROR] Exception in predict_molecule for model '{model_id}': {type(pred_error).__name__}: {error_msg}")
                    raise  # Re-raise to be caught by outer try-except
                
                # Handle None return values (prediction failed)
                if prediction_result is None:
                    # Skip this model and continue with others
                    print(f"[WARNING] Prediction failed for model '{model_id}': prediction_result is None. Skipping this model.")
                    continue
                
                # Extract the prediction value based on task type
                prediction_value = None
                if task_type == 'classification':
                    print(f"[DEBUG] Model '{model_id}': Processing classification result")
                    # Classification returns a tuple: (prediction, prob)
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        prediction_class, prediction_prob = prediction_result
                        # Safe formatting for debug - use repr to avoid format issues
                        class_repr = repr(prediction_class) if prediction_class is not None else "None"
                        prob_repr = repr(prediction_prob) if prediction_prob is not None else "None"
                        print(f"[DEBUG] Model '{model_id}': Unpacked tuple - class: {class_repr} (type: {type(prediction_class)}), prob: {prob_repr} (type: {type(prediction_prob)})")
                        # Check if prediction failed (both values are None)
                        if prediction_class is None or prediction_prob is None:
                            print(f"[WARNING] Prediction failed for model '{model_id}': class={prediction_class}, prob={prediction_prob}. Skipping this model.")
                            continue
                        # Use the probability as the value for display
                        try:
                            prediction_value = float(prediction_prob)
                            print(f"[DEBUG] Model '{model_id}': Converted probability to float: {prediction_value} (type: {type(prediction_value)})")
                        except (TypeError, ValueError) as e:
                            # Safe formatting - use repr to avoid format issues
                            prob_repr = repr(prediction_prob) if prediction_prob is not None else "None"
                            error_msg = str(e).replace('{', '{{').replace('}', '}}')
                            print(f"[ERROR] Could not convert prediction probability to float for model '{model_id}': {type(e).__name__}: {error_msg}. Value was: {prob_repr} (type: {type(prediction_prob)}). Skipping this model.")
                            continue
                    else:
                        print(f"[WARNING] Unexpected prediction format for classification model '{model_id}': expected tuple of length 2, got {type(prediction_result)} with length {len(prediction_result) if hasattr(prediction_result, '__len__') else 'N/A'}. Skipping this model.")
                        continue
                else:
                    print(f"[DEBUG] Model '{model_id}': Processing regression result")
                    # Regression returns a single value
                    # Additional safety check
                    if prediction_result is None:
                        print(f"[WARNING] Prediction failed for model '{model_id}': prediction_result is None. Skipping this model.")
                        continue
                    try:
                        prediction_value = float(prediction_result)
                        print(f"[DEBUG] Model '{model_id}': Converted regression result to float: {prediction_value} (type: {type(prediction_value)})")
                    except (TypeError, ValueError) as e:
                        # Safe formatting - use repr to avoid format issues
                        result_repr = repr(prediction_result) if prediction_result is not None else "None"
                        error_msg = str(e).replace('{', '{{').replace('}', '}}')
                        print(f"[ERROR] Could not convert prediction value to float for model '{model_id}': {type(e).__name__}: {error_msg}. Value was: {result_repr} (type: {type(prediction_result)}). Skipping this model.")
                        continue
                
                # Final safety check: ensure prediction_value is a valid float
                if prediction_value is None:
                    print(f"[ERROR] Prediction value is None for model '{model_id}' after conversion. This should not happen. Skipping this model.")
                    continue
                
                # Ensure prediction_value is a float (not int, not None)
                try:
                    prediction_value = float(prediction_value)
                    print(f"[DEBUG] Model '{model_id}': Final prediction_value: {prediction_value} (type: {type(prediction_value)}, is None: {prediction_value is None})")
                except (TypeError, ValueError) as e:
                    # Safe formatting - use repr to avoid format issues
                    value_repr = repr(prediction_value) if prediction_value is not None else "None"
                    error_msg = str(e).replace('{', '{{').replace('}', '}}')
                    print(f"[ERROR] Could not ensure prediction value is float for model '{model_id}': {type(e).__name__}: {error_msg}. Value was: {value_repr} (type: {type(prediction_value)}). Skipping this model.")
                    continue
                
                # Final validation before creating PropertyPrediction
                # Ensure prediction_value is definitely a float and not None
                if prediction_value is None:
                    print(f"[ERROR] Model '{model_id}': prediction_value is None right before PropertyPrediction creation. This should not happen. Skipping this model.")
                    continue
                
                if not isinstance(prediction_value, (int, float)):
                    print(f"[ERROR] Model '{model_id}': prediction_value is not a number: {type(prediction_value)}. Skipping this model.")
                    continue
                
                # Convert to float one more time to be absolutely sure
                try:
                    prediction_value = float(prediction_value)
                    if not isinstance(prediction_value, float) or prediction_value is None:
                        raise ValueError("Value is not a valid float")
                except (TypeError, ValueError) as e:
                    print(f"[ERROR] Model '{model_id}': Final float conversion failed: {type(e).__name__}: {e}. Skipping this model.")
                    continue
                
                # Extract property predictions for this model
                property_predictions = []
                
                # For each property defined in the model config, extract the prediction
                for prop in config['properties']:
                    # For now, we only support single property models
                    # In a more complex implementation, we'd handle multiple outputs from a single model
                    print(f"[DEBUG] Model '{model_id}': Creating PropertyPrediction for property '{prop['id']}' with value: {prediction_value} (type: {type(prediction_value)}, is None: {prediction_value is None})")
                    try:
                        property_pred = PropertyPrediction(
                            property_id=prop['id'],
                            property_name=prop['name'],
                            value=prediction_value,  # This is guaranteed to be a float at this point
                            unit=prop['unit']
                        )
                        property_predictions.append(property_pred)
                        print(f"[DEBUG] Model '{model_id}': Successfully created PropertyPrediction")
                    except Exception as prop_error:
                        # Safe formatting - escape braces in error messages and use repr for values
                        value_repr = repr(prediction_value) if prediction_value is not None else "None"
                        error_msg = str(prop_error).replace('{', '{{').replace('}', '}}')
                        print(f"[ERROR] Failed to create PropertyPrediction for model '{model_id}', property '{prop['id']}': {type(prop_error).__name__}: {error_msg}. Value was: {value_repr} (type: {type(prediction_value)}). Skipping this property.")
                        # Continue to next property instead of breaking the whole model
                        continue
                
                # Add model prediction to the response
                if property_predictions:
                    try:
                        model_result = ModelPredictionResult(
                            model_id=model_id,
                            model_name=config['name'],
                            properties=property_predictions
                        )
                        all_predictions.append(model_result)
                        print(f"[DEBUG] Model '{model_id}': Successfully added to predictions list")
                    except Exception as model_error:
                        # Safe formatting - escape braces in error messages
                        error_msg = str(model_error).replace('{', '{{').replace('}', '}}')
                        print(f"[ERROR] Failed to create ModelPredictionResult for model '{model_id}': {type(model_error).__name__}: {error_msg}. Skipping this model.")
                        continue
                else:
                    print(f"[WARNING] Model '{model_id}': No valid properties created. Skipping this model.")
                    continue
                    
            except Exception as e:
                # Log the error but continue with other models
                import traceback
                # Safe formatting - escape braces in error messages
                error_msg = str(e).replace('{', '{{').replace('}', '}}')
                print(f"[ERROR] Exception processing model '{model_id}': {type(e).__name__}: {error_msg}")
                # Format traceback safely
                try:
                    tb_str = traceback.format_exc()
                    tb_str_safe = tb_str.replace('{', '{{').replace('}', '}}')
                    print(f"[ERROR] Traceback: {tb_str_safe}")
                except Exception:
                    print(f"[ERROR] Could not format traceback for model '{model_id}'")
                # Skip this model and continue with others
                continue
        
        # Check if we have any successful predictions
        if not all_predictions:
            raise HTTPException(
                status_code=422,
                detail="All model predictions failed. Could not process the input SMILES string with any available model."
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