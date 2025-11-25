"""
FastAPI Backend for Toxicity Prediction Web Application
Research-Grade Application with GCN and Baseline Models
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import json

# RDKit for molecular validation and processing
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import base64
from io import BytesIO

# Try to import GCN model loader (optional if torch/dgl not available)
try:
    from gcn_model_loader import gcn_loader
    GCN_AVAILABLE = True
    print("✓ GCN models loaded successfully")
except ImportError as e:
    GCN_AVAILABLE = False
    print(f"⚠ GCN models not available: {e}")
    print("  Backend will run with mock predictions")

# Try to import Baseline model loader
try:
    from baseline_model_loader import baseline_loader
    BASELINE_AVAILABLE = baseline_loader is not None
    if BASELINE_AVAILABLE:
        print("✓ Baseline models loaded successfully")
        print(f"  Trained toxicities: {', '.join(baseline_loader.trained_toxicities)}")
    else:
        print("⚠ Baseline models not available")
except ImportError as e:
    BASELINE_AVAILABLE = False
    baseline_loader = None
    print(f"⚠ Baseline models not available: {e}")

app = FastAPI(
    title="Toxicity Prediction API",
    description="Research-grade toxicity prediction using GCN and baseline ML models",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class SMILESInput(BaseModel):
    """Single SMILES input model"""
    smiles: str
    compound_id: Optional[str] = None
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """Validate SMILES using RDKit"""
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        
        try:
            mol = Chem.MolFromSmiles(v.strip())
            if mol is None:
                raise ValueError(f"Invalid SMILES string. Please check the format and try again.")
        except Exception as e:
            raise ValueError(f"Invalid SMILES string: {str(e)}")
        
        return v.strip()


class BatchSMILESInput(BaseModel):
    """Batch SMILES input model"""
    smiles_list: List[str]
    compound_ids: Optional[List[str]] = None
    
    @validator('smiles_list')
    def validate_smiles_list(cls, v):
        """Validate list of SMILES"""
        if not v or len(v) == 0:
            raise ValueError("SMILES list cannot be empty")
        
        if len(v) > 100:
            raise ValueError("Maximum 100 compounds per batch")
        
        validated = []
        invalid = []
        
        for i, smiles in enumerate(v):
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                invalid.append({"index": i, "smiles": smiles, "error": "Invalid SMILES"})
            else:
                validated.append(smiles.strip())
        
        if invalid:
            raise ValueError(f"Invalid SMILES found: {invalid}")
        
        return validated


class PredictionRequest(BaseModel):
    """Prediction request model"""
    smiles: str
    endpoints: List[str]
    compare_baseline: bool = False
    user_id: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    smiles_list: List[str]
    endpoints: List[str]
    compare_baseline: bool = False
    user_id: Optional[str] = None


# ==================== Configuration ====================

# Available toxicity endpoints
TOXICITY_ENDPOINTS = [
    {"id": "NR-AhR", "name": "Aryl hydrocarbon Receptor", "category": "Nuclear Receptor"},
    {"id": "NR-AR", "name": "Androgen Receptor", "category": "Nuclear Receptor"},
    {"id": "NR-AR-LBD", "name": "Androgen Receptor Ligand Binding Domain", "category": "Nuclear Receptor"},
    {"id": "NR-Aromatase", "name": "Aromatase", "category": "Nuclear Receptor"},
    {"id": "NR-ER", "name": "Estrogen Receptor", "category": "Nuclear Receptor"},
    {"id": "NR-ER-LBD", "name": "Estrogen Receptor Ligand Binding Domain", "category": "Nuclear Receptor"},
    {"id": "NR-PPAR-gamma", "name": "Peroxisome Proliferator-Activated Receptor Gamma", "category": "Nuclear Receptor"},
    {"id": "SR-ARE", "name": "Antioxidant Response Element", "category": "Stress Response"},
    {"id": "SR-ATAD5", "name": "ATPase Family AAA Domain-Containing Protein 5", "category": "Stress Response"},
    {"id": "SR-HSE", "name": "Heat Shock Factor Response Element", "category": "Stress Response"},
    {"id": "SR-MMP", "name": "Mitochondrial Membrane Potential", "category": "Stress Response"},
    {"id": "SR-p53", "name": "Tumor Suppressor p53", "category": "Stress Response"}
]

# Available models
AVAILABLE_MODELS = ["GCN", "KNN", "NN", "RF", "SVM", "XGBoost"]

# Endpoint presets
ENDPOINT_PRESETS = {
    "all": [ep["id"] for ep in TOXICITY_ENDPOINTS],
    "nuclear_receptor": [ep["id"] for ep in TOXICITY_ENDPOINTS if ep["category"] == "Nuclear Receptor"],
    "stress_response": [ep["id"] for ep in TOXICITY_ENDPOINTS if ep["category"] == "Stress Response"],
    "environmental": ["NR-AhR", "NR-ER", "NR-AR", "SR-ARE"],  # Common environmental toxicants
    "endocrine": ["NR-AR", "NR-ER", "NR-Aromatase", "NR-PPAR-gamma"]  # Endocrine disruption
}


# ==================== Helper Functions ====================

def validate_smiles(smiles: str) -> Dict[str, Any]:
    """
    Validate SMILES and return molecule info
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with validation status and info
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                "valid": False,
                "smiles": smiles,
                "error": "Invalid SMILES string. Please check the format and try again."
            }
        
        # Calculate basic molecular properties
        try:
            properties = {
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
                "tpsa": round(Descriptors.TPSA(mol), 2),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "num_hba": Descriptors.NumHAcceptors(mol),
                "num_hbd": Descriptors.NumHDonors(mol),
                "num_h_donors": Descriptors.NumHDonors(mol),
                "num_h_acceptors": Descriptors.NumHAcceptors(mol),
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "num_heavy_atoms": mol.GetNumHeavyAtoms()
            }
            
            return {
                "valid": True,
                "smiles": smiles,
                "canonical_smiles": Chem.MolToSmiles(mol),
                "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "properties": properties
            }
        except Exception as e:
            return {
                "valid": False,
                "smiles": smiles,
                "error": f"Error calculating molecular properties: {str(e)}"
            }
    except Exception as e:
        return {
            "valid": False,
            "smiles": smiles,
            "error": f"Error validating SMILES: {str(e)}"
        }


def generate_molecule_image(smiles: str, size: tuple = (300, 300)) -> str:
    """
    Generate molecule image as base64 string
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        
    Returns:
        Base64 encoded PNG image
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        img = Draw.MolToImage(mol, size=size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def get_endpoint_info(endpoint_id: str) -> Dict[str, Any]:
    """Get endpoint information by ID"""
    for ep in TOXICITY_ENDPOINTS:
        if ep["id"] == endpoint_id:
            return ep
    return {"id": endpoint_id, "name": endpoint_id, "category": "Unknown"}


def mock_gcn_prediction(smiles: str, endpoint: str) -> Dict[str, Any]:
    """
    Mock GCN prediction (fallback when torch/dgl not available)
    """
    import random
    random.seed(hash(smiles + endpoint) % 10000)
    
    prediction = random.choice([0, 1])
    probability = random.uniform(0.5, 0.99) if prediction == 1 else random.uniform(0.01, 0.5)
    
    endpoint_info = get_endpoint_info(endpoint)
    
    return {
        "model": "GCN",
        "endpoint": endpoint,
        "endpoint_name": endpoint_info["name"],
        "category": endpoint_info["category"],
        "prediction": "Toxic" if prediction == 1 else "Non-toxic",
        "prediction_value": prediction,
        "probability": round(probability, 4),
        "confidence": round(probability, 4),
        "confidence_level": "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low",
        "is_mock": True
    }


def get_baseline_prediction(smiles: str, endpoint: str, model_type: str = 'RF') -> Optional[Dict[str, Any]]:
    """
    Get baseline model prediction for a SMILES string and endpoint.
    Returns None if model is not trained for this endpoint.
    
    Args:
        smiles: SMILES string
        endpoint: Toxicity endpoint ID
        model_type: Model type (default: RF)
        
    Returns:
        Prediction result dictionary or None if not available
    """
    if not BASELINE_AVAILABLE or baseline_loader is None:
        return None
    
    # Check if this toxicity has trained models
    if not baseline_loader.is_toxicity_trained(endpoint):
        return None
    
    # Get endpoint info
    endpoint_info = get_endpoint_info(endpoint)
    
    # Get prediction
    try:
        pred_result = baseline_loader.predict(smiles, endpoint, model_type)
        
        if pred_result is None or 'error' in pred_result:
            return None
        
        return {
            "model": model_type,
            "endpoint": endpoint,
            "endpoint_name": endpoint_info["name"],
            "category": endpoint_info["category"],
            "prediction": pred_result['label'],
            "prediction_value": pred_result['prediction'],
            "probability": round(pred_result['probability'], 4),
            "confidence": round(pred_result['confidence'], 4),
            "confidence_level": "High" if pred_result['confidence'] > 0.6 
                              else "Medium" if pred_result['confidence'] > 0.3 
                              else "Low",
            "is_trained": True
        }
    except Exception as e:
        print(f"Error getting baseline prediction for {endpoint}: {e}")
        return None


def get_all_baseline_predictions(smiles: str, endpoint: str) -> Dict[str, Any]:
    """
    Get predictions from all available baseline models for an endpoint.
    Returns empty dict if endpoint has no trained models.
    
    Args:
        smiles: SMILES string
        endpoint: Toxicity endpoint ID
        
    Returns:
        Dictionary with predictions from all model types
    """
    if not BASELINE_AVAILABLE or baseline_loader is None:
        return {}
    
    if not baseline_loader.is_toxicity_trained(endpoint):
        return {}
    
    results = {}
    endpoint_info = get_endpoint_info(endpoint)
    
    for model_type in ['KNN', 'NN', 'RF', 'SVM', 'XGBoost']:
        pred = baseline_loader.predict(smiles, endpoint, model_type)
        if pred and 'error' not in pred:
            results[model_type] = {
                "model": model_type,
                "endpoint": endpoint,
                "endpoint_name": endpoint_info["name"],
                "category": endpoint_info["category"],
                "prediction": pred['label'],
                "prediction_value": pred['prediction'],
                "probability": round(pred['probability'], 4),
                "confidence": round(pred['confidence'], 4),
                "confidence_level": "High" if pred['confidence'] > 0.6 
                                  else "Medium" if pred['confidence'] > 0.3 
                                  else "Low",
                "is_trained": True
            }
    
    return results


def get_gcn_prediction(smiles: str, endpoint: str) -> Dict[str, Any]:
    """
    Get GCN model prediction for a SMILES string and endpoint.
    Uses trained model if available, otherwise falls back to mock.
    
    Args:
        smiles: SMILES string
        endpoint: Toxicity endpoint ID
        
    Returns:
        Prediction result dictionary
    """
    # Get endpoint info
    endpoint_info = get_endpoint_info(endpoint)
    
    # Try to get real prediction from trained model if available
    if GCN_AVAILABLE:
        try:
            pred_result = gcn_loader.predict(smiles, endpoint)
            
            if pred_result is not None:
                # Return formatted prediction from trained model
                return {
                    "model": "GCN",
                    "endpoint": endpoint,
                    "endpoint_name": endpoint_info["name"],
                    "category": endpoint_info["category"],
                    "prediction": pred_result['label'],
                    "prediction_value": pred_result['prediction'],
                    "probability": round(pred_result['probability'], 4),
                    "confidence": round(pred_result['confidence'], 4),
                    "confidence_level": "High" if pred_result['confidence'] > 0.8 
                                      else "Medium" if pred_result['confidence'] > 0.6 
                                      else "Low",
                    "is_mock": False
                }
        except Exception as e:
            print(f"Error getting GCN prediction: {e}")
    
    # Fall back to mock prediction
    return mock_gcn_prediction(smiles, endpoint)


# ==================== API Routes ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Toxicity Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "validate": "/api/validate",
            "predict": "/api/predict",
            "batch_predict": "/api/batch-predict",
            "endpoints": "/api/endpoints",
            "models": "/api/models",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    baseline_loaded = BASELINE_AVAILABLE and baseline_loader is not None
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gcn_available": GCN_AVAILABLE,
        "baseline_available": baseline_loaded,
        "baseline_trained_toxicities": baseline_loader.trained_toxicities if baseline_loaded else [],
        "available_endpoints": len(TOXICITY_ENDPOINTS)
    }


@app.get("/api/endpoints")
async def get_endpoints():
    """Get available toxicity endpoints"""
    return {
        "endpoints": TOXICITY_ENDPOINTS,
        "presets": {
            "all": {"name": "All Toxicities", "count": len(ENDPOINT_PRESETS["all"])},
            "nuclear_receptor": {"name": "Nuclear Receptor Panel", "count": len(ENDPOINT_PRESETS["nuclear_receptor"])},
            "stress_response": {"name": "Stress Response Panel", "count": len(ENDPOINT_PRESETS["stress_response"])},
            "environmental": {"name": "Environmental Toxicants", "count": len(ENDPOINT_PRESETS["environmental"])},
            "endocrine": {"name": "Endocrine Disruption", "count": len(ENDPOINT_PRESETS["endocrine"])}
        }
    }


@app.get("/api/models")
async def get_models():
    """Get available prediction models"""
    # Check which baseline models are actually loaded
    baseline_loaded = BASELINE_AVAILABLE and baseline_loader is not None
    trained_toxicities = baseline_loader.trained_toxicities if baseline_loaded else []
    
    return {
        "models": [
            {"id": "GCN", "name": "Graph Convolutional Network", "type": "Deep Learning", "loaded": GCN_AVAILABLE},
            {"id": "KNN", "name": "K-Nearest Neighbors", "type": "Baseline", "loaded": baseline_loaded},
            {"id": "NN", "name": "Neural Network", "type": "Baseline", "loaded": baseline_loaded},
            {"id": "RF", "name": "Random Forest", "type": "Baseline", "loaded": baseline_loaded},
            {"id": "SVM", "name": "Support Vector Machine", "type": "Baseline", "loaded": baseline_loaded},
            {"id": "XGBoost", "name": "XGBoost", "type": "Baseline", "loaded": baseline_loaded}
        ],
        "baseline_info": {
            "available": baseline_loaded,
            "trained_toxicities": trained_toxicities,
            "trained_count": len(trained_toxicities),
            "total_toxicities": len(TOXICITY_ENDPOINTS)
        }
    }


@app.post("/api/validate")
async def validate_smiles_endpoint(input_data: SMILESInput):
    """
    Validate a single SMILES string
    
    Args:
        input_data: SMILESInput model
        
    Returns:
        Validation result with molecular properties
    """
    result = validate_smiles(input_data.smiles)
    
    if result["valid"]:
        # Generate molecule image
        result["image"] = generate_molecule_image(input_data.smiles)
        
        if input_data.compound_id:
            result["compound_id"] = input_data.compound_id
    
    return result


@app.post("/api/validate-batch")
async def validate_batch_smiles(input_data: BatchSMILESInput):
    """
    Validate multiple SMILES strings
    
    Args:
        input_data: BatchSMILESInput model
        
    Returns:
        List of validation results
    """
    results = []
    
    for i, smiles in enumerate(input_data.smiles_list):
        result = validate_smiles(smiles)
        
        if input_data.compound_ids and i < len(input_data.compound_ids):
            result["compound_id"] = input_data.compound_ids[i]
        else:
            result["compound_id"] = f"Compound_{i+1}"
        
        results.append(result)
    
    return {
        "total": len(results),
        "valid": sum(1 for r in results if r["valid"]),
        "invalid": sum(1 for r in results if not r["valid"]),
        "results": results
    }


@app.post("/api/predict")
async def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity for a single compound
    
    Args:
        request: PredictionRequest model
        
    Returns:
        Prediction results
    """
    # Validate SMILES
    validation = validate_smiles(request.smiles)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])
    
    # Validate endpoints
    valid_endpoint_ids = [ep["id"] for ep in TOXICITY_ENDPOINTS]
    invalid_endpoints = [ep for ep in request.endpoints if ep not in valid_endpoint_ids]
    if invalid_endpoints:
        raise HTTPException(status_code=400, detail=f"Invalid endpoints: {invalid_endpoints}")
    
    # Generate GCN predictions using trained models
    gcn_predictions = []
    for endpoint in request.endpoints:
        pred = get_gcn_prediction(request.smiles, endpoint)
        gcn_predictions.append(pred)
    
    # Baseline model predictions (if requested and available)
    baseline_predictions = None
    if request.compare_baseline:
        baseline_predictions = []
        for endpoint in request.endpoints:
            # Get all baseline model predictions for this endpoint
            baseline_preds = get_all_baseline_predictions(request.smiles, endpoint)
            
            if baseline_preds:
                # Has trained models - add predictions
                baseline_predictions.append({
                    "endpoint": endpoint,
                    "endpoint_name": get_endpoint_info(endpoint)["name"],
                    "models": baseline_preds,
                    "is_trained": True
                })
            else:
                # No trained models - will show as '-' in UI
                baseline_predictions.append({
                    "endpoint": endpoint,
                    "endpoint_name": get_endpoint_info(endpoint)["name"],
                    "models": {},
                    "is_trained": False
                })
    
    response_data = {
        "smiles": request.smiles,
        "canonical_smiles": validation["canonical_smiles"],
        "molecular_formula": validation["molecular_formula"],
        "properties": validation["properties"],
        "image": generate_molecule_image(request.smiles),
        "predictions": gcn_predictions,
        "baseline_predictions": baseline_predictions,
        "compare_baseline": request.compare_baseline,
        "timestamp": datetime.now().isoformat()
    }
    
    return response_data


@app.post("/api/batch-predict")
async def batch_predict_toxicity(request: BatchPredictionRequest):
    """
    Predict toxicity for multiple compounds
    
    Args:
        request: BatchPredictionRequest model
        
    Returns:
        Batch prediction results
    """
    # Validate all SMILES
    validated_smiles = []
    invalid_smiles = []
    
    for i, smiles in enumerate(request.smiles_list):
        validation = validate_smiles(smiles)
        if validation["valid"]:
            validated_smiles.append({
                "index": i,
                "smiles": smiles,
                "canonical_smiles": validation["canonical_smiles"],
                "properties": validation["properties"]
            })
        else:
            invalid_smiles.append({
                "index": i,
                "smiles": smiles,
                "error": validation["error"]
            })
    
    if len(validated_smiles) == 0:
        raise HTTPException(status_code=400, detail="No valid SMILES found")
    
    # Generate predictions for all valid compounds using trained models
    results = []
    
    for compound in validated_smiles:
        compound_predictions = []
        
        for endpoint in request.endpoints:
            pred = get_gcn_prediction(compound["smiles"], endpoint)
            compound_predictions.append(pred)
        
        # Add baseline predictions if requested
        baseline_preds = None
        if request.compare_baseline:
            baseline_preds = []
            for endpoint in request.endpoints:
                baseline_endpoint_preds = get_all_baseline_predictions(compound["smiles"], endpoint)
                
                if baseline_endpoint_preds:
                    baseline_preds.append({
                        "endpoint": endpoint,
                        "endpoint_name": get_endpoint_info(endpoint)["name"],
                        "models": baseline_endpoint_preds,
                        "is_trained": True
                    })
                else:
                    baseline_preds.append({
                        "endpoint": endpoint,
                        "endpoint_name": get_endpoint_info(endpoint)["name"],
                        "models": {},
                        "is_trained": False
                    })
        
        results.append({
            "smiles": compound["smiles"],
            "canonical_smiles": compound["canonical_smiles"],
            "properties": compound["properties"],
            "predictions": compound_predictions,
            "baseline_predictions": baseline_preds
        })
    
    return {
        "total_submitted": len(request.smiles_list),
        "valid_compounds": len(validated_smiles),
        "invalid_compounds": len(invalid_smiles),
        "invalid_details": invalid_smiles,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/endpoint-presets/{preset_name}")
async def get_endpoint_preset(preset_name: str):
    """
    Get endpoint IDs for a preset
    
    Args:
        preset_name: Preset name (all, nuclear_receptor, stress_response, etc.)
        
    Returns:
        List of endpoint IDs
    """
    if preset_name not in ENDPOINT_PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    return {
        "preset": preset_name,
        "endpoints": ENDPOINT_PRESETS[preset_name],
        "count": len(ENDPOINT_PRESETS[preset_name])
    }


# ==================== Future Endpoints (Placeholders) ====================

@app.post("/api/compare-models")
async def compare_models():
    """Compare predictions across different models (placeholder)"""
    return {"message": "Model comparison endpoint - to be implemented"}


@app.post("/api/visualize")
async def generate_visualizations():
    """Generate visualization data (placeholder)"""
    return {"message": "Visualization endpoint - to be implemented"}


@app.post("/api/explain-gcn")
async def explain_gcn_prediction():
    """Generate GCN explainability visualization (placeholder)"""
    return {"message": "GCN explainability endpoint - to be implemented"}


@app.get("/api/research-metrics")
async def get_research_metrics():
    """Get model performance metrics and results data from trained models - v2"""
    import os
    import csv
    import json
    from pathlib import Path
    
    try:
        # Base results directory
        results_dir = Path(__file__).parent.parent.parent / "results"
        
        # Read overall GCN summary
        overall_summary_path = results_dir / "overall_summary.csv"
        gcn_results = []
        
        if overall_summary_path.exists():
            with open(overall_summary_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gcn_results.append({
                        "toxicity": row['dataset'],
                        "train_samples": int(row['train_samples']),
                        "test_samples": int(row['test_samples']),
                        "test_accuracy": float(row['test_accuracy']),
                        "test_roc_auc": float(row['test_roc_auc']),
                        "test_precision": float(row['test_precision']),
                        "test_recall": float(row['test_recall']),
                        "test_f1": float(row['test_f1']),
                        "best_val_auc": float(row['best_val_auc']),
                        "best_epoch": int(row['best_epoch'])
                    })
        
        # Read detailed GCN results from individual toxicity directories
        gcn_detailed_results = {}
        toxicity_endpoints = [
            "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        
        for toxicity in toxicity_endpoints:
            toxicity_dir = results_dir / toxicity
            summary_path = toxicity_dir / "summary.csv"
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    reader = csv.DictReader(f)
                    details = {}
                    for row in reader:
                        details[row['attribute']] = row['value']
                    gcn_detailed_results[toxicity] = details
        
        # Read baseline model results from baseline_models directory
        baseline_results = {}
        baseline_dir = results_dir / "baseline_models"
        
        if baseline_dir.exists():
            for toxicity_dir in baseline_dir.iterdir():
                if toxicity_dir.is_dir():
                    toxicity = toxicity_dir.name
                    summary_path = toxicity_dir / "summary.csv"
                    
                    if summary_path.exists():
                        with open(summary_path, 'r') as f:
                            reader = csv.DictReader(f)
                            models = []
                            for row in reader:
                                models.append({
                                    "model": row['Model'],
                                    "cv_roc_auc": float(row['CV_ROC_AUC']),
                                    "test_roc_auc": float(row['Test_ROC_AUC']),
                                    "test_accuracy": float(row['Test_Accuracy']),
                                    "test_precision": float(row['Test_Precision']),
                                    "test_recall": float(row['Test_Recall']),
                                    "test_f1": float(row['Test_F1'])
                                })
                            baseline_results[toxicity] = models
        
        # Get ROC data for all available baseline toxicities
        roc_data = {}
        if baseline_dir.exists():
            for toxicity_dir in baseline_dir.iterdir():
                if toxicity_dir.is_dir():
                    toxicity = toxicity_dir.name
                    toxicity_roc = {}
                    
                    # Get baseline ROC data from JSON files
                    for model_type in ['RF', 'XGBoost', 'SVM', 'NN', 'KNN']:
                        result_file = toxicity_dir / f"{model_type}_results.json"
                        if result_file.exists():
                            try:
                                with open(result_file, 'r') as f:
                                    data = json.load(f)
                                    
                                    # Try different JSON structures
                                    probabilities = None
                                    labels = None
                                    
                                    # Structure 1: Direct test_probabilities and true_labels
                                    if 'test_probabilities' in data and 'true_labels' in data:
                                        probabilities = data['test_probabilities']
                                        labels = data['true_labels']
                                    # Structure 2: Nested in test_results
                                    elif 'test_results' in data:
                                        test_results = data['test_results']
                                        if 'probabilities' in test_results and 'true_labels' in test_results:
                                            probabilities = test_results['probabilities']
                                            labels = test_results['true_labels']
                                        # Try predictions instead of probabilities
                                        elif 'predictions' in test_results and 'true_labels' in test_results:
                                            probabilities = test_results['predictions']
                                            labels = test_results['true_labels']
                                    
                                    # Add to ROC data if we found both
                                    if probabilities is not None and labels is not None:
                                        toxicity_roc[model_type] = {
                                            'probabilities': probabilities,
                                            'labels': labels
                                        }
                            except Exception as e:
                                print(f"Warning: Could not load ROC data for {toxicity}/{model_type}: {e}")
                    
                    if toxicity_roc:
                        roc_data[toxicity] = toxicity_roc
        
        # Get list of all available toxicities (union of GCN and baseline)
        all_toxicities = list(set([r['toxicity'] for r in gcn_results] + list(baseline_results.keys())))
        all_toxicities.sort()
        
        return {
            "status": "success",
            "gcn_results": gcn_results,
            "gcn_detailed_results": gcn_detailed_results,
            "baseline_results": baseline_results,
            "roc_data": roc_data,
            "available_toxicities": list(baseline_results.keys()),
            "all_toxicities": all_toxicities,
            "data_source": "trained_models"
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error loading research metrics: {e}")
        print(error_details)
        return {
            "status": "error",
            "message": str(e),
            "error_details": error_details,
            "gcn_results": [],
            "baseline_results": {},
            "roc_data": {},
            "available_toxicities": [],
            "all_toxicities": []
        }


# ==================== Run Application ====================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
