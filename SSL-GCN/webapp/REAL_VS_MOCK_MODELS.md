# Using Real Trained Baseline Models in WebApp

## Current Situation

### What's Happening Now
The webapp backend is currently using **MOCK/SIMULATED predictions** instead of actual trained models:

```python
def mock_prediction(smiles: str, endpoint: str, model: str = "GCN") -> Dict[str, Any]:
    """
    Mock prediction function (replace with actual model inference later)
    """
    # TODO: Replace with actual model inference
    import random
    random.seed(hash(smiles + endpoint + model) % 10000)
    
    prediction = random.choice([0, 1])  # ← Random prediction!
    probability = random.uniform(0.5, 0.99) if prediction == 1 else random.uniform(0.01, 0.5)
    ...
```

### Why Mock Data?
- The webapp was built as a **prototype/demo** to showcase the UI and API structure
- Real model inference requires loading large model files and dependencies
- The mock function provides consistent "predictions" for testing (seeded random)

### What Trained Models Exist?

You have **actual trained baseline models** in `models/baseline_models/`:

```
models/baseline_models/
├── NR-AhR/
│   ├── KNN_model.pkl
│   ├── NN_model.pkl
│   ├── RF_model.pkl
│   ├── SVM_model.pkl
│   └── XGBoost_model.pkl
├── NR-AR/
│   ├── KNN_model.pkl
│   ├── NN_model.pkl
│   ├── RF_model.pkl
│   ├── SVM_model.pkl
│   └── XGBoost_model.pkl
└── NR-AR-LBD/
    └── KNN_model.pkl
```

**Trained for:** NR-AhR, NR-AR, NR-AR-LBD (partial)

**Not yet trained for:** NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

## Solution: Integrate Real Models

There are two approaches:

### Option 1: Hybrid Approach (Recommended)
- Use **real trained models** where available
- Fall back to **mock predictions** for untrained endpoints
- Show a disclaimer in UI when showing mock data

### Option 2: Train All Models First
- Train baseline models for all 12 toxicity endpoints
- Then integrate into webapp
- More time-consuming but provides complete real data

## Implementation Plan

### Step 1: Create Model Loader for WebApp

Create `webapp/backend/model_loader.py`:

```python
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Any, Optional
import sys

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))
from train_baseline_models import ECFP4Encoder


class BaselineModelLoader:
    """Load and use trained baseline models"""
    
    def __init__(self, models_dir='../../models/baseline_models'):
        self.models_dir = models_dir
        self.models = {}
        self.encoder = ECFP4Encoder(radius=2, n_bits=2048)
        self.loaded_endpoints = set()
        
    def load_endpoint_models(self, endpoint: str):
        """Load all baseline models for a specific endpoint"""
        if endpoint in self.loaded_endpoints:
            return
            
        endpoint_dir = os.path.join(self.models_dir, endpoint)
        if not os.path.exists(endpoint_dir):
            return
            
        self.models[endpoint] = {}
        model_types = ['KNN', 'NN', 'RF', 'SVM', 'XGBoost']
        
        for model_type in model_types:
            model_path = os.path.join(endpoint_dir, f"{model_type}_model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[endpoint][model_type] = pickle.load(f)
                    print(f"Loaded {endpoint} - {model_type}")
                except Exception as e:
                    print(f"Error loading {endpoint} - {model_type}: {e}")
        
        self.loaded_endpoints.add(endpoint)
    
    def predict(self, smiles: str, endpoint: str, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Make a prediction using a trained model.
        Returns None if model not available.
        """
        # Load models if not already loaded
        if endpoint not in self.loaded_endpoints:
            self.load_endpoint_models(endpoint)
        
        # Check if model exists
        if endpoint not in self.models or model_type not in self.models[endpoint]:
            return None
        
        # Encode SMILES
        fingerprint = self.encoder.smiles_to_ecfp4(smiles)
        if fingerprint is None:
            return None
        
        fingerprint = fingerprint.reshape(1, -1)
        
        # Get model
        model_dict = self.models[endpoint][model_type]
        model = model_dict['model']
        
        # Make prediction
        try:
            if model_type == 'SVM' and 'scaler' in model_dict:
                fingerprint_scaled = model_dict['scaler'].transform(fingerprint)
                pred = model.predict(fingerprint_scaled)[0]
                proba = model.predict_proba(fingerprint_scaled)[0, 1]
            else:
                pred = model.predict(fingerprint)[0]
                proba = model.predict_proba(fingerprint)[0, 1]
            
            return {
                'prediction': int(pred),
                'probability': float(proba),
                'label': 'Toxic' if pred == 1 else 'Non-toxic'
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None


# Global model loader instance
baseline_loader = BaselineModelLoader()
```

### Step 2: Update app.py to Use Real Models

Modify `webapp/backend/app.py`:

```python
from model_loader import baseline_loader

def get_baseline_prediction(smiles: str, endpoint: str, model_type: str) -> Dict[str, Any]:
    """
    Get baseline model prediction, falling back to mock if model not available.
    """
    # Try to get real prediction
    real_pred = baseline_loader.predict(smiles, endpoint, model_type)
    
    if real_pred:
        endpoint_info = get_endpoint_info(endpoint)
        return {
            "model": model_type,
            "endpoint": endpoint,
            "endpoint_name": endpoint_info["name"],
            "category": endpoint_info["category"],
            "prediction": real_pred['label'],
            "prediction_value": real_pred['prediction'],
            "probability": round(real_pred['probability'], 4),
            "confidence": round(real_pred['probability'], 4),
            "confidence_level": "High" if abs(real_pred['probability'] - 0.5) > 0.3 
                              else "Medium" if abs(real_pred['probability'] - 0.5) > 0.15 
                              else "Low",
            "is_real": True  # Flag to indicate real prediction
        }
    else:
        # Fall back to mock prediction
        pred = mock_prediction(smiles, endpoint, model_type)
        pred["is_real"] = False  # Flag to indicate mock prediction
        return pred
```

Then update the prediction endpoint:

```python
@app.post("/api/predict")
async def predict_toxicity(request: PredictionRequest):
    # ... validation code ...
    
    # Baseline model predictions (if requested)
    baseline_predictions = {}
    if request.compare_baseline:
        baseline_models = {
            "knn": "KNN",
            "neural_network": "NN", 
            "random_forest": "RF",
            "svm": "SVM",
            "xgboost": "XGBoost"
        }
        
        # Organize baseline predictions by endpoint
        for endpoint in request.endpoints:
            baseline_predictions[endpoint] = {}
            for model_key, model_name in baseline_models.items():
                # Use real model if available, otherwise mock
                pred = get_baseline_prediction(request.smiles, endpoint, model_name)
                baseline_predictions[endpoint][model_key] = pred["prediction"]
                baseline_predictions[endpoint][f"{model_key}_is_real"] = pred.get("is_real", False)
```

### Step 3: Update Frontend to Show Data Source

Modify `PredictionResults.jsx` to indicate which predictions are real vs mock:

```jsx
{baselinePred && (
  <td className="px-4 py-3 text-center">
    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
      baselinePred.knn === 'Toxic'
        ? 'bg-toxic-red/20 text-toxic-red'
        : 'bg-safe-green/20 text-safe-green'
    }`}>
      {baselinePred.knn}
      {!baselinePred.knn_is_real && (
        <span className="ml-1 text-xs text-gray-400" title="Mock prediction">*</span>
      )}
    </span>
  </td>
)}
```

Add a legend:
```jsx
{baseline_predictions && compare_baseline && (
  <div className="mt-2 text-xs text-gray-400">
    * = Simulated prediction (model not yet trained)
  </div>
)}
```

## Training Remaining Models

To train models for the remaining endpoints, use:

```bash
# Train all baseline models for a specific toxicity
python src/train_baseline_models.py --toxicity NR-Aromatase

# Or train all at once
python src/train_all_baseline_models.py
```

## Current Model Coverage

| Endpoint | KNN | NN | RF | SVM | XGBoost | Status |
|----------|-----|----|----|-----|---------|--------|
| NR-AhR | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| NR-AR | ✅ | ✅ | ✅ | ✅ | ✅ | Complete |
| NR-AR-LBD | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| NR-Aromatase | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| NR-ER | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| NR-ER-LBD | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| NR-PPAR-gamma | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| SR-ARE | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| SR-ATAD5 | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| SR-HSE | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| SR-MMP | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |
| SR-p53 | ❌ | ❌ | ❌ | ❌ | ❌ | Not trained |

## Next Steps

1. **Option A - Quick Demo:** Keep using mock data with clear disclaimer in UI
2. **Option B - Partial Real:** Implement hybrid approach (use real models where available)
3. **Option C - Complete:** Train all models first, then integrate

Which approach would you like to take?
