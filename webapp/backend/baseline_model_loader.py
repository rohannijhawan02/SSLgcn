"""
Baseline Model Loader for Web Application
Loads trained baseline ML models (KNN, NN, RF, SVM, XGBoost) for toxicity prediction
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem


class ECFP4Encoder:
    """
    Molecular fingerprint encoder using ECFP4 (Extended Connectivity Fingerprints)
    """
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits
    
    def smiles_to_ecfp4(self, smiles):
        """Convert SMILES to ECFP4 fingerprint"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            return np.array(fp)
        except:
            return None


class BaselineModelLoader:
    """
    Loads and manages trained baseline ML models for toxicity prediction.
    Only loads models for toxicities that have been trained.
    """
    
    def __init__(self, models_dir='../../models/baseline_models'):
        """
        Initialize the baseline model loader.
        
        Args:
            models_dir: Directory containing trained model checkpoints
        """
        # Get absolute path from current file
        current_dir = Path(__file__).parent
        self.models_dir = (current_dir / models_dir).resolve()
        
        # All possible toxicity endpoints
        self.all_toxicities = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        # Model types
        self.model_types = ['KNN', 'NN', 'RF', 'SVM', 'XGBoost']
        
        # Storage for loaded models
        self.models = {}
        self.trained_toxicities = []
        
        # Initialize encoder
        self.encoder = ECFP4Encoder(radius=2, n_bits=2048)
        
        # Load models automatically
        self._load_available_models()
    
    def _load_available_models(self):
        """
        Automatically load all available trained models.
        Only loads models for toxicities that have trained models.
        """
        print(f"\n{'='*60}")
        print(f"Loading Baseline Models")
        print(f"{'='*60}")
        print(f"Models directory: {self.models_dir}")
        
        if not os.path.exists(self.models_dir):
            print(f"⚠ Warning: Models directory not found")
            print(f"  No baseline models will be available")
            return
        
        loaded_count = 0
        
        for toxicity in self.all_toxicities:
            toxicity_dir = os.path.join(self.models_dir, toxicity)
            
            if not os.path.exists(toxicity_dir):
                continue
            
            # Check if any model files exist for this toxicity
            has_models = False
            self.models[toxicity] = {}
            
            for model_type in self.model_types:
                model_path = os.path.join(toxicity_dir, f"{model_type}_model.pkl")
                
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_dict = pickle.load(f)
                        
                        self.models[toxicity][model_type] = model_dict
                        loaded_count += 1
                        has_models = True
                        print(f"  ✓ Loaded: {toxicity} - {model_type}")
                    except Exception as e:
                        print(f"  ✗ Error loading {toxicity} - {model_type}: {str(e)}")
            
            # Track which toxicities have trained models
            if has_models:
                self.trained_toxicities.append(toxicity)
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total models loaded: {loaded_count}")
        print(f"  Toxicities with trained models: {len(self.trained_toxicities)}")
        print(f"  Trained toxicities: {', '.join(self.trained_toxicities)}")
        print(f"{'='*60}\n")
        
        if loaded_count == 0:
            print("⚠ No baseline models were loaded.")
            print("  Baseline predictions will show as '-' for all toxicities.\n")
    
    def is_toxicity_trained(self, toxicity: str) -> bool:
        """
        Check if a toxicity has trained models available.
        
        Args:
            toxicity: Toxicity endpoint ID
            
        Returns:
            True if models are available, False otherwise
        """
        return toxicity in self.trained_toxicities
    
    def predict(self, smiles: str, toxicity: str, model_type: str = 'RF') -> Optional[Dict[str, Any]]:
        """
        Predict toxicity for a SMILES string using a specific model.
        
        Args:
            smiles: SMILES string
            toxicity: Toxicity endpoint ID
            model_type: Model type (KNN, NN, RF, SVM, XGBoost)
            
        Returns:
            Prediction dictionary or None if model not available
        """
        # Check if toxicity has trained models
        if toxicity not in self.trained_toxicities:
            return None
        
        # Check if specific model type is available
        if toxicity not in self.models or model_type not in self.models[toxicity]:
            return None
        
        # Encode SMILES to ECFP4
        fingerprint = self.encoder.smiles_to_ecfp4(smiles)
        if fingerprint is None:
            return {'error': 'Invalid SMILES string'}
        
        fingerprint = fingerprint.reshape(1, -1)
        
        # Get model
        model_dict = self.models[toxicity][model_type]
        model = model_dict['model']
        
        try:
            # Handle SVM with scaler
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
                'label': 'Toxic' if pred == 1 else 'Non-toxic',
                'confidence': float(abs(proba - 0.5) * 2)  # Scale to 0-1
            }
        except Exception as e:
            print(f"Error predicting with {model_type} for {toxicity}: {e}")
            return None
    
    def predict_all_models(self, smiles: str, toxicity: str) -> Dict[str, Any]:
        """
        Predict toxicity using all available model types for a toxicity endpoint.
        
        Args:
            smiles: SMILES string
            toxicity: Toxicity endpoint ID
            
        Returns:
            Dictionary with predictions from all available models
        """
        if toxicity not in self.trained_toxicities:
            return {}
        
        results = {}
        for model_type in self.model_types:
            if model_type in self.models.get(toxicity, {}):
                pred = self.predict(smiles, toxicity, model_type)
                if pred and 'error' not in pred:
                    results[model_type] = pred
        
        return results


# Global instance
baseline_loader = None

def initialize_baseline_loader():
    """Initialize the global baseline model loader"""
    global baseline_loader
    try:
        baseline_loader = BaselineModelLoader()
        return True
    except Exception as e:
        print(f"Error initializing baseline models: {e}")
        return False


# Initialize on import
try:
    initialize_baseline_loader()
except Exception as e:
    print(f"Failed to initialize baseline models: {e}")
    baseline_loader = None
