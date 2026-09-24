"""
Prediction Module for Baseline ML Models
Load trained baseline models and predict toxicity for new SMILES strings
Supports KNN, NN, RF, SVM, and XGBoost models
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from train_baseline_models import ECFP4Encoder


class BaselineModelPredictor:
    """
    Predict toxicity using baseline ML models (KNN, NN, RF, SVM, XGBoost).
    """
    
    def __init__(self, models_dir='models/baseline_models', model_type='all'):
        """
        Initialize predictor with trained models.
        
        Args:
            models_dir: Directory containing trained model checkpoints
            model_type: Type of model to load ('KNN', 'NN', 'RF', 'SVM', 'XGBoost', or 'all')
        """
        self.models_dir = models_dir
        self.model_type = model_type
        self.models = {}
        
        self.toxicity_names = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.model_types = ['KNN', 'NN', 'RF', 'SVM', 'XGBoost']
        
        # Initialize encoder
        self.encoder = ECFP4Encoder(radius=2, n_bits=2048)
        
        print(f"Baseline Model Predictor Initialized")
        print(f"Models directory: {models_dir}")
        print(f"Model type: {model_type}")
    
    def load_models(self, toxicity_list=None):
        """
        Load trained models for specified toxicities.
        
        Args:
            toxicity_list: List of toxicity endpoints to load (default: all)
        """
        if toxicity_list is None:
            toxicity_list = self.toxicity_names
        
        model_types_to_load = self.model_types if self.model_type == 'all' else [self.model_type]
        
        print(f"\nLoading models...")
        loaded_count = 0
        
        for toxicity in toxicity_list:
            toxicity_dir = os.path.join(self.models_dir, toxicity)
            
            if not os.path.exists(toxicity_dir):
                print(f"Warning: No models found for {toxicity}")
                continue
            
            self.models[toxicity] = {}
            
            for model_type in model_types_to_load:
                model_path = os.path.join(toxicity_dir, f"{model_type}_model.pkl")
                
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model_dict = pickle.load(f)
                        
                        self.models[toxicity][model_type] = model_dict
                        loaded_count += 1
                        print(f"  Loaded: {toxicity} - {model_type}")
                    except Exception as e:
                        print(f"  Error loading {toxicity} - {model_type}: {str(e)}")
                else:
                    print(f"  Not found: {toxicity} - {model_type}")
        
        print(f"\nTotal models loaded: {loaded_count}")
        
        if loaded_count == 0:
            print("\nWarning: No models were loaded. Please train models first.")
    
    def predict_smiles(self, smiles, toxicity=None, model_type=None):
        """
        Predict toxicity for a SMILES string.
        
        Args:
            smiles: SMILES string
            toxicity: Specific toxicity endpoint (default: all loaded)
            model_type: Specific model type (default: all loaded)
            
        Returns:
            dict: Predictions from all models
        """
        # Encode SMILES to ECFP4
        fingerprint = self.encoder.smiles_to_ecfp4(smiles)
        
        if fingerprint is None:
            return {'error': 'Invalid SMILES string'}
        
        fingerprint = fingerprint.reshape(1, -1)
        
        # Determine which models to use
        toxicities = [toxicity] if toxicity else list(self.models.keys())
        model_types = [model_type] if model_type else None
        
        results = {}
        
        for tox in toxicities:
            if tox not in self.models:
                continue
            
            results[tox] = {}
            
            models_to_use = model_types if model_types else list(self.models[tox].keys())
            
            for mtype in models_to_use:
                if mtype not in self.models[tox]:
                    continue
                
                model_dict = self.models[tox][mtype]
                model = model_dict['model']
                
                # Handle SVM with scaler
                if mtype == 'SVM' and 'scaler' in model_dict:
                    fingerprint_scaled = model_dict['scaler'].transform(fingerprint)
                    pred = model.predict(fingerprint_scaled)[0]
                    proba = model.predict_proba(fingerprint_scaled)[0, 1]
                else:
                    pred = model.predict(fingerprint)[0]
                    proba = model.predict_proba(fingerprint)[0, 1]
                
                results[tox][mtype] = {
                    'prediction': int(pred),
                    'probability': float(proba),
                    'label': 'Toxic' if pred == 1 else 'Non-toxic'
                }
        
        return results
    
    def predict_batch(self, smiles_list, toxicity=None, model_type=None):
        """
        Predict toxicity for multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            toxicity: Specific toxicity endpoint (default: all)
            model_type: Specific model type (default: all)
            
        Returns:
            list: Predictions for all SMILES
        """
        results = []
        
        for i, smiles in enumerate(smiles_list):
            print(f"Processing {i+1}/{len(smiles_list)}: {smiles}")
            pred = self.predict_smiles(smiles, toxicity, model_type)
            pred['smiles'] = smiles
            results.append(pred)
        
        return results
    
    def predict_from_file(self, input_file, output_file=None, toxicity=None, model_type=None):
        """
        Predict toxicity for SMILES from a file.
        
        Args:
            input_file: Path to CSV file with 'SMILES' column
            output_file: Path to save results (default: auto-generated)
            toxicity: Specific toxicity endpoint (default: all)
            model_type: Specific model type (default: all)
        """
        print(f"\nReading SMILES from: {input_file}")
        df = pd.read_csv(input_file)
        
        if 'SMILES' not in df.columns:
            raise ValueError("Input file must have a 'SMILES' column")
        
        smiles_list = df['SMILES'].tolist()
        print(f"Total SMILES to predict: {len(smiles_list)}")
        
        # Predict
        results = self.predict_batch(smiles_list, toxicity, model_type)
        
        # Format results
        output_data = []
        
        for result in results:
            smiles = result['smiles']
            base_row = {'SMILES': smiles}
            
            if 'error' in result:
                base_row['error'] = result['error']
                output_data.append(base_row)
                continue
            
            # Flatten results
            for tox, models in result.items():
                if tox == 'smiles':
                    continue
                for mtype, pred in models.items():
                    base_row[f"{tox}_{mtype}_prediction"] = pred['prediction']
                    base_row[f"{tox}_{mtype}_probability"] = pred['probability']
                    base_row[f"{tox}_{mtype}_label"] = pred['label']
            
            output_data.append(base_row)
        
        # Save results
        output_df = pd.DataFrame(output_data)
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_predictions.csv')
        
        output_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        return output_df


def main():
    """
    Main function for baseline model prediction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict toxicity using baseline models')
    parser.add_argument('--smiles', type=str, help='Single SMILES string to predict')
    parser.add_argument('--input', type=str, help='Input CSV file with SMILES')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--toxicity', type=str, help='Specific toxicity endpoint')
    parser.add_argument('--model', type=str, choices=['KNN', 'NN', 'RF', 'SVM', 'XGBoost', 'all'],
                       default='all', help='Model type to use')
    parser.add_argument('--models_dir', type=str, default='models/baseline_models',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = BaselineModelPredictor(
        models_dir=args.models_dir,
        model_type=args.model
    )
    
    # Load models
    predictor.load_models()
    
    if args.smiles:
        # Predict single SMILES
        print(f"\nPredicting toxicity for: {args.smiles}")
        results = predictor.predict_smiles(args.smiles, args.toxicity, args.model)
        
        print("\nResults:")
        for tox, models in results.items():
            if tox == 'smiles' or tox == 'error':
                continue
            print(f"\n{tox}:")
            for model_type, pred in models.items():
                print(f"  {model_type}: {pred['label']} (probability: {pred['probability']:.4f})")
    
    elif args.input:
        # Predict from file
        predictor.predict_from_file(args.input, args.output, args.toxicity, args.model)
    
    else:
        print("Please provide either --smiles or --input argument")
        parser.print_help()


if __name__ == '__main__':
    main()
