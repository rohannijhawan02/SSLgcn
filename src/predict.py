"""
Prediction Module for New Unknown Molecular Samples
Load trained models and predict toxicity for new SMILES strings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dgl
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import pickle
from pathlib import Path

from model import create_gcn_model


class ToxicityPredictor:
    """
    Predict toxicity for new molecular samples across all toxicity endpoints
    """
    
    def __init__(self, models_dir='checkpoints', device='cpu'):
        """
        Initialize predictor with trained models
        
        Args:
            models_dir: Directory containing trained model checkpoints
            device: Device to run predictions on ('cpu' or 'cuda')
        """
        self.models_dir = models_dir
        self.device = device
        self.models = {}
        self.toxicity_names = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        print(f"Initializing Toxicity Predictor...")
        print(f"Device: {device}")
        print(f"Models directory: {models_dir}")
        
    def load_models(self, toxicity_list=None):
        """
        Load trained models for specified toxicities
        
        Args:
            toxicity_list: List of toxicity names (None for all)
        """
        if toxicity_list is None:
            toxicity_list = self.toxicity_names
        
        print(f"\nLoading {len(toxicity_list)} trained models...")
        
        for toxicity_name in toxicity_list:
            model_path = os.path.join(self.models_dir, toxicity_name, 'best_model.pt')
            
            if not os.path.exists(model_path):
                print(f"  ⚠️  Warning: Model not found for {toxicity_name}")
                print(f"      Expected at: {model_path}")
                continue
            
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Get model architecture parameters
                in_feats = checkpoint.get('in_feats', 74)
                hidden_dims = checkpoint.get('hidden_dims', [64, 128, 256])
                num_layers = checkpoint.get('num_layers', 3)
                classifier_hidden = checkpoint.get('classifier_hidden', 128)
                dropout = checkpoint.get('dropout', 0.3)
                
                # Create model with same architecture
                model = create_gcn_model(
                    in_feats=in_feats,
                    hidden_dims=hidden_dims,
                    num_layers=num_layers,
                    classifier_hidden=classifier_hidden,
                    num_classes=2,
                    dropout=dropout
                )
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[toxicity_name] = {
                    'model': model,
                    'in_feats': in_feats
                }
                print(f"  ✓ Loaded {toxicity_name}")
                
            except Exception as e:
                print(f"  ✗ Error loading {toxicity_name}: {str(e)}")
        
        print(f"\n✓ Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def smiles_to_graph(self, smiles):
        """
        Convert SMILES string to DGL graph
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            DGL graph object or None if conversion fails
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  ✗ Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Get atom features
            num_atoms = mol.GetNumAtoms()
            atom_features = []
            
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)
            
            # Build adjacency matrix
            adjacency = Chem.GetAdjacencyMatrix(mol)
            
            # Create DGL graph
            src, dst = np.where(adjacency > 0)
            g = dgl.graph((src, dst), num_nodes=num_atoms)
            
            # Add self-loops
            g = dgl.add_self_loop(g)
            
            # Add node features
            g.ndata['feat'] = torch.FloatTensor(atom_features)
            
            return g
            
        except Exception as e:
            print(f"  ✗ Error converting SMILES to graph: {str(e)}")
            return None
    
    def _get_atom_features(self, atom):
        """
        Extract features for a single atom
        
        Args:
            atom: RDKit atom object
            
        Returns:
            List of atom features
        """
        # Atom type (one-hot encoding for common elements)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']
        atom_type = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
        
        # Degree (number of bonds)
        degree = [atom.GetDegree()]
        
        # Formal charge
        formal_charge = [atom.GetFormalCharge()]
        
        # Hybridization
        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        hybridization = [1 if atom.GetHybridization() == h else 0 for h in hybridization_types]
        
        # Aromaticity
        aromatic = [1 if atom.GetIsAromatic() else 0]
        
        # Total number of hydrogens
        total_h = [atom.GetTotalNumHs()]
        
        # In ring
        in_ring = [1 if atom.IsInRing() else 0]
        
        # Chirality
        chiral_types = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        chirality = [1 if atom.GetChiralTag() == c else 0 for c in chiral_types]
        
        # Number of radical electrons
        radical = [atom.GetNumRadicalElectrons()]
        
        # Combine all features
        features = (
            atom_type +           # 12 features
            degree +              # 1 feature
            formal_charge +       # 1 feature
            hybridization +       # 5 features
            aromatic +            # 1 feature
            total_h +             # 1 feature
            in_ring +             # 1 feature
            chirality +           # 4 features
            radical               # 1 feature
        )
        # Total: 27 features (may need padding to match training feature size)
        
        # Pad to expected feature size (74 from training)
        while len(features) < 74:
            features.append(0)
        
        return features[:74]  # Ensure exact size
    
    def predict_single(self, smiles, return_probabilities=False):
        """
        Predict toxicity for a single molecule
        
        Args:
            smiles: SMILES string of the molecule
            return_probabilities: If True, return probabilities instead of binary predictions
            
        Returns:
            Dictionary with predictions for each toxicity endpoint
        """
        if not self.models:
            print("Error: No models loaded. Call load_models() first.")
            return None
        
        # Convert SMILES to graph
        graph = self.smiles_to_graph(smiles)
        if graph is None:
            return None
        
        # Make predictions for each toxicity endpoint
        predictions = {
            'smiles': smiles,
            'predictions': {}
        }
        
        with torch.no_grad():
            for toxicity_name, model_info in self.models.items():
                model = model_info['model']
                
                # Get features
                graph = graph.to(self.device)
                features = graph.ndata['feat'].float()
                
                # Forward pass
                logits = model(graph, features)
                probs = torch.softmax(logits, dim=1)[0]
                
                # Get prediction
                pred_class = logits.argmax(dim=1).item()
                prob_toxic = probs[1].item()
                
                if return_probabilities:
                    predictions['predictions'][toxicity_name] = {
                        'prediction': 'Toxic' if pred_class == 1 else 'Non-toxic',
                        'probability': prob_toxic,
                        'confidence': max(probs[0].item(), probs[1].item())
                    }
                else:
                    predictions['predictions'][toxicity_name] = 'Toxic' if pred_class == 1 else 'Non-toxic'
        
        return predictions
    
    def predict_batch(self, smiles_list, return_probabilities=False):
        """
        Predict toxicity for multiple molecules
        
        Args:
            smiles_list: List of SMILES strings
            return_probabilities: If True, return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        print(f"\nPredicting toxicity for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list, 1):
            print(f"  Processing {i}/{len(smiles_list)}: {smiles[:50]}...")
            pred = self.predict_single(smiles, return_probabilities)
            if pred is not None:
                results.append(pred)
        
        print(f"\n✓ Completed predictions for {len(results)}/{len(smiles_list)} molecules")
        return results
    
    def predict_from_csv(self, input_csv, smiles_column='smiles', output_csv=None):
        """
        Predict toxicity from a CSV file containing SMILES
        
        Args:
            input_csv: Path to input CSV file
            smiles_column: Name of column containing SMILES strings
            output_csv: Path to save results (None for auto-naming)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nReading molecules from: {input_csv}")
        
        # Read input CSV
        df = pd.read_csv(input_csv)
        
        if smiles_column not in df.columns:
            print(f"Error: Column '{smiles_column}' not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        smiles_list = df[smiles_column].tolist()
        print(f"Found {len(smiles_list)} molecules")
        
        # Make predictions
        predictions = self.predict_batch(smiles_list, return_probabilities=True)
        
        # Create results DataFrame
        results_data = []
        
        for pred in predictions:
            row = {'smiles': pred['smiles']}
            
            # Add predictions for each toxicity
            for toxicity_name in self.toxicity_names:
                if toxicity_name in pred['predictions']:
                    pred_info = pred['predictions'][toxicity_name]
                    row[f'{toxicity_name}_prediction'] = pred_info['prediction']
                    row[f'{toxicity_name}_probability'] = pred_info['probability']
                    row[f'{toxicity_name}_confidence'] = pred_info['confidence']
                else:
                    row[f'{toxicity_name}_prediction'] = 'N/A'
                    row[f'{toxicity_name}_probability'] = 0.0
                    row[f'{toxicity_name}_confidence'] = 0.0
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        if output_csv is None:
            output_csv = input_csv.replace('.csv', '_predictions.csv')
        
        results_df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")
        
        return results_df
    
    def get_summary_statistics(self, predictions):
        """
        Get summary statistics from predictions
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with summary statistics
        """
        summary = []
        
        for toxicity_name in self.toxicity_names:
            if toxicity_name not in self.models:
                continue
            
            toxic_count = 0
            total_count = 0
            avg_prob = 0.0
            
            for pred in predictions:
                if toxicity_name in pred['predictions']:
                    pred_info = pred['predictions'][toxicity_name]
                    if pred_info['prediction'] == 'Toxic':
                        toxic_count += 1
                    avg_prob += pred_info['probability']
                    total_count += 1
            
            if total_count > 0:
                summary.append({
                    'toxicity': toxicity_name,
                    'total_molecules': total_count,
                    'predicted_toxic': toxic_count,
                    'predicted_non_toxic': total_count - toxic_count,
                    'toxic_percentage': (toxic_count / total_count) * 100,
                    'avg_toxicity_probability': avg_prob / total_count
                })
        
        return pd.DataFrame(summary)


def main():
    """
    Example usage of the ToxicityPredictor
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict toxicity for new molecules')
    parser.add_argument('--smiles', type=str, help='Single SMILES string to predict')
    parser.add_argument('--input_csv', type=str, help='CSV file containing SMILES')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='Name of SMILES column in CSV')
    parser.add_argument('--output_csv', type=str, help='Output CSV file for predictions')
    parser.add_argument('--models_dir', type=str, default='checkpoints', help='Directory with trained models')
    parser.add_argument('--toxicities', nargs='+', help='Specific toxicities to predict (default: all)')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize predictor
    predictor = ToxicityPredictor(models_dir=args.models_dir, device=device)
    
    # Load models
    success = predictor.load_models(toxicity_list=args.toxicities)
    if not success:
        print("\n❌ Failed to load models. Please check the models directory.")
        return
    
    # Make predictions
    if args.smiles:
        # Single SMILES prediction
        print(f"\n{'='*80}")
        print(f"Predicting toxicity for: {args.smiles}")
        print(f"{'='*80}\n")
        
        result = predictor.predict_single(args.smiles, return_probabilities=True)
        
        if result:
            print(f"\nResults:")
            print(f"SMILES: {result['smiles']}\n")
            
            for toxicity, pred_info in result['predictions'].items():
                print(f"{toxicity:20s}: {pred_info['prediction']:12s} "
                      f"(probability: {pred_info['probability']:.4f}, "
                      f"confidence: {pred_info['confidence']:.4f})")
        
    elif args.input_csv:
        # Batch prediction from CSV
        results_df = predictor.predict_from_csv(
            args.input_csv,
            smiles_column=args.smiles_column,
            output_csv=args.output_csv
        )
        
        if results_df is not None:
            print(f"\n{'='*80}")
            print(f"Prediction Summary:")
            print(f"{'='*80}")
            
            # Get summary statistics
            predictions = predictor.predict_batch(results_df['smiles'].tolist(), return_probabilities=True)
            summary = predictor.get_summary_statistics(predictions)
            print(f"\n{summary.to_string(index=False)}")
    
    else:
        # Example usage
        print("\n" + "="*80)
        print("Example Usage:")
        print("="*80)
        print("\n1. Predict for a single molecule:")
        print("   python src/predict.py --smiles 'CC(=O)OC1=CC=CC=C1C(=O)O'")
        print("\n2. Predict for molecules in a CSV file:")
        print("   python src/predict.py --input_csv molecules.csv --smiles_column SMILES")
        print("\n3. Predict specific toxicities only:")
        print("   python src/predict.py --smiles 'CCO' --toxicities NR-AhR NR-AR")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
