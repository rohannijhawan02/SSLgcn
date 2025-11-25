"""
GCN Model Loader for WebApp Backend
Loads trained GCN models and provides prediction interface
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

import torch
import dgl
import numpy as np
from rdkit import Chem
from typing import Dict, Any, Optional, List

from model import create_gcn_model


class GCNModelLoader:
    """Load and use trained GCN models for toxicity prediction"""
    
    def __init__(self, models_dir='../../checkpoints', device='cpu'):
        """
        Initialize GCN model loader.
        
        Args:
            models_dir: Directory containing trained model checkpoints
            device: Device to run predictions on ('cpu' or 'cuda')
        """
        self.models_dir = models_dir
        self.device = device
        self.models = {}
        self.loaded_endpoints = set()
        
        self.toxicity_names = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER',
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        print(f"GCN Model Loader initialized (device: {device})")
    
    def load_model(self, endpoint: str) -> bool:
        """
        Load a trained GCN model for a specific endpoint.
        
        Args:
            endpoint: Toxicity endpoint name
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if endpoint in self.loaded_endpoints:
            return True
        
        model_path = os.path.join(self.models_dir, endpoint, 'best_model.pt')
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found for {endpoint} at {model_path}")
            return False
        
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
            
            self.models[endpoint] = {
                'model': model,
                'in_feats': in_feats
            }
            self.loaded_endpoints.add(endpoint)
            
            print(f"Loaded GCN model for {endpoint}")
            return True
            
        except Exception as e:
            print(f"Error loading model for {endpoint}: {str(e)}")
            return False
    
    def smiles_to_graph(self, smiles: str):
        """
        Convert SMILES string to DGL graph.
        
        Args:
            smiles: SMILES string
            
        Returns:
            DGL graph or None if conversion fails
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
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
            print(f"Error converting SMILES to graph: {str(e)}")
            return None
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom"""
        # Atom type (one-hot encoding)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']
        atom_type = [1 if atom.GetSymbol() == t else 0 for t in atom_types]
        
        # Degree
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
        
        # Total hydrogens
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
        
        # Radical electrons
        radical = [atom.GetNumRadicalElectrons()]
        
        # Combine features
        features = (
            atom_type +           # 12
            degree +              # 1
            formal_charge +       # 1
            hybridization +       # 5
            aromatic +            # 1
            total_h +             # 1
            in_ring +             # 1
            chirality +           # 4
            radical               # 1
        )
        
        # Pad to 74 features
        while len(features) < 74:
            features.append(0)
        
        return features[:74]
    
    def predict(self, smiles: str, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Make a prediction for a SMILES string at a specific endpoint.
        
        Args:
            smiles: SMILES string
            endpoint: Toxicity endpoint name
            
        Returns:
            Prediction dictionary or None if prediction fails
        """
        # Load model if not already loaded
        if endpoint not in self.loaded_endpoints:
            if not self.load_model(endpoint):
                return None
        
        # Convert SMILES to graph
        graph = self.smiles_to_graph(smiles)
        if graph is None:
            return None
        
        # Get model
        model_info = self.models[endpoint]
        model = model_info['model']
        
        # Make prediction
        try:
            with torch.no_grad():
                graph = graph.to(self.device)
                features = graph.ndata['feat'].float()
                
                # Forward pass
                logits = model(graph, features)
                probs = torch.softmax(logits, dim=1)[0]
                
                # Get prediction
                pred_class = logits.argmax(dim=1).item()
                prob_toxic = probs[1].item()
                prob_nontoxic = probs[0].item()
                
                return {
                    'prediction': int(pred_class),
                    'label': 'Toxic' if pred_class == 1 else 'Non-toxic',
                    'probability': float(prob_toxic),
                    'confidence': float(max(prob_toxic, prob_nontoxic))
                }
        
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None


# Global GCN model loader instance
gcn_loader = GCNModelLoader()
