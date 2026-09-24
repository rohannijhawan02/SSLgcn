"""
Molecule to Graph Conversion for SSL-GCN
Converts SMILES strings to graph representations using Deep Graph Library (DGL).

For each molecule:
- Adjacency matrix A: N×N (connectivity of atoms)
- Feature matrix H: N×74 (node features for each atom)

Where N is the number of atoms in the molecule.
"""

import dgl
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, List, Tuple, Optional
import pickle
import os


class MoleculeToGraphConverter:
    """
    Converts molecular SMILES strings to graph representations.
    Uses DGL's molecule-graph conversion with 8 default atom features.
    """
    
    # Default atom features from DGL (total dimension: 74)
    # Table 2: Default atom features
    ATOM_FEATURES = {
        'atom_type': ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 
                      'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 
                      'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 
                      'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 
                      'Pb'],  # 43 + 1 (unknown)
        'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 11 values
        'formal_charge': [-1, -2, 1, 2, 0],  # 5 values
        'num_Hs': [0, 1, 2, 3, 4],  # 5 values
        'hybridization': [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ],  # 5 values
        'is_aromatic': [False, True],  # 2 values
        'is_in_ring': [False, True],  # 2 values
        'chirality': [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]  # 4 values (but typically represented as 1-hot)
    }
    
    # Total dimension: 44 + 11 + 5 + 5 + 5 + 2 + 2 + 1 = 75 (approximately 74 with variations)
    
    def __init__(self):
        """Initialize the molecule to graph converter."""
        self.feature_dim = 74  # Fixed dimension as per DGL default
    
    def get_atom_features(self, atom) -> np.ndarray:
        """
        Extract atom features and encode them into a binary/numerical vector.
        
        Args:
            atom: RDKit atom object
            
        Returns:
            numpy array of shape (74,) with encoded features
        """
        features = []
        
        # 1. Atom type (one-hot encoding) - 44 dimensions
        atom_type = atom.GetSymbol()
        atom_type_encoding = [0] * 44
        if atom_type in self.ATOM_FEATURES['atom_type']:
            idx = self.ATOM_FEATURES['atom_type'].index(atom_type)
            atom_type_encoding[idx] = 1
        else:
            atom_type_encoding[-1] = 1  # Unknown atom type
        features.extend(atom_type_encoding)
        
        # 2. Degree (one-hot encoding) - 11 dimensions
        degree = atom.GetDegree()
        degree_encoding = [0] * 11
        if degree < 11:
            degree_encoding[degree] = 1
        else:
            degree_encoding[-1] = 1
        features.extend(degree_encoding)
        
        # 3. Formal charge (one-hot encoding) - 5 dimensions
        formal_charge = atom.GetFormalCharge()
        charge_encoding = [0] * 5
        if formal_charge in self.ATOM_FEATURES['formal_charge']:
            idx = self.ATOM_FEATURES['formal_charge'].index(formal_charge)
            charge_encoding[idx] = 1
        features.extend(charge_encoding)
        
        # 4. Number of hydrogens (one-hot encoding) - 5 dimensions
        num_hs = atom.GetTotalNumHs()
        hs_encoding = [0] * 5
        if num_hs < 5:
            hs_encoding[num_hs] = 1
        else:
            hs_encoding[-1] = 1
        features.extend(hs_encoding)
        
        # 5. Hybridization (one-hot encoding) - 5 dimensions
        hybridization = atom.GetHybridization()
        hybrid_encoding = [0] * 5
        if hybridization in self.ATOM_FEATURES['hybridization']:
            idx = self.ATOM_FEATURES['hybridization'].index(hybridization)
            hybrid_encoding[idx] = 1
        features.extend(hybrid_encoding)
        
        # 6. Is aromatic (binary) - 1 dimension
        features.append(int(atom.GetIsAromatic()))
        
        # 7. Is in ring (binary) - 1 dimension
        features.append(int(atom.IsInRing()))
        
        # 8. Chirality (one-hot encoding) - 2 dimensions (simplified)
        chirality = atom.GetChiralTag()
        features.append(int(chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW or 
                           chirality == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW))
        features.append(int(chirality == Chem.rdchem.ChiralType.CHI_OTHER))
        
        # Ensure we have exactly 74 features
        features = features[:74]  # Truncate if needed
        while len(features) < 74:
            features.append(0)  # Pad if needed
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(self, smiles: str) -> Optional[Dict]:
        """
        Convert a SMILES string to a graph representation.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary containing:
                - 'graph': DGL graph object
                - 'adjacency_matrix': N×N adjacency matrix
                - 'feature_matrix': N×74 feature matrix
                - 'num_atoms': Number of atoms N
                - 'smiles': Original SMILES string
            Returns None if conversion fails
        """
        try:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES: {smiles}")
                return None
            
            # Add explicit hydrogens if needed (optional)
            # mol = Chem.AddHs(mol)
            
            num_atoms = mol.GetNumAtoms()
            
            # Extract node features (N×74 feature matrix)
            node_features = []
            for atom in mol.GetAtoms():
                atom_feat = self.get_atom_features(atom)
                node_features.append(atom_feat)
            
            feature_matrix = np.array(node_features, dtype=np.float32)
            
            # Build adjacency matrix and edge list
            adjacency_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int32)
            edges_src = []
            edges_dst = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Undirected graph: add edges in both directions
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                
                # Update adjacency matrix
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
            
            # Create DGL graph
            graph = dgl.graph((edges_src, edges_dst), num_nodes=num_atoms)
            
            # Add node features to the graph
            graph.ndata['h'] = torch.tensor(feature_matrix, dtype=torch.float32)
            
            # Add self-loops (optional, depends on GCN architecture)
            # graph = dgl.add_self_loop(graph)
            
            return {
                'graph': graph,
                'adjacency_matrix': adjacency_matrix,
                'feature_matrix': feature_matrix,
                'num_atoms': num_atoms,
                'smiles': smiles
            }
            
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {str(e)}")
            return None
    
    def batch_convert(self, smiles_list: List[str]) -> List[Dict]:
        """
        Convert a batch of SMILES strings to graphs.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of graph dictionaries (None entries for failed conversions)
        """
        graphs = []
        for smiles in smiles_list:
            graph_data = self.smiles_to_graph(smiles)
            graphs.append(graph_data)
        
        valid_graphs = [g for g in graphs if g is not None]
        print(f"Successfully converted {len(valid_graphs)}/{len(smiles_list)} molecules")
        
        return graphs
    
    def save_graphs(self, graphs: List[Dict], save_path: str):
        """
        Save converted graphs to disk.
        
        Args:
            graphs: List of graph dictionaries
            save_path: Path to save the graphs
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Separate DGL graphs and other data
        dgl_graphs = [g['graph'] for g in graphs if g is not None]
        metadata = [
            {
                'adjacency_matrix': g['adjacency_matrix'],
                'feature_matrix': g['feature_matrix'],
                'num_atoms': g['num_atoms'],
                'smiles': g['smiles']
            }
            for g in graphs if g is not None
        ]
        
        # Save DGL graphs
        dgl_save_path = save_path.replace('.pkl', '.bin')
        dgl.save_graphs(dgl_save_path, dgl_graphs)
        
        # Save metadata
        with open(save_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved {len(dgl_graphs)} graphs to {dgl_save_path}")
        print(f"Saved metadata to {save_path}")
    
    def load_graphs(self, load_path: str) -> List[Dict]:
        """
        Load saved graphs from disk.
        
        Args:
            load_path: Path to the saved graphs
            
        Returns:
            List of graph dictionaries
        """
        # Load DGL graphs
        dgl_load_path = load_path.replace('.pkl', '.bin')
        dgl_graphs, _ = dgl.load_graphs(dgl_load_path)
        
        # Load metadata
        with open(load_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Combine graphs and metadata
        graphs = []
        for dgl_graph, meta in zip(dgl_graphs, metadata):
            graphs.append({
                'graph': dgl_graph,
                'adjacency_matrix': meta['adjacency_matrix'],
                'feature_matrix': meta['feature_matrix'],
                'num_atoms': meta['num_atoms'],
                'smiles': meta['smiles']
            })
        
        print(f"Loaded {len(graphs)} graphs from {load_path}")
        return graphs


def print_graph_info(graph_data: Dict):
    """
    Print information about a graph.
    
    Args:
        graph_data: Dictionary containing graph data
    """
    if graph_data is None:
        print("Invalid graph data")
        return
    
    print(f"\nMolecule: {graph_data['smiles']}")
    print(f"Number of atoms (N): {graph_data['num_atoms']}")
    print(f"Adjacency matrix shape: {graph_data['adjacency_matrix'].shape}")
    print(f"Feature matrix shape: {graph_data['feature_matrix'].shape}")
    print(f"DGL graph: {graph_data['graph']}")
    print(f"  - Number of nodes: {graph_data['graph'].num_nodes()}")
    print(f"  - Number of edges: {graph_data['graph'].num_edges()}")
    print(f"\nFeature matrix (first atom features):")
    print(f"  Shape: (N={graph_data['num_atoms']}, 74)")
    print(f"  First row: {graph_data['feature_matrix'][0][:10]}... (showing first 10 features)")
    print(f"\nAdjacency matrix:")
    print(f"  Shape: (N={graph_data['num_atoms']}, N={graph_data['num_atoms']})")
    print(f"  Matrix:\n{graph_data['adjacency_matrix']}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Molecule to Graph Conversion Example")
    print("="*60)
    
    # Initialize converter
    converter = MoleculeToGraphConverter()
    
    # Example molecules
    example_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"  # More complex molecule from dataset
    ]
    
    print(f"\nConverting {len(example_smiles)} example molecules...")
    
    for smiles in example_smiles:
        graph_data = converter.smiles_to_graph(smiles)
        if graph_data:
            print_graph_info(graph_data)
            print("-" * 60)
    
    # Batch conversion example
    print("\n" + "="*60)
    print("Batch Conversion Example")
    print("="*60)
    graphs = converter.batch_convert(example_smiles)
    
    # Save and load example
    print("\n" + "="*60)
    print("Save and Load Example")
    print("="*60)
    save_path = "Data/cache/example_graphs.pkl"
    converter.save_graphs(graphs, save_path)
    
    loaded_graphs = converter.load_graphs(save_path)
    print(f"Successfully loaded {len(loaded_graphs)} graphs")
