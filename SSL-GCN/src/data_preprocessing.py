"""
Data Preprocessing for SSL-GCN
This script handles dataset loading and scaffold-based splitting for molecular datasets.
Splitting ratio: 0.8:0.1:0.1 (train:validation:test)
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from typing import List, Dict, Tuple
import random

# Import graph conversion functionality
try:
    from molecule_to_graph import MoleculeToGraphConverter
    GRAPH_CONVERSION_AVAILABLE = True
except ImportError:
    GRAPH_CONVERSION_AVAILABLE = False
    print("Warning: molecule_to_graph module not available. Graph conversion will be disabled.")


class ScaffoldSplitter:
    """
    Implements scaffold-based splitting to overcome data bias.
    Splits molecules based on their 2D structural framework (Bemis-Murcko scaffolds).
    """
    
    def __init__(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """
        Initialize the scaffold splitter.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.8)
            val_ratio: Proportion of data for validation (default: 0.1)
            test_ratio: Proportion of data for testing (default: 0.1)
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
        """
        Generate Bemis-Murcko scaffold for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            include_chirality: Whether to include chirality in scaffold
            
        Returns:
            Scaffold SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=include_chirality
            )
            return scaffold
        except:
            return None
    
    def scaffold_split(
        self, 
        smiles_list: List[str], 
        labels: List[float] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset based on molecular scaffolds.
        Ensures structurally different molecules are in different subsets.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Optional list of labels (for stratification awareness)
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Generate scaffolds for all molecules
        scaffolds = defaultdict(list)
        
        for idx, smiles in enumerate(smiles_list):
            scaffold = self.generate_scaffold(smiles)
            if scaffold is not None:
                scaffolds[scaffold].append(idx)
            else:
                # Handle invalid SMILES by assigning to a unique scaffold
                scaffolds[f"invalid_{idx}"].append(idx)
        
        # Sort scaffolds by size (descending) for balanced splitting
        scaffold_sets = [
            (scaffold, indices) 
            for scaffold, indices in scaffolds.items()
        ]
        scaffold_sets.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Distribute scaffolds into train/val/test sets
        train_indices, val_indices, test_indices = [], [], []
        train_size, val_size, test_size = 0, 0, 0
        total_size = len(smiles_list)
        
        for scaffold, indices in scaffold_sets:
            # Calculate current proportions
            train_prop = train_size / total_size if total_size > 0 else 0
            val_prop = val_size / total_size if total_size > 0 else 0
            test_prop = test_size / total_size if total_size > 0 else 0
            
            # Assign to the set that needs more data
            if train_prop < self.train_ratio:
                train_indices.extend(indices)
                train_size += len(indices)
            elif val_prop < self.val_ratio:
                val_indices.extend(indices)
                val_size += len(indices)
            else:
                test_indices.extend(indices)
                test_size += len(indices)
        
        return train_indices, val_indices, test_indices


class DatasetProcessor:
    """
    Processes molecular datasets for SSL-GCN training.
    Handles loading, splitting, and caching of data.
    """
    
    def __init__(
        self, 
        data_dir: str = "Data",
        cache_dir: str = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        convert_to_graph: bool = True
    ):
        """
        Initialize the dataset processor.
        
        Args:
            data_dir: Directory containing the dataset CSV files
            cache_dir: Directory for caching processed data
            split_ratios: Tuple of (train, val, test) ratios
            convert_to_graph: Whether to convert molecules to graph representations
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "csv")
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        self.split_ratios = split_ratios
        self.convert_to_graph = convert_to_graph and GRAPH_CONVERSION_AVAILABLE
        
        self.splitter = ScaffoldSplitter(
            train_ratio=split_ratios[0],
            val_ratio=split_ratios[1],
            test_ratio=split_ratios[2]
        )
        
        # Initialize graph converter if available
        if self.convert_to_graph:
            self.graph_converter = MoleculeToGraphConverter()
            print("Graph conversion enabled")
        else:
            self.graph_converter = None
            if convert_to_graph:
                print("Warning: Graph conversion requested but not available")
        
        # Available datasets
        self.datasets = [
            "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a dataset from CSV file.
        
        Args:
            dataset_name: Name of the dataset (e.g., "NR-AhR")
            
        Returns:
            DataFrame with the dataset
        """
        csv_path = os.path.join(self.csv_dir, f"{dataset_name}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {dataset_name}: {len(df)} molecules")
        
        return df
    
    def process_dataset(
        self, 
        dataset_name: str, 
        save_cache: bool = True
    ) -> Dict:
        """
        Process a single dataset with scaffold splitting.
        
        Args:
            dataset_name: Name of the dataset
            save_cache: Whether to save processed data to cache
            
        Returns:
            Dictionary containing split data and statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_dataset(dataset_name)
        
        # Clean data - remove rows with missing labels
        df_clean = df.dropna(subset=[dataset_name])
        print(f"Valid molecules (with labels): {len(df_clean)}")
        
        # Extract SMILES and labels
        smiles_list = df_clean['SMILES'].tolist()
        labels = df_clean[dataset_name].tolist()
        mol_ids = df_clean['mol_id'].tolist()
        
        # Perform scaffold split
        print("\nPerforming scaffold-based splitting...")
        train_idx, val_idx, test_idx = self.splitter.scaffold_split(
            smiles_list, labels
        )
        
        # Create split datasets
        train_data = {
            'smiles': [smiles_list[i] for i in train_idx],
            'labels': [labels[i] for i in train_idx],
            'mol_ids': [mol_ids[i] for i in train_idx]
        }
        
        val_data = {
            'smiles': [smiles_list[i] for i in val_idx],
            'labels': [labels[i] for i in val_idx],
            'mol_ids': [mol_ids[i] for i in val_idx]
        }
        
        test_data = {
            'smiles': [smiles_list[i] for i in test_idx],
            'labels': [labels[i] for i in test_idx],
            'mol_ids': [mol_ids[i] for i in test_idx]
        }
        
        # Print statistics
        print(f"\nSplit Statistics:")
        print(f"  Train: {len(train_idx)} ({len(train_idx)/len(df_clean)*100:.1f}%)")
        print(f"  Val:   {len(val_idx)} ({len(val_idx)/len(df_clean)*100:.1f}%)")
        print(f"  Test:  {len(test_idx)} ({len(test_idx)/len(df_clean)*100:.1f}%)")
        
        # Label distribution
        train_labels = np.array(train_data['labels'])
        val_labels = np.array(val_data['labels'])
        test_labels = np.array(test_data['labels'])
        
        print(f"\nLabel Distribution (Positive Rate):")
        print(f"  Train: {np.mean(train_labels):.3f}")
        print(f"  Val:   {np.mean(val_labels):.3f}")
        print(f"  Test:  {np.mean(test_labels):.3f}")
        
        # Prepare result
        result = {
            'dataset_name': dataset_name,
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'statistics': {
                'total': len(df_clean),
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'test_size': len(test_idx),
                'train_pos_rate': float(np.mean(train_labels)),
                'val_pos_rate': float(np.mean(val_labels)),
                'test_pos_rate': float(np.mean(test_labels))
            }
        }
        
        # Save to cache
        if save_cache:
            self.save_splits(dataset_name, result)
            
            # Convert to graphs if enabled
            if self.convert_to_graph:
                print("\nConverting molecules to graphs...")
                self.convert_and_save_graphs(dataset_name, result)
        
        return result
    
    def convert_and_save_graphs(self, dataset_name: str, result: Dict):
        """
        Convert molecules to graph representations and save them.
        
        Args:
            dataset_name: Name of the dataset
            result: Dictionary containing split data with SMILES
        """
        if not self.graph_converter:
            print("Warning: Graph converter not initialized")
            return
        
        dataset_cache_dir = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Convert each split
        for split_name in ['train', 'val', 'test']:
            split_data = result[split_name]
            smiles_list = split_data['smiles']
            
            print(f"\nConverting {split_name} set ({len(smiles_list)} molecules)...")
            
            # Convert SMILES to graphs
            graphs = self.graph_converter.batch_convert(smiles_list)
            
            # Filter out invalid conversions
            valid_graphs = []
            valid_indices = []
            for i, graph in enumerate(graphs):
                if graph is not None:
                    valid_graphs.append(graph)
                    valid_indices.append(i)
            
            # Save graphs
            if valid_graphs:
                graph_path = os.path.join(dataset_cache_dir, f"{split_name}_graphs.pkl")
                self.graph_converter.save_graphs(valid_graphs, graph_path)
                
                # Also save mapping of valid indices
                mapping_path = os.path.join(dataset_cache_dir, f"{split_name}_graph_indices.pkl")
                with open(mapping_path, 'wb') as f:
                    pickle.dump(valid_indices, f)
                
                print(f"Saved {len(valid_graphs)} valid graphs for {split_name} set")
            else:
                print(f"Warning: No valid graphs for {split_name} set")
    
    def load_graphs(self, dataset_name: str, split_name: str = 'train'):
        """
        Load pre-converted graphs from cache.
        
        Args:
            dataset_name: Name of the dataset
            split_name: Name of the split ('train', 'val', or 'test')
            
        Returns:
            List of graph dictionaries
        """
        if not self.graph_converter:
            raise RuntimeError("Graph converter not initialized")
        
        graph_path = os.path.join(
            self.cache_dir, dataset_name, f"{split_name}_graphs.pkl"
        )
        
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Graphs not found: {graph_path}. "
                f"Run process_dataset() with convert_to_graph=True first."
            )
        
        graphs = self.graph_converter.load_graphs(graph_path)
        return graphs
    
    def save_splits(self, dataset_name: str, result: Dict):
        """
        Save split data to cache directory.
        
        Args:
            dataset_name: Name of the dataset
            result: Dictionary containing split data
        """
        dataset_cache_dir = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Save splits as pickle
        splits_path = os.path.join(dataset_cache_dir, "splits.pkl")
        with open(splits_path, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"\nSaved splits to: {splits_path}")
    
    def load_splits(self, dataset_name: str) -> Dict:
        """
        Load pre-processed splits from cache.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing split data
        """
        splits_path = os.path.join(
            self.cache_dir, dataset_name, "splits.pkl"
        )
        
        if not os.path.exists(splits_path):
            raise FileNotFoundError(
                f"Splits not found: {splits_path}. "
                f"Run process_dataset() first."
            )
        
        with open(splits_path, 'rb') as f:
            result = pickle.load(f)
        
        return result
    
    def process_all_datasets(self, save_cache: bool = True) -> Dict[str, Dict]:
        """
        Process all available datasets.
        
        Args:
            save_cache: Whether to save processed data to cache
            
        Returns:
            Dictionary mapping dataset names to their split data
        """
        results = {}
        
        print("\n" + "="*60)
        print("Processing All Datasets with Scaffold Splitting")
        print(f"Split Ratios: {self.split_ratios[0]}:{self.split_ratios[1]}:{self.split_ratios[2]}")
        print("="*60)
        
        for dataset_name in self.datasets:
            try:
                result = self.process_dataset(dataset_name, save_cache)
                results[dataset_name] = result
            except Exception as e:
                print(f"\nError processing {dataset_name}: {str(e)}")
                continue
        
        # Print summary
        print("\n" + "="*60)
        print("Summary of All Datasets")
        print("="*60)
        print(f"{'Dataset':<20} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-"*60)
        
        for dataset_name, result in results.items():
            stats = result['statistics']
            print(
                f"{dataset_name:<20} "
                f"{stats['total']:<8} "
                f"{stats['train_size']:<8} "
                f"{stats['val_size']:<8} "
                f"{stats['test_size']:<8}"
            )
        
        return results


def main():
    """
    Main function to demonstrate dataset processing.
    """
    # Initialize processor
    processor = DatasetProcessor(
        data_dir="Data",
        split_ratios=(0.8, 0.1, 0.1)
    )
    
    # Process all datasets
    results = processor.process_all_datasets(save_cache=True)
    
    print("\n" + "="*60)
    print("Dataset Processing Complete!")
    print("="*60)
    print(f"Processed {len(results)} datasets")
    print(f"Splits saved to: {processor.cache_dir}")


if __name__ == "__main__":
    main()
