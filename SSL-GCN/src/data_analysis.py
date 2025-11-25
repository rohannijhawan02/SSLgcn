"""
Data Analysis and Visualization for SSL-GCN Datasets
Provides tools to analyze and visualize dataset statistics and scaffold distributions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    # Suppress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')  # Disable all RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Some features will be disabled.")


class DatasetAnalyzer:
    """
    Analyzes molecular datasets and their scaffold-based splits.
    """
    
    def __init__(self, data_dir: str = "Data", cache_dir: str = None, verbose: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing the dataset CSV files
            cache_dir: Directory containing cached splits
            verbose: Whether to print detailed information
        """
        self.data_dir = data_dir
        self.csv_dir = os.path.join(data_dir, "csv")
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        self.verbose = verbose
        
        self.datasets = [
            "NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    
    def analyze_dataset_statistics(self, dataset_name: str) -> Dict:
        """
        Analyze basic statistics of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing statistics
        """
        csv_path = os.path.join(self.csv_dir, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path)
        
        # Clean data
        df_clean = df.dropna(subset=[dataset_name])
        
        stats = {
            'total_molecules': len(df),
            'valid_molecules': len(df_clean),
            'missing_labels': len(df) - len(df_clean),
            'positive_samples': int(df_clean[dataset_name].sum()),
            'negative_samples': int((df_clean[dataset_name] == 0).sum()),
            'positive_rate': float(df_clean[dataset_name].mean()),
            'unique_smiles': df_clean['SMILES'].nunique()
        }
        
        return stats
    
    def _generate_scaffold_safe(self, smiles: str) -> str:
        """
        Safely generate scaffold from SMILES, handling errors.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Scaffold SMILES or None if invalid
        """
        try:
            # Try to create molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Try to generate scaffold
            # Use includeChirality=False to reduce complexity
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, 
                includeChirality=False
            )
            return scaffold
            
        except Exception:
            # Silently handle any RDKit errors
            return None
    
    def analyze_scaffold_diversity(self, dataset_name: str) -> Dict:
        """
        Analyze scaffold diversity in a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing scaffold statistics
        """
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        csv_path = os.path.join(self.csv_dir, f"{dataset_name}.csv")
        df = pd.read_csv(csv_path)
        df_clean = df.dropna(subset=[dataset_name])
        
        if self.verbose:
            print(f"  Processing {len(df_clean)} molecules for scaffold analysis...")
        
        # Generate scaffolds with error tracking
        scaffolds = []
        invalid_count = 0
        
        for idx, smiles in enumerate(df_clean['SMILES']):
            # Show progress for large datasets
            if self.verbose and (idx + 1) % 1000 == 0:
                print(f"    Processed {idx + 1}/{len(df_clean)} molecules...")
            
            scaffold = self._generate_scaffold_safe(smiles)
            if scaffold is not None:
                scaffolds.append(scaffold)
            else:
                invalid_count += 1
        
        if self.verbose and invalid_count > 0:
            print(f"  ⚠️  Skipped {invalid_count} invalid/problematic molecules ({invalid_count/len(df_clean)*100:.1f}%)")
        
        # Analyze scaffold distribution
        scaffold_counts = Counter(scaffolds)
        
        stats = {
            'total_molecules': len(df_clean),
            'valid_scaffolds': len(scaffolds),
            'invalid_molecules': invalid_count,
            'unique_scaffolds': len(scaffold_counts),
            'scaffold_diversity_ratio': len(scaffold_counts) / len(scaffolds) if len(scaffolds) > 0 else 0,
            'avg_molecules_per_scaffold': len(scaffolds) / len(scaffold_counts) if len(scaffold_counts) > 0 else 0,
            'max_molecules_in_scaffold': max(scaffold_counts.values()) if scaffold_counts else 0,
            'scaffolds_with_single_molecule': sum(1 for count in scaffold_counts.values() if count == 1),
            'top_5_scaffold_sizes': sorted(scaffold_counts.values(), reverse=True)[:5]
        }
        
        return stats
    
    def compare_split_distributions(self, dataset_name: str) -> Dict:
        """
        Compare label distributions across train/val/test splits.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing distribution comparison
        """
        splits_path = os.path.join(self.cache_dir, dataset_name, "splits.pkl")
        
        if not os.path.exists(splits_path):
            return {"error": f"Splits not found for {dataset_name}"}
        
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        train_labels = np.array(splits['train']['labels'])
        val_labels = np.array(splits['val']['labels'])
        test_labels = np.array(splits['test']['labels'])
        
        comparison = {
            'train': {
                'size': len(train_labels),
                'positive_rate': float(np.mean(train_labels)),
                'positive_count': int(np.sum(train_labels)),
                'negative_count': int(len(train_labels) - np.sum(train_labels))
            },
            'val': {
                'size': len(val_labels),
                'positive_rate': float(np.mean(val_labels)),
                'positive_count': int(np.sum(val_labels)),
                'negative_count': int(len(val_labels) - np.sum(val_labels))
            },
            'test': {
                'size': len(test_labels),
                'positive_rate': float(np.mean(test_labels)),
                'positive_count': int(np.sum(test_labels)),
                'negative_count': int(len(test_labels) - np.sum(test_labels))
            }
        }
        
        return comparison
    
    def generate_summary_report(self, output_file: str = "dataset_summary.txt"):
        """
        Generate a comprehensive summary report for all datasets.
        
        Args:
            output_file: Path to save the report
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SSL-GCN Dataset Summary Report")
        report_lines.append("="*80)
        report_lines.append("")
        
        for dataset_name in self.datasets:
            try:
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Analyzing: {dataset_name}")
                    print(f"{'='*60}")
                
                report_lines.append(f"\n{'='*80}")
                report_lines.append(f"Dataset: {dataset_name}")
                report_lines.append(f"{'='*80}")
                
                # Basic statistics
                stats = self.analyze_dataset_statistics(dataset_name)
                report_lines.append("\nBasic Statistics:")
                report_lines.append(f"  Total molecules: {stats['total_molecules']}")
                report_lines.append(f"  Valid molecules: {stats['valid_molecules']}")
                report_lines.append(f"  Missing labels: {stats['missing_labels']}")
                report_lines.append(f"  Unique SMILES: {stats['unique_smiles']}")
                report_lines.append(f"  Positive samples: {stats['positive_samples']}")
                report_lines.append(f"  Negative samples: {stats['negative_samples']}")
                report_lines.append(f"  Positive rate: {stats['positive_rate']:.4f}")
                
                # Scaffold diversity
                if RDKIT_AVAILABLE:
                    scaffold_stats = self.analyze_scaffold_diversity(dataset_name)
                    if 'error' not in scaffold_stats:
                        report_lines.append("\nScaffold Diversity:")
                        report_lines.append(f"  Total molecules: {scaffold_stats['total_molecules']}")
                        report_lines.append(f"  Valid scaffolds: {scaffold_stats['valid_scaffolds']}")
                        report_lines.append(f"  Invalid molecules: {scaffold_stats['invalid_molecules']}")
                        report_lines.append(f"  Unique scaffolds: {scaffold_stats['unique_scaffolds']}")
                        report_lines.append(f"  Diversity ratio: {scaffold_stats['scaffold_diversity_ratio']:.4f}")
                        report_lines.append(f"  Avg molecules/scaffold: {scaffold_stats['avg_molecules_per_scaffold']:.2f}")
                        report_lines.append(f"  Max molecules in scaffold: {scaffold_stats['max_molecules_in_scaffold']}")
                        report_lines.append(f"  Single-molecule scaffolds: {scaffold_stats['scaffolds_with_single_molecule']}")
                
                # Split comparison
                split_comp = self.compare_split_distributions(dataset_name)
                if 'error' not in split_comp:
                    report_lines.append("\nSplit Distribution (with 0.8:0.1:0.1 ratio):")
                    for split_name in ['train', 'val', 'test']:
                        split_data = split_comp[split_name]
                        report_lines.append(f"\n  {split_name.capitalize()}:")
                        report_lines.append(f"    Size: {split_data['size']}")
                        report_lines.append(f"    Positive: {split_data['positive_count']} ({split_data['positive_rate']:.4f})")
                        report_lines.append(f"    Negative: {split_data['negative_count']}")
                else:
                    report_lines.append(f"\n  Note: {split_comp['error']}")
                
            except Exception as e:
                report_lines.append(f"\n  Error analyzing {dataset_name}: {str(e)}")
                if self.verbose:
                    print(f"  ✗ Error: {str(e)}")
        
        # Write report
        report_text = "\n".join(report_lines)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to: {output_file}")
        return report_text
    
    def plot_dataset_overview(self, save_path: str = "dataset_overview.png"):
        """
        Create visualization of dataset overview.
        
        Args:
            save_path: Path to save the plot
        """
        # Collect statistics for all datasets
        dataset_names = []
        total_sizes = []
        positive_rates = []
        
        for dataset_name in self.datasets:
            try:
                stats = self.analyze_dataset_statistics(dataset_name)
                dataset_names.append(dataset_name)
                total_sizes.append(stats['valid_molecules'])
                positive_rates.append(stats['positive_rate'])
            except:
                continue
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Dataset sizes
        axes[0].barh(dataset_names, total_sizes, color='steelblue')
        axes[0].set_xlabel('Number of Molecules', fontsize=12)
        axes[0].set_title('Dataset Sizes', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Positive rates
        colors = ['green' if rate > 0.5 else 'orange' for rate in positive_rates]
        axes[1].barh(dataset_names, positive_rates, color=colors)
        axes[1].set_xlabel('Positive Rate', fontsize=12)
        axes[1].set_title('Label Distribution (Positive Rate)', fontsize=14, fontweight='bold')
        axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
        plt.close()


def main():
    """
    Main function to demonstrate dataset analysis.
    """
    print("="*80)
    print("SSL-GCN Dataset Analysis")
    print("="*80)
    
    analyzer = DatasetAnalyzer(data_dir="Data", verbose=True)
    
    print("\n" + "="*80)
    print("Generating comprehensive dataset analysis report...")
    print("="*80)
    analyzer.generate_summary_report("dataset_summary.txt")
    
    print("\n" + "="*80)
    print("Creating visualization...")
    print("="*80)
    try:
        analyzer.plot_dataset_overview("dataset_overview.png")
    except Exception as e:
        print(f"✗ Could not create plots: {e}")
        print("  Make sure matplotlib and seaborn are installed.")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()