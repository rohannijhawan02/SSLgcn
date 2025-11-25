# SSL-GCN Data Processing

This repository contains tools for processing molecular datasets for the SSL-GCN (Semi-Supervised Learning with Graph Convolutional Networks) project.

## Dataset Overview

The project includes 12 labeled molecular datasets:

### Nuclear Receptor (NR) Datasets:
- NR-AhR (Aryl hydrocarbon Receptor)
- NR-AR (Androgen Receptor)
- NR-AR-LBD (Androgen Receptor Ligand Binding Domain)
- NR-Aromatase
- NR-ER (Estrogen Receptor)
- NR-ER-LBD (Estrogen Receptor Ligand Binding Domain)
- NR-PPAR-gamma

### Stress Response (SR) Datasets:
- SR-ARE (Antioxidant Response Element)
- SR-ATAD5
- SR-HSE (Heat Shock Element)
- SR-MMP (Mitochondrial Membrane Potential)
- SR-p53

## Data Splitting Strategy

### Scaffold-Based Splitting
To overcome data bias, we use **scaffold splitting** with a ratio of **0.8:0.1:0.1** (train:validation:test).

**What is Scaffold Splitting?**
- Splits molecules based on their 2D structural framework (Bemis-Murcko scaffolds)
- Ensures structurally different molecules are assigned to different subsets
- Prevents data leakage from similar structures
- Provides more realistic evaluation of model generalization

## Files Created

### 1. `data_preprocessing.py`
Main script for dataset processing and scaffold-based splitting.

**Key Classes:**
- `ScaffoldSplitter`: Implements scaffold-based splitting algorithm
- `DatasetProcessor`: Handles loading, splitting, and caching of datasets

**Features:**
- Load datasets from CSV files
- Generate Bemis-Murcko scaffolds for molecules
- Split data with 0.8:0.1:0.1 ratio
- Cache processed splits for fast loading
- Print detailed statistics and label distributions

### 2. `data_analysis.py`
Tools for analyzing and visualizing dataset statistics.

**Key Classes:**
- `DatasetAnalyzer`: Analyzes datasets and their splits

**Features:**
- Calculate dataset statistics
- Analyze scaffold diversity
- Compare split distributions
- Generate comprehensive summary reports
- Create visualizations

## Installation

### Required Dependencies

```bash
pip install pandas numpy rdkit matplotlib seaborn
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage - Process All Datasets

```python
from data_preprocessing import DatasetProcessor

# Initialize processor
processor = DatasetProcessor(
    data_dir="Data",
    split_ratios=(0.8, 0.1, 0.1)
)

# Process all datasets with scaffold splitting
results = processor.process_all_datasets(save_cache=True)
```

Run from command line:
```bash
python data_preprocessing.py
```

### Process a Single Dataset

```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor()

# Process specific dataset
result = processor.process_dataset("NR-AhR", save_cache=True)

# Access split data
train_data = result['train']
val_data = result['val']
test_data = result['test']

print(f"Train size: {len(train_data['smiles'])}")
print(f"Val size: {len(val_data['smiles'])}")
print(f"Test size: {len(test_data['smiles'])}")
```

### Load Pre-processed Splits

```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor()

# Load cached splits
splits = processor.load_splits("NR-AhR")

# Access data
train_smiles = splits['train']['smiles']
train_labels = splits['train']['labels']
```

### Analyze Datasets

```python
from data_analysis import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer(data_dir="Data")

# Analyze single dataset
stats = analyzer.analyze_dataset_statistics("NR-AhR")
print(f"Positive rate: {stats['positive_rate']:.4f}")

# Analyze scaffold diversity
scaffold_stats = analyzer.analyze_scaffold_diversity("NR-AhR")
print(f"Unique scaffolds: {scaffold_stats['unique_scaffolds']}")

# Compare split distributions
split_comparison = analyzer.compare_split_distributions("NR-AhR")
```

Run from command line:
```bash
python data_analysis.py
```

### Generate Reports and Visualizations

```python
from data_analysis import DatasetAnalyzer

analyzer = DatasetAnalyzer()

# Generate text report
analyzer.generate_summary_report("dataset_summary.txt")

# Create visualization
analyzer.plot_dataset_overview("dataset_overview.png")
```

## Output Structure

After processing, the cache directory will contain:

```
Data/
├── cache/
│   ├── NR-AhR/
│   │   ├── splits.pkl         # Processed splits
│   │   ├── ecfp_data.pkl      # (existing)
│   │   └── graph.bin          # (existing)
│   ├── NR-AR/
│   │   └── ...
│   └── ...
└── csv/
    ├── NR-AhR.csv
    └── ...
```

## Data Format

### CSV Format
Each CSV file contains:
- `NR-*` or `SR-*`: Label column (1.0 for positive, 0.0 for negative, NaN for unlabeled)
- `mol_id`: Unique molecule identifier
- `SMILES`: SMILES string representation of the molecule

### Processed Split Format (pickle)
```python
{
    'dataset_name': 'NR-AhR',
    'train': {
        'smiles': [...],
        'labels': [...],
        'mol_ids': [...]
    },
    'val': {...},
    'test': {...},
    'statistics': {
        'total': 7833,
        'train_size': 6266,
        'val_size': 783,
        'test_size': 784,
        'train_pos_rate': 0.123,
        'val_pos_rate': 0.125,
        'test_pos_rate': 0.121
    }
}
```

## Example Output

```
==============================================================
Processing dataset: NR-AhR
==============================================================
Loaded NR-AhR: 7833 molecules
Valid molecules (with labels): 7833

Performing scaffold-based splitting...

Split Statistics:
  Train: 6266 (80.0%)
  Val:   783 (10.0%)
  Test:  784 (10.0%)

Label Distribution (Positive Rate):
  Train: 0.123
  Val:   0.125
  Test:  0.121

Saved splits to: Data/cache/NR-AhR/splits.pkl
```

## Key Features

1. **Scaffold-Based Splitting**: Ensures structurally diverse molecules in train/val/test sets
2. **Reproducibility**: Fixed random seed for consistent splits
3. **Data Caching**: Fast loading of pre-processed splits
4. **Comprehensive Statistics**: Detailed analysis of datasets and splits
5. **Label Distribution**: Monitors positive/negative balance across splits
6. **Error Handling**: Robust handling of invalid SMILES strings

## Notes

- The scaffold splitting method may not achieve exactly 0.8:0.1:0.1 ratios due to discrete scaffold sizes
- Invalid SMILES strings are handled gracefully by assigning them to unique scaffolds
- All splits maintain label distribution as much as possible
- The existing cache files (`ecfp_data.pkl`, `graph.bin`) are preserved and not modified

## References

- Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks. *Journal of Medicinal Chemistry*, 39(15), 2887-2893.
- RDKit: Open-source cheminformatics; http://www.rdkit.org

## License

This code is part of the SSL-GCN project.
