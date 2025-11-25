# SSL-GCN Data Processing - Quick Start Guide

## 📋 Overview
This package provides complete tools for processing molecular datasets with **scaffold-based splitting** (0.8:0.1:0.1 ratio) for the SSL-GCN project.

## 🎯 What Was Created

### 1. **data_preprocessing.py** (Main Processing Script)
   - **ScaffoldSplitter**: Implements Bemis-Murcko scaffold splitting
   - **DatasetProcessor**: Handles all 12 datasets (NR-* and SR-*)
   - Splits data: 80% train, 10% validation, 10% test
   - Ensures structurally different molecules go to different subsets
   - Caches results for fast loading

### 2. **data_analysis.py** (Analysis & Visualization)
   - **DatasetAnalyzer**: Comprehensive statistics and analysis
   - Scaffold diversity analysis
   - Label distribution comparison
   - Report generation
   - Visualization tools

### 3. **example_usage.py** (Interactive Demo)
   - 7 ready-to-run examples
   - Interactive menu for testing features
   - Step-by-step demonstrations

### 4. **README_DATA.md** (Documentation)
   - Complete documentation
   - Usage examples
   - API reference
   - Output format descriptions

### 5. **requirements.txt** (Dependencies)
   - All required Python packages

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install pandas numpy rdkit matplotlib seaborn
```

### Step 2: Process All Datasets
```bash
python data_preprocessing.py
```

This will:
- Load all 12 datasets from `Data/csv/`
- Apply scaffold-based splitting (0.8:0.1:0.1)
- Save processed splits to `Data/cache/*/splits.pkl`
- Print detailed statistics

### Step 3: Analyze Results
```bash
python data_analysis.py
```

This generates:
- `dataset_summary.txt`: Comprehensive text report
- `dataset_overview.png`: Visual comparison of datasets

## 📊 Available Datasets

**Nuclear Receptor (NR) Series:**
- NR-AhR, NR-AR, NR-AR-LBD, NR-Aromatase
- NR-ER, NR-ER-LBD, NR-PPAR-gamma

**Stress Response (SR) Series:**
- SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

## 💡 Key Features

### Scaffold-Based Splitting
- Uses **Bemis-Murcko scaffolds** to identify molecular frameworks
- Prevents data leakage from similar structures
- Ensures realistic model evaluation
- Assigns structurally different molecules to different subsets

### Advantages Over Random Splitting
- ✅ Tests generalization to novel structures
- ✅ Prevents overfitting on molecular patterns
- ✅ More realistic evaluation
- ✅ Better for drug discovery applications

## 📝 Code Examples

### Process a Single Dataset
```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor()
result = processor.process_dataset("NR-AhR", save_cache=True)

# Access splits
train_smiles = result['train']['smiles']
train_labels = result['train']['labels']
```

### Load Pre-processed Splits
```python
processor = DatasetProcessor()
splits = processor.load_splits("NR-AhR")

# Use in your model
for smiles, label in zip(splits['train']['smiles'], 
                          splits['train']['labels']):
    # Train your model
    pass
```

### Analyze Dataset
```python
from data_analysis import DatasetAnalyzer

analyzer = DatasetAnalyzer()

# Get statistics
stats = analyzer.analyze_dataset_statistics("NR-AhR")
print(f"Positive rate: {stats['positive_rate']:.4f}")

# Analyze scaffolds
scaffold_stats = analyzer.analyze_scaffold_diversity("NR-AhR")
print(f"Unique scaffolds: {scaffold_stats['unique_scaffolds']}")
```

## 📂 Output Structure

After processing:
```
Data/
├── cache/
│   ├── NR-AhR/
│   │   └── splits.pkl          # NEW: Train/val/test splits
│   ├── NR-AR/
│   │   └── splits.pkl
│   └── ... (for all 12 datasets)
└── csv/
    ├── NR-AhR.csv              # Original data
    └── ...
```

## 🔍 What's in splits.pkl?

```python
{
    'dataset_name': 'NR-AhR',
    'train': {
        'smiles': ['CCO...', 'CC(C)...', ...],    # Training molecules
        'labels': [1.0, 0.0, ...],                # Labels
        'mol_ids': ['TOX3021', ...]               # Molecule IDs
    },
    'val': {...},      # 10% for validation
    'test': {...},     # 10% for testing
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

## 🎮 Interactive Demo

```bash
python example_usage.py
```

Available examples:
1. Process Single Dataset
2. Load Cached Splits
3. Analyze Dataset
4. Compare Splits
5. Process All Datasets
6. Generate Reports
7. Custom Split Ratios

## 📈 Expected Output

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

## 🔧 Customization

### Change Split Ratios
```python
processor = DatasetProcessor(
    split_ratios=(0.7, 0.15, 0.15)  # 70:15:15 instead
)
```

### Change Random Seed
```python
splitter = ScaffoldSplitter(seed=123)  # For reproducibility
```

### Process Custom Dataset
```python
# Add your CSV to Data/csv/
processor.datasets.append("MY-DATASET")
result = processor.process_dataset("MY-DATASET")
```

## ⚠️ Important Notes

1. **Exact Ratios**: Scaffold splitting may not achieve exactly 0.8:0.1:0.1 due to discrete scaffold sizes
2. **Invalid SMILES**: Handled gracefully - assigned to unique scaffolds
3. **Existing Cache**: `ecfp_data.pkl` and `graph.bin` are preserved
4. **RDKit Required**: Essential for scaffold generation

## 🐛 Troubleshooting

### "RDKit not found"
```bash
pip install rdkit
# or
conda install -c conda-forge rdkit
```

### "Splits not found"
Run `python data_preprocessing.py` first to generate splits.

### "CSV not found"
Ensure your CSV files are in `Data/csv/` directory.

## 📚 References

- **Scaffold Splitting**: Bemis & Murcko (1996) - Molecular frameworks
- **RDKit**: Open-source cheminformatics toolkit
- **SSL-GCN**: Semi-supervised learning for molecular property prediction

## 🤝 Integration with Existing Code

These scripts work alongside your existing files:
- ✅ Preserves existing `ecfp_data.pkl` and `graph.bin`
- ✅ Adds new `splits.pkl` for each dataset
- ✅ Compatible with dataset.py, localrun, utils (if present)
- ✅ Can be imported and used in your training scripts

## 📞 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Process datasets: `python data_preprocessing.py`
3. ✅ Analyze results: `python data_analysis.py`
4. ✅ Try examples: `python example_usage.py`
5. ✅ Integrate with your model training code

---

**Created for SSL-GCN Project - Scaffold-Based Dataset Splitting**
