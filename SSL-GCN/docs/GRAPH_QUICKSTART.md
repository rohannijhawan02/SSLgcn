# Quick Start Guide: Molecule to Graph Conversion

## Overview

This guide helps you get started with converting molecules from SMILES strings to graph representations for Graph Convolutional Neural Networks (GCN).

## What You'll Learn

- How molecules are represented as graphs
- Understanding adjacency matrices (N×N)
- Understanding feature matrices (N×74)
- Converting SMILES to DGL graph objects

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `rdkit` - Chemistry library
- `torch` - PyTorch
- `dgl` - Deep Graph Library

### Step 2: Verify Installation

```bash
python test_graph_conversion.py
```

This will run tests to ensure everything is installed correctly.

## Quick Examples

### Example 1: Convert a Single Molecule

```python
from molecule_to_graph import MoleculeToGraphConverter

# Initialize converter
converter = MoleculeToGraphConverter()

# Convert ethanol (CCO)
graph_data = converter.smiles_to_graph("CCO")

# Access the results
print(f"Atoms: {graph_data['num_atoms']}")  # 3 atoms
print(f"Adjacency matrix shape: {graph_data['adjacency_matrix'].shape}")  # (3, 3)
print(f"Feature matrix shape: {graph_data['feature_matrix'].shape}")  # (3, 74)
```

**Output:**
```
Atoms: 3
Adjacency matrix shape: (3, 3)
Feature matrix shape: (3, 74)
```

### Example 2: Understanding the Graph

For ethanol (CCO) with 3 atoms:

**Adjacency Matrix (3×3)** - shows connectivity:
```
[[0, 1, 0],   # Atom 0 (C) connected to Atom 1
 [1, 0, 1],   # Atom 1 (C) connected to Atoms 0 and 2
 [0, 1, 0]]   # Atom 2 (O) connected to Atom 1
```

**Feature Matrix (3×74)** - shows atom properties:
- Row 1: 74 features for first Carbon atom
- Row 2: 74 features for second Carbon atom
- Row 3: 74 features for Oxygen atom

### Example 3: Batch Conversion

```python
from molecule_to_graph import MoleculeToGraphConverter

converter = MoleculeToGraphConverter()

# Convert multiple molecules
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
graphs = converter.batch_convert(smiles_list)

for smiles, graph in zip(smiles_list, graphs):
    if graph:
        print(f"{smiles}: {graph['num_atoms']} atoms")
```

**Output:**
```
CCO: 3 atoms
c1ccccc1: 6 atoms
CC(=O)O: 4 atoms
```

## The 74 Atom Features

Each atom is encoded with 74 features (8 types):

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Atom Type | 44 | Element (C, N, O, S, F, etc.) |
| Degree | 11 | Number of bonds (0-10) |
| Formal Charge | 5 | Charge (-2, -1, 0, +1, +2) |
| Num Hydrogens | 5 | H atoms (0-4+) |
| Hybridization | 5 | SP, SP2, SP3, SP3D, SP3D2 |
| Is Aromatic | 1 | In aromatic system? |
| Is in Ring | 1 | In ring structure? |
| Chirality | 2 | Stereochemistry |

**Total**: 44 + 11 + 5 + 5 + 5 + 1 + 1 + 2 = **74 dimensions**

## Processing Complete Datasets

### Option 1: Process with Graph Conversion

```python
from data_preprocessing import DatasetProcessor

# Initialize with graph conversion enabled
processor = DatasetProcessor(
    data_dir="Data",
    split_ratios=(0.8, 0.1, 0.1),
    convert_to_graph=True  # Enable graph conversion
)

# Process a dataset
result = processor.process_dataset("NR-AhR", save_cache=True)

# Load converted graphs
train_graphs = processor.load_graphs("NR-AhR", "train")
print(f"Loaded {len(train_graphs)} training graphs")
```

### Option 2: Run Interactive Examples

```bash
python example_usage.py
```

Select option 8 or 9 for graph conversion examples.

### Option 3: Run Comprehensive Examples

```bash
python graph_conversion_examples.py
```

This runs through all conversion examples with detailed explanations.

## Output Files

After processing, you'll find these files in `Data/cache/{dataset_name}/`:

```
Data/cache/NR-AhR/
├── splits.pkl              # Train/val/test SMILES splits
├── train_graphs.pkl        # Train graph metadata
├── train_graphs.bin        # Train DGL graphs (binary)
├── val_graphs.pkl          # Validation metadata
├── val_graphs.bin          # Validation DGL graphs
├── test_graphs.pkl         # Test metadata
└── test_graphs.bin         # Test DGL graphs
```

## Loading Saved Graphs

```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor(convert_to_graph=True)

# Load training graphs
train_graphs = processor.load_graphs("NR-AhR", "train")

# Access individual graphs
for graph_data in train_graphs[:5]:
    print(f"SMILES: {graph_data['smiles']}")
    print(f"Atoms: {graph_data['num_atoms']}")
    print(f"Adjacency: {graph_data['adjacency_matrix'].shape}")
    print(f"Features: {graph_data['feature_matrix'].shape}")
    print(f"DGL Graph: {graph_data['graph']}")
```

## Common Issues

### Issue 1: Import Error

```
ImportError: No module named 'dgl'
```

**Solution:**
```bash
pip install torch dgl
```

### Issue 2: SMILES Conversion Fails

```
Warning: Invalid SMILES: ...
```

**Solution:** This is normal. Some SMILES in the dataset may be invalid. The converter skips these automatically.

### Issue 3: Out of Memory

**Solution:** Process datasets one at a time instead of all at once:

```python
processor.process_dataset("NR-AhR", save_cache=True)
```

## Understanding Graph Structure

### Adjacency Matrix
- **Size**: N×N (N = number of atoms)
- **Values**: Binary (0 or 1)
- **Meaning**: A[i,j] = 1 means atoms i and j are connected
- **Property**: Symmetric (undirected graph)

### Feature Matrix
- **Size**: N×74
- **Values**: Binary and numerical
- **Meaning**: Each row is the 74 features for one atom
- **Purpose**: Encodes physicochemical properties

### DGL Graph
- **Type**: `dgl.DGLGraph` object
- **Storage**: Node features in `graph.ndata['h']`
- **Usage**: Ready for GCN input

## Next Steps

1. ✅ Convert molecules to graphs
2. ⏭️ Build Graph Convolutional Network (GCN)
3. ⏭️ Train model on molecular graphs
4. ⏭️ Make predictions on new molecules

## Additional Resources

- **Full Guide**: See `GRAPH_CONVERSION_GUIDE.md`
- **Examples**: Run `python graph_conversion_examples.py`
- **Tests**: Run `python test_graph_conversion.py`
- **API**: See docstrings in `molecule_to_graph.py`

## Summary

You now know how to:
- ✓ Convert SMILES → Graph (adjacency matrix + feature matrix)
- ✓ Understand the 74-dimensional atom features
- ✓ Process complete datasets with graph conversion
- ✓ Load and access converted graphs

The graphs are now ready for Graph Convolutional Neural Network training!
