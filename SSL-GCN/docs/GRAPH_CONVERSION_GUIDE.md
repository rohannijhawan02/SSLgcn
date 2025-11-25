# Molecule to Graph Conversion for SSL-GCN

## Overview

This guide explains how molecules are converted from SMILES strings to graph representations for use with Graph Convolutional Neural Networks (GCN). The conversion process uses the Deep Graph Library (DGL) and follows the methodology described in the SSL-GCN paper.

## Graph Representation

An undirected graph can be described by two matrices:

1. **Signal (Feature) Matrix H**: Contains node features for each atom
2. **Adjacency Matrix A**: Contains connectivity information between atoms

### For a molecule with N atoms:

- **Adjacency Matrix**: `N × N` (connectivity of atoms)
- **Feature Matrix**: `N × 74` (physicochemical properties of each atom)

## The 8 Default Atom Features (Total: 74 dimensions)

According to Table 2 from the paper, DGL provides 8 default atom features:

| Feature | Type | Dimensions | Description |
|---------|------|------------|-------------|
| 1. Atom Type | One-hot | 44 | Chemical element (C, N, O, S, F, etc.) + unknown |
| 2. Degree | One-hot | 11 | Number of bonds (0-10) |
| 3. Formal Charge | One-hot | 5 | Electrical charge (-2, -1, 0, +1, +2) |
| 4. Number of Hs | One-hot | 5 | Total hydrogen atoms (0-4+) |
| 5. Hybridization | One-hot | 5 | Orbital type (SP, SP2, SP3, SP3D, SP3D2) |
| 6. Is Aromatic | Binary | 1 | Whether atom is in aromatic system |
| 7. Is in Ring | Binary | 1 | Whether atom is in a ring structure |
| 8. Chirality | Encoding | 2 | Stereochemistry information |

**Total**: 44 + 11 + 5 + 5 + 5 + 1 + 1 + 2 = **74 dimensions**

## Conversion Process

### Step 1: SMILES to RDKit Molecule

```python
from rdkit import Chem

smiles = "CCO"  # Ethanol
mol = Chem.MolFromSmiles(smiles)
```

### Step 2: Extract Node Features

For each atom in the molecule, extract the 8 features and encode them into a 74-dimensional vector:

```python
converter = MoleculeToGraphConverter()
graph_data = converter.smiles_to_graph(smiles)
```

**Result for Ethanol (CCO)**:
- N = 3 atoms (C-C-O)
- Feature matrix: `3 × 74`
- Each row represents one atom's features

### Step 3: Build Adjacency Matrix

The adjacency matrix stores connectivity:

```python
# For Ethanol (CCO):
# Atom 0 (C) — Atom 1 (C) — Atom 2 (O)

Adjacency Matrix (3×3):
[[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]]
```

- `A[i,j] = 1` if atoms i and j are connected
- `A[i,j] = 0` otherwise
- Undirected graph: `A[i,j] = A[j,i]`

### Step 4: Create DGL Graph

```python
import dgl

# Create graph from edge list
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=N)

# Add node features
graph.ndata['h'] = feature_matrix
```

## Example: Ethanol (CCO)

```python
from molecule_to_graph import MoleculeToGraphConverter, print_graph_info

converter = MoleculeToGraphConverter()
graph_data = converter.smiles_to_graph("CCO")

print_graph_info(graph_data)
```

**Output**:
```
Molecule: CCO
Number of atoms (N): 3
Adjacency matrix shape: (3, 3)
Feature matrix shape: (3, 74)
DGL graph: Graph(num_nodes=3, num_edges=4)
```

## Example: Benzene (c1ccccc1)

```python
graph_data = converter.smiles_to_graph("c1ccccc1")
```

**Output**:
```
Molecule: c1ccccc1
Number of atoms (N): 6
Adjacency matrix shape: (6, 6)
Feature matrix shape: (6, 74)
DGL graph: Graph(num_nodes=6, num_edges=12)
```

**Adjacency Matrix** (benzene ring):
```
[[0, 1, 0, 0, 0, 1],
 [1, 0, 1, 0, 0, 0],
 [0, 1, 0, 1, 0, 0],
 [0, 0, 1, 0, 1, 0],
 [0, 0, 0, 1, 0, 1],
 [1, 0, 0, 0, 1, 0]]
```

## Feature Matrix Details

Each row in the feature matrix is a 74-dimensional vector representing one atom.

### Example: Carbon atom in benzene

```python
features = graph_data['feature_matrix'][0]
# Shape: (74,)
```

**Feature breakdown**:
- `features[0:44]` → Atom type (one-hot, C is encoded here)
- `features[44:55]` → Degree (one-hot, benzene C has degree 2)
- `features[55:60]` → Formal charge (usually 0 for neutral atoms)
- `features[60:65]` → Number of Hs
- `features[65:70]` → Hybridization (SP2 for aromatic C)
- `features[70]` → Is aromatic (1 for benzene)
- `features[71]` → Is in ring (1 for benzene)
- `features[72:74]` → Chirality

## Usage in Pipeline

### 1. Process Single Dataset

```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor(
    data_dir="Data",
    split_ratios=(0.8, 0.1, 0.1),
    convert_to_graph=True  # Enable graph conversion
)

# Process NR-AhR dataset
result = processor.process_dataset("NR-AhR", save_cache=True)
```

**Output**:
- SMILES splits saved to: `Data/cache/NR-AhR/splits.pkl`
- Graphs saved to: `Data/cache/NR-AhR/train_graphs.pkl` and `.bin`
- Graphs saved to: `Data/cache/NR-AhR/val_graphs.pkl` and `.bin`
- Graphs saved to: `Data/cache/NR-AhR/test_graphs.pkl` and `.bin`

### 2. Load Converted Graphs

```python
# Load training graphs
train_graphs = processor.load_graphs("NR-AhR", "train")

# Access graph data
for graph_data in train_graphs[:5]:
    print(f"SMILES: {graph_data['smiles']}")
    print(f"Atoms: {graph_data['num_atoms']}")
    print(f"DGL Graph: {graph_data['graph']}")
    print(f"Adjacency: {graph_data['adjacency_matrix'].shape}")
    print(f"Features: {graph_data['feature_matrix'].shape}")
```

### 3. Batch Processing

```python
# Process all datasets
processor = DatasetProcessor(convert_to_graph=True)
results = processor.process_all_datasets(save_cache=True)
```

## Files Created

After conversion, the following files are created for each dataset:

```
Data/cache/{dataset_name}/
├── splits.pkl                  # Train/val/test splits with SMILES
├── train_graphs.pkl            # Train graph metadata
├── train_graphs.bin            # Train DGL graphs
├── train_graph_indices.pkl     # Valid graph indices
├── val_graphs.pkl              # Validation graph metadata
├── val_graphs.bin              # Validation DGL graphs
├── val_graph_indices.pkl       # Valid graph indices
├── test_graphs.pkl             # Test graph metadata
├── test_graphs.bin             # Test DGL graphs
└── test_graph_indices.pkl      # Valid graph indices
```

## Running the Examples

### Basic Conversion
```bash
python molecule_to_graph.py
```

### Comprehensive Examples
```bash
python graph_conversion_examples.py
```

### Full Dataset Processing
```bash
python data_preprocessing.py
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `rdkit` - Chemistry and molecular processing
- `torch` - PyTorch for tensor operations
- `dgl` - Deep Graph Library for graph operations

## Next Steps

After graph conversion, the data is ready for:

1. **Graph Convolutional Network (GCN)** training
2. **Message passing** between atoms
3. **Graph-level predictions** for molecular properties
4. **Semi-supervised learning** with SSL-GCN

The converted graphs contain:
- **Structural information** (adjacency matrix)
- **Atomic features** (feature matrix)
- **DGL graph objects** ready for neural network input

## Summary

The molecule-to-graph conversion process transforms SMILES strings into rich graph representations:

1. **Input**: SMILES string (e.g., "CCO")
2. **Output**: 
   - Adjacency matrix (N×N) - connectivity
   - Feature matrix (N×74) - atom properties
   - DGL graph object - ready for GCN

This representation allows Graph Convolutional Neural Networks to learn from molecular structure and make predictions about molecular properties, toxicity, and biological activity.
