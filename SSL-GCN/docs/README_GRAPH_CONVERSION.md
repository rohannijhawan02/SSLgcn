# SSL-GCN: Molecule to Graph Conversion

## 🎯 Overview

Complete implementation of **molecule-to-graph conversion** for SSL-GCN (Semi-Supervised Learning on Graph Convolutional Networks). Converts molecular SMILES strings into graph representations using Deep Graph Library (DGL) with:

- **Adjacency Matrix** (N×N): Atom connectivity
- **Feature Matrix** (N×74): Atom properties (8 default features from DGL)
- **DGL Graph**: Ready for Graph Convolutional Neural Networks

## 📦 What's Included

### Core Files
- `molecule_to_graph.py` - Main conversion implementation
- `data_preprocessing.py` - Integration with data pipeline
- `graph_conversion_examples.py` - Comprehensive examples
- `test_graph_conversion.py` - Complete test suite
- `visualize_conversion.py` - Visual demonstrations

### Documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical summary
- `GRAPH_CONVERSION_GUIDE.md` - Complete guide
- `GRAPH_QUICKSTART.md` - Quick start tutorial
- `README_GRAPH_CONVERSION.md` - This file

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_graph_conversion.py
```

### 3. Run Examples
```bash
python graph_conversion_examples.py
```

### 4. Convert Single Molecule
```python
from molecule_to_graph import MoleculeToGraphConverter

converter = MoleculeToGraphConverter()
graph_data = converter.smiles_to_graph("CCO")

print(f"Atoms: {graph_data['num_atoms']}")  # 3
print(f"Adjacency: {graph_data['adjacency_matrix'].shape}")  # (3, 3)
print(f"Features: {graph_data['feature_matrix'].shape}")  # (3, 74)
```

### 5. Process Complete Dataset
```python
from data_preprocessing import DatasetProcessor

processor = DatasetProcessor(convert_to_graph=True)
result = processor.process_dataset("NR-AhR", save_cache=True)

# Load converted graphs
train_graphs = processor.load_graphs("NR-AhR", "train")
```

## 📊 Graph Structure

### Example: Ethanol (CCO)

```
Molecular Structure:  H₃C—CH₂—OH

Adjacency Matrix (3×3):
  [[0, 1, 0],   ← C connected to C
   [1, 0, 1],   ← C connected to C and O
   [0, 1, 0]]   ← O connected to C

Feature Matrix (3×74):
  [[C features...],   ← 74 features for first C
   [C features...],   ← 74 features for second C
   [O features...]]   ← 74 features for O
```

## 🔑 The 74 Atom Features

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| Atom Type | 44 | C, N, O, S, F, etc. |
| Degree | 11 | Number of bonds (0-10) |
| Formal Charge | 5 | -2, -1, 0, +1, +2 |
| Num Hydrogens | 5 | 0, 1, 2, 3, 4+ |
| Hybridization | 5 | SP, SP2, SP3, SP3D, SP3D2 |
| Is Aromatic | 1 | In aromatic system? |
| Is in Ring | 1 | In ring structure? |
| Chirality | 2 | Stereochemistry |

**Total: 74 dimensions**

## 📁 Files & Commands

### Run Examples
```bash
# Basic conversion examples
python molecule_to_graph.py

# Comprehensive examples with explanations
python graph_conversion_examples.py

# Visual demonstrations
python visualize_conversion.py

# Interactive examples
python example_usage.py
```

### Run Tests
```bash
# Full test suite
python test_graph_conversion.py

# Process all datasets
python data_preprocessing.py
```

### Access Documentation
- **Quick Start**: `GRAPH_QUICKSTART.md`
- **Full Guide**: `GRAPH_CONVERSION_GUIDE.md`
- **Summary**: `IMPLEMENTATION_SUMMARY.md`

## 💡 Key Concepts

### Why Graphs?
Molecules are naturally graphs where:
- **Nodes** = Atoms (with 74 features each)
- **Edges** = Chemical bonds
- **Structure** = Molecular topology

### Adjacency Matrix (N×N)
- Stores which atoms are connected
- Symmetric (undirected graph)
- Binary values (0 or 1)
- Enables message passing in GCN

### Feature Matrix (N×74)
- Each row = one atom
- 74 features encode atom properties
- Binary and numerical encoding
- Input to Graph Convolutional layers

### DGL Graph
- PyTorch-compatible graph object
- Features stored in `graph.ndata['h']`
- Ready for neural network training

## 🎓 Usage Patterns

### Pattern 1: Single Molecule
```python
converter = MoleculeToGraphConverter()
graph = converter.smiles_to_graph("CCO")
```

### Pattern 2: Batch Processing
```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
graphs = converter.batch_convert(smiles_list)
```

### Pattern 3: Full Dataset Pipeline
```python
processor = DatasetProcessor(convert_to_graph=True)
processor.process_all_datasets(save_cache=True)
```

### Pattern 4: Load Cached Graphs
```python
processor = DatasetProcessor(convert_to_graph=True)
train_graphs = processor.load_graphs("NR-AhR", "train")
val_graphs = processor.load_graphs("NR-AhR", "val")
test_graphs = processor.load_graphs("NR-AhR", "test")
```

## 📈 Output Files

After processing, graphs are saved to:
```
Data/cache/{dataset_name}/
├── splits.pkl              # SMILES and labels
├── train_graphs.pkl        # Train metadata
├── train_graphs.bin        # Train DGL graphs
├── val_graphs.pkl          # Validation metadata
├── val_graphs.bin          # Validation DGL graphs
├── test_graphs.pkl         # Test metadata
└── test_graphs.bin         # Test DGL graphs
```

## ✅ Testing & Verification

The test suite verifies:
- ✓ All imports working
- ✓ Basic SMILES conversion
- ✓ Batch processing
- ✓ Feature dimensions (N×74)
- ✓ Adjacency matrix properties
- ✓ Integration with data pipeline

Run: `python test_graph_conversion.py`

## 🔧 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
rdkit>=2022.3.1
torch>=1.10.0
dgl>=0.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install: `pip install -r requirements.txt`

## 🎯 Next Steps

### Current Status: ✅ Graph Conversion Complete

**What's Ready:**
- ✓ SMILES → Graph conversion
- ✓ Adjacency matrices (N×N)
- ✓ Feature matrices (N×74)
- ✓ DGL graph objects
- ✓ Batch processing
- ✓ Caching system

**Next: GCN Implementation**
1. Build Graph Convolutional layers
2. Implement message passing
3. Train on molecular graphs
4. Make predictions

## 📚 Learning Path

1. **Start Here**: `GRAPH_QUICKSTART.md`
2. **Run Examples**: `python graph_conversion_examples.py`
3. **Visual Guide**: `python visualize_conversion.py`
4. **Full Details**: `GRAPH_CONVERSION_GUIDE.md`
5. **Implementation**: `IMPLEMENTATION_SUMMARY.md`

## 🐛 Troubleshooting

### Import Error: torch/dgl
```bash
pip install torch dgl
```

### Invalid SMILES
Some molecules may fail conversion - this is normal and handled gracefully.

### Memory Issues
Process datasets one at a time instead of all at once.

## 📞 Summary

**Input**: SMILES strings (e.g., "CCO")

**Process**: 
1. Parse with RDKit
2. Extract 74 atom features
3. Build N×N adjacency matrix
4. Create DGL graph

**Output**: Graph with:
- Adjacency matrix (connectivity)
- Feature matrix (properties)
- DGL graph object (GCN-ready)

**Status**: ✅ **Complete and tested**

The molecular data is now in perfect format for Graph Convolutional Neural Networks!

---

## 📖 Quick Reference

| Command | Purpose |
|---------|---------|
| `python test_graph_conversion.py` | Run all tests |
| `python graph_conversion_examples.py` | See examples |
| `python visualize_conversion.py` | Visual guide |
| `python molecule_to_graph.py` | Basic demo |
| `python example_usage.py` | Interactive menu |

**Ready to implement GCN!** 🚀
