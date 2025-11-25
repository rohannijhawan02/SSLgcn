# Step-by-Step Guide to Get Split Data & Prepare for GCN

## 🎯 Your Goal
1. ✅ Get train/val/test splits with graphs (adjacency + features)
2. ✅ Prepare data for GCN implementation

## 📋 Steps to Follow

### Step 1: Install Dependencies ⚙️

Run this command in PowerShell:

```powershell
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- rdkit (chemistry)
- torch, dgl (graph neural networks)
- matplotlib, seaborn (visualization)

**⏱️ Time**: 2-5 minutes

---

### Step 2: Test Installation ✅

```powershell
python test_graph_conversion.py
```

**What this does:**
- ✓ Checks if all packages installed correctly
- ✓ Tests basic SMILES → graph conversion
- ✓ Verifies feature dimensions (N×74)
- ✓ Validates adjacency matrices

**Expected output**: "🎉 All tests passed!"

**⏱️ Time**: 30 seconds

---

### Step 3: Process ONE Dataset (Recommended First) 📊

**Option A: Interactive (Recommended)**
```powershell
python example_usage.py
```
Then select:
- Option **9**: "Process with Graphs"

**Option B: Direct Script**
```powershell
python -c "from data_preprocessing import DatasetProcessor; p = DatasetProcessor(convert_to_graph=True); p.process_dataset('NR-AhR', save_cache=True)"
```

**What this does:**
1. Loads NR-AhR dataset from CSV
2. Performs scaffold-based splitting (80/10/10)
3. Converts all SMILES to graphs
4. Saves everything to cache

**Output files created:**
```
Data/cache/NR-AhR/
├── splits.pkl              ← SMILES and labels
├── train_graphs.pkl        ← Train graphs metadata
├── train_graphs.bin        ← Train DGL graphs (binary)
├── val_graphs.pkl          ← Validation metadata
├── val_graphs.bin          ← Validation DGL graphs
├── test_graphs.pkl         ← Test metadata
└── test_graphs.bin         ← Test DGL graphs
```

**⏱️ Time**: 1-3 minutes (depending on dataset size)

---

### Step 4: Process ALL Datasets (Optional) 🔄

```powershell
python data_preprocessing.py
```

**What this does:**
- Processes all 12 datasets
- Creates splits for each
- Converts all to graphs
- Saves to cache

**Datasets processed:**
- NR-AhR, NR-AR, NR-AR-LBD, NR-Aromatase
- NR-ER, NR-ER-LBD, NR-PPAR-gamma
- SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

**⏱️ Time**: 5-15 minutes (all datasets)

---

### Step 5: Verify Your Data 🔍

```powershell
python -c "from data_preprocessing import DatasetProcessor; p = DatasetProcessor(convert_to_graph=True); graphs = p.load_graphs('NR-AhR', 'train'); print(f'Loaded {len(graphs)} training graphs'); print(f'First graph: {graphs[0][\"num_atoms\"]} atoms, adjacency {graphs[0][\"adjacency_matrix\"].shape}, features {graphs[0][\"feature_matrix\"].shape}')"
```

**Expected output:**
```
Loaded 6266 training graphs
First graph: 15 atoms, adjacency (15, 15), features (15, 74)
```

---

## 🎯 What You Have Now

After Step 3 or 4, you have:

### ✅ Split Data
- **Train set** (80%): For training GCN
- **Validation set** (10%): For hyperparameter tuning
- **Test set** (10%): For final evaluation

### ✅ Graph Representations
For each molecule:
- **Adjacency Matrix** (N×N): Atom connectivity
- **Feature Matrix** (N×74): Atom properties
- **DGL Graph**: Ready for GCN input

### ✅ Cached Files
All saved in `Data/cache/{dataset_name}/`

---

## 🚀 Quick Start (All in One)

```powershell
# Install & test
pip install -r requirements.txt
python test_graph_conversion.py

# Process one dataset
python -c "from data_preprocessing import DatasetProcessor; p = DatasetProcessor(convert_to_graph=True); p.process_dataset('NR-AhR', save_cache=True)"

# Verify
python -c "from data_preprocessing import DatasetProcessor; p = DatasetProcessor(convert_to_graph=True); graphs = p.load_graphs('NR-AhR', 'train'); print(f'✅ Success! Loaded {len(graphs)} training graphs')"
```

---

## 📊 Using Your Data for GCN

Once you have the graphs, use them like this:

```python
from data_preprocessing import DatasetProcessor

# Load processed graphs
processor = DatasetProcessor(convert_to_graph=True)

# Get train/val/test graphs
train_graphs = processor.load_graphs("NR-AhR", "train")
val_graphs = processor.load_graphs("NR-AhR", "val")
test_graphs = processor.load_graphs("NR-AhR", "test")

# Get splits (SMILES and labels)
splits = processor.load_splits("NR-AhR")
train_labels = splits['train']['labels']
val_labels = splits['val']['labels']
test_labels = splits['test']['labels']

# Access individual graphs
for graph_data in train_graphs[:5]:
    print(f"SMILES: {graph_data['smiles']}")
    print(f"Adjacency: {graph_data['adjacency_matrix'].shape}")
    print(f"Features: {graph_data['feature_matrix'].shape}")
    print(f"DGL Graph: {graph_data['graph']}")
    print(f"Nodes: {graph_data['graph'].num_nodes()}")
    print(f"Edges: {graph_data['graph'].num_edges()}")
    print()
```

---

## 🎓 Next: GCN Implementation

Your data is ready! Now you need to:

1. **Build GCN Model**
   ```python
   import dgl.nn.pytorch as dglnn
   import torch.nn as nn
   
   class GCN(nn.Module):
       def __init__(self, in_feats=74, hidden_size=128, num_classes=2):
           super().__init__()
           self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
           self.conv2 = dglnn.GraphConv(hidden_size, hidden_size)
           self.classify = nn.Linear(hidden_size, num_classes)
   ```

2. **Train on Your Graphs**
   - Use train_graphs for training
   - Use val_graphs for validation
   - Use test_graphs for final evaluation

3. **Make Predictions**
   - Input: DGL graph with node features
   - Output: Molecular property prediction

---

## 🐛 Troubleshooting

### Error: "No module named 'torch'"
```powershell
pip install torch
```

### Error: "No module named 'dgl'"
```powershell
pip install dgl
```

### Error: "No module named 'rdkit'"
```powershell
pip install rdkit
```

### Warning: "Some SMILES failed to convert"
This is normal - some molecules in the dataset may be invalid. They are automatically skipped.

---

## ✅ Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Run tests (`python test_graph_conversion.py`)
- [ ] Process at least one dataset
- [ ] Verify graphs loaded correctly
- [ ] Ready for GCN implementation!

---

**Current Status**: You have the graph conversion code ready.
**Next Step**: Run Step 1 (install dependencies) in PowerShell!
