# SSL-GCN: Graph Convolutional Network for Molecular Toxicity Prediction

This project implements a Graph Convolutional Network (GCN) for predicting molecular toxicity across multiple targets using the Tox21 dataset.

## 📁 Project Structure

```
SSL-GCN/
├── src/                           # Source code
│   ├── model.py                   # GCN model architecture
│   ├── train.py                   # Training pipeline
│   ├── data_preprocessing.py      # Data preprocessing
│   ├── molecule_to_graph.py       # Molecule to graph conversion
│   ├── hyperparameter_tuning.py   # Hyperparameter optimization
│   ├── train_all_toxicities.py    # Train all datasets
│   └── visualize_results.py       # Result visualization
│
├── Data/                          # Dataset files
│   ├── csv/                       # Raw CSV data (Tox21)
│   └── cache/                     # Processed graph data cache
│
├── models/                        # Trained models
│   ├── checkpoints/               # Model checkpoints
│   └── tuning_results/            # Hyperparameter tuning results
│
├── results/                       # Training results and metrics
│   └── [dataset_name]/            # Results per dataset
│
├── docs/                          # Documentation
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

3. **Check packages**:
   ```bash
   python main/check_packages.py
   ```

4. **Run data preprocessing**:
   ```bash
   python main/data_preprocessing.py
   ```

### 2. Train a Model
```bash
# Train on a specific dataset
python src/train.py --dataset NR-AhR --epochs 100

# Train on all datasets
python src/train_all_toxicities.py
```

### 3. Hyperparameter Tuning
```bash
python src/hyperparameter_tuning.py --dataset NR-AhR --trials 20
```

### 4. Visualize Results
```bash
python src/visualize_results.py --dataset NR-AhR
```

## 📊 Datasets

The project uses the Tox21 dataset with 12 toxicity assays:
- **Nuclear Receptor (NR)**: AhR, AR, AR-LBD, Aromatase, ER, ER-LBD, PPAR-gamma
- **Stress Response (SR)**: ARE, ATAD5, HSE, MMP, p53

All data files are located in `Data/csv/`, with processed cache files in `Data/cache/`.

## � Model Architecture

- **Graph Convolutional Network (GCN)** with multiple layers
- **Molecular features**: Atom types, bonds, molecular properties
- **Node features**: Atom properties, connectivity
- **Graph pooling** for molecular-level predictions

## 📈 Results

Training results, metrics, and visualizations are saved in the `results/` directory, organized by dataset.

## 📝 Documentation

Detailed documentation is available in the `docs/` directory.

## 🤝 Contributing

Contributions are welcome! Please follow the existing code structure when adding new features.

