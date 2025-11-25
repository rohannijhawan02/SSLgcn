# GCN Model Implementation Guide

## Overview

This implementation provides a complete Graph Convolutional Network (GCN) architecture for molecular toxicity prediction based on the SSL-GCN framework.

## Architecture Description

The GCN model consists of two main parts:

### 1. **Encoder** (Graph Convolutional Network)
- Extracts and updates node representations through several graph convolutional layers
- Each layer is followed by a dropout layer for regularization
- The last layer merges all node features using:
  - **Max-pooling**: Captures the most prominent features
  - **Weighted sum**: Captures average features
  - Combined with learnable weights

### 2. **Classifier** (Multi-Layer Perceptron)
- Two-layer perceptron with:
  - Dropout layer for regularization
  - Batch normalization layer for stable training
- Computes the final toxicity prediction

## Model Components

### Files Created

1. **`main/model.py`** - Core GCN architecture
   - `GCNEncoder`: Graph convolutional encoder
   - `MLPClassifier`: Classification head
   - `GCNModel`: Complete model combining encoder and classifier
   - `create_gcn_model()`: Factory function for model creation

2. **`main/train.py`** - Training pipeline
   - `ToxicityTrainer`: Handles training, validation, and testing
   - ROC-AUC based evaluation (critical for imbalanced data)
   - Early stopping based on validation ROC-AUC
   - `run_multiple_trials()`: Runs 5 independent trials for reliable performance estimation

3. **`main/hyperparameter_tuning.py`** - Hyperparameter optimization
   - `HyperparameterTuner`: Grid search and random search
   - Validation ROC-AUC as selection metric
   - Saves all trial results for analysis

4. **`examples/train_gcn_example.py`** - Complete training examples
   - Single model training
   - Multiple trials training
   - Hyperparameter tuning
   - Training on all datasets

## Key Features

### Handling Imbalanced Data
The Tox21 dataset is highly imbalanced (toxic/non-toxic ratio ~1:17). The implementation addresses this with:

1. **ROC-AUC as primary metric**: More reliable than accuracy for imbalanced data
2. **Class weights in loss function**: Adjustable to balance classes
3. **Early stopping on validation ROC-AUC**: Prevents overfitting to majority class

### Hyperparameter Tuning
The model selects optimal hyperparameters through:
- Grid search or random search
- Validation ROC-AUC as selection criterion
- Configurable parameter space

### Multiple Trials
All experiments are repeated 5 times to:
- Observe result variability
- Obtain accurate performance measures through averaging
- Calculate standard deviations

## Usage

### 1. Quick Start - Train Single Model

```python
from main.model import create_gcn_model
from main.train import ToxicityTrainer

# Create model
model = create_gcn_model(
    in_feats=74,  # Node feature dimension
    hidden_dims=[64, 128, 256],
    num_layers=3,
    classifier_hidden=128,
    num_classes=2,
    dropout=0.2
)

# Create trainer
trainer = ToxicityTrainer(
    model=model,
    device='cuda',
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=10
)

# Train
history = trainer.train(train_loader, val_loader, num_epochs=100)

# Evaluate
test_metrics = trainer.evaluate(test_loader)
```

### 2. Multiple Trials Training

```python
from main.train import run_multiple_trials

def create_model():
    return create_gcn_model(
        in_feats=74,
        hidden_dims=[64, 128, 256],
        num_layers=3,
        classifier_hidden=128,
        num_classes=2,
        dropout=0.2
    )

# Run 5 trials
aggregated, all_metrics, histories = run_multiple_trials(
    model_fn=create_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_trials=5,
    num_epochs=100,
    device='cuda'
)

# Results include mean ± std for all metrics
print(f"ROC-AUC: {aggregated['auc']['mean']:.4f} ± {aggregated['auc']['std']:.4f}")
```

### 3. Hyperparameter Tuning

```python
from main.hyperparameter_tuning import tune_hyperparameters

# Random search (recommended for large parameter spaces)
results = tune_hyperparameters(
    train_loader=train_loader,
    val_loader=val_loader,
    in_feats=74,
    method='random',
    n_trials=20,
    device='cuda'
)

# Get best hyperparameters
best_params = results['best_params']
best_auc = results['best_auc']
```

### 4. Command Line Interface

```bash
# Train single model
python examples/train_gcn_example.py --mode single --dataset NR-AhR --epochs 100

# Train multiple trials
python examples/train_gcn_example.py --mode multiple --dataset NR-AhR --trials 5

# Hyperparameter tuning
python examples/train_gcn_example.py --mode tune --dataset NR-AhR --trials 20

# Train on all datasets
python examples/train_gcn_example.py --mode all --trials 5
```

## Hyperparameters

### Model Architecture
- `in_feats`: Input node feature dimension (typically 74 for molecular graphs)
- `hidden_dims`: List of hidden dimensions for encoder layers (e.g., [64, 128, 256])
- `num_layers`: Number of graph convolutional layers (typically 2-4)
- `classifier_hidden`: Hidden dimension for classifier (typically 128)
- `dropout`: Dropout rate for regularization (typically 0.2-0.5)

### Training
- `learning_rate`: Learning rate for Adam optimizer (typically 0.0001-0.01)
- `weight_decay`: L2 regularization weight (typically 1e-5 to 1e-3)
- `batch_size`: Batch size (typically 32-128)
- `num_epochs`: Maximum number of epochs (typically 100-200)
- `patience`: Early stopping patience (typically 10-20 epochs)

### Default Hyperparameters (After Tuning)
The implementation includes sensible defaults based on common values:

```python
default_params = {
    'hidden_dims': [64, 128, 256],
    'num_layers': 3,
    'classifier_hidden': 128,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'patience': 10
}
```

## Evaluation Metrics

### Primary Metric: ROC-AUC
- Used for hyperparameter selection
- Used for early stopping
- Most reliable for imbalanced datasets

### Additional Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall

## Results Format

### Single Trial Results
```python
{
    'accuracy': 0.85,
    'auc': 0.92,
    'precision': 0.80,
    'recall': 0.75,
    'f1': 0.77
}
```

### Multiple Trials Results
```python
{
    'auc': {
        'mean': 0.92,
        'std': 0.02,
        'min': 0.89,
        'max': 0.94,
        'all_values': [0.91, 0.93, 0.92, 0.90, 0.94]
    },
    # ... similar for other metrics
}
```

## Output Files

### Training
- `checkpoints/{dataset}/best_model.pt`: Best model checkpoint
- `results/{dataset}/trial_*/best_model.pt`: Individual trial checkpoints
- `results/{dataset}/aggregated_results.json`: Aggregated results across trials

### Hyperparameter Tuning
- `tuning_results/{dataset}/trial_*/best_model.pt`: Trial checkpoints
- `tuning_results/{dataset}/tuning_results.json`: All tuning results

## Model Architecture Details

### GCN Encoder
```
Input: Molecular graph with node features (N x D)
├─ GraphConv Layer 1: D → 64
│  ├─ ReLU activation
│  └─ Dropout (0.2)
├─ GraphConv Layer 2: 64 → 128
│  ├─ ReLU activation
│  └─ Dropout (0.2)
├─ GraphConv Layer 3: 128 → 256
│  ├─ ReLU activation
│  └─ Dropout (0.2)
└─ Pooling: Max-pooling + Weighted Sum → 256
```

### MLP Classifier
```
Input: Graph representation (256)
├─ Linear Layer 1: 256 → 128
│  ├─ Batch Normalization
│  ├─ ReLU activation
│  └─ Dropout (0.2)
└─ Linear Layer 2: 128 → 2
   └─ Output: Logits (toxic/non-toxic)
```

## Best Practices

1. **Always use ROC-AUC** for model selection with imbalanced data
2. **Run multiple trials** (5 recommended) for reliable performance estimates
3. **Use early stopping** to prevent overfitting
4. **Tune hyperparameters** for each dataset separately
5. **Monitor validation metrics** during training
6. **Save checkpoints** regularly

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Reduce model size (hidden dimensions)
   - Use gradient accumulation

2. **Overfitting**
   - Increase dropout rate
   - Increase weight decay
   - Use more data augmentation
   - Reduce model capacity

3. **Underfitting**
   - Increase model capacity
   - Reduce dropout rate
   - Train for more epochs
   - Increase learning rate

4. **Training is slow**
   - Use GPU if available
   - Increase batch size
   - Use mixed precision training
   - Reduce validation frequency

## References

1. Original SSL-GCN paper
2. DGL Documentation: https://docs.dgl.ai/
3. PyTorch Documentation: https://pytorch.org/docs/

## Next Steps

After implementing the basic GCN model, consider:

1. **Self-Supervised Learning**: Add contrastive learning objectives
2. **Data Augmentation**: Implement molecular graph augmentation
3. **Ensemble Methods**: Combine multiple models
4. **Attention Mechanisms**: Add graph attention layers
5. **Interpretability**: Implement attention visualization and feature importance

---

For more information, see the documentation in the `docs/` directory.
