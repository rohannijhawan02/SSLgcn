# Bayesian Hyperparameter Optimization Guide

## Overview

This guide explains how to use Bayesian optimization for hyperparameter tuning in the SSL-GCN project, following the methodology described in the paper:

- **Algorithm**: Tree-structured Parzen Estimator (TPE) via Optuna
- **Number of trials**: 32 (as per paper)
- **Early stopping patience**: 30 epochs (as per paper)
- **Optimization metric**: Validation ROC-AUC

## What Gets Optimized?

The Bayesian optimization searches over the following hyperparameters:

| Hyperparameter | Search Space | Description |
|----------------|--------------|-------------|
| `num_layers` | [2, 3, 4] | Number of GCN layers |
| `hidden_dim1` | [32, 64, 128] | Hidden dimension for layer 1 |
| `hidden_dim2` | [64, 128, 256] | Hidden dimension for layer 2 |
| `hidden_dim3` | [128, 256, 512] | Hidden dimension for layer 3 |
| `classifier_hidden` | [64, 128, 256] | Classifier hidden layer size |
| `dropout` | [0.1, 0.5] | Dropout rate |
| `learning_rate` | [1e-4, 1e-2] | Learning rate (log scale) |
| `weight_decay` | [1e-6, 1e-3] | L2 regularization (log scale) |

## Installation

First, ensure you have Optuna installed:

```bash
pip install optuna
```

Or if using conda environment:

```bash
conda activate sslgcn
pip install optuna
```

## Usage

### Option 1: Tune a Single Dataset

```python
from src.bayesian_hyperparameter_tuning import BayesianHyperparameterTuner
import torch

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create tuner for a specific dataset
tuner = BayesianHyperparameterTuner(
    dataset_name='NR-AhR',
    device=device,
    n_trials=32,  # Number of optimization trials
    patience=30   # Early stopping patience
)

# Run optimization
results = tuner.optimize()

# Best hyperparameters are saved automatically to:
# models/tuning_results/NR-AhR/best_hyperparameters.json
```

### Option 2: Tune All Datasets

```python
from src.bayesian_hyperparameter_tuning import tune_all_datasets

# This will take a LONG time (hours to days)
results = tune_all_datasets(n_trials=32, device='cpu')
```

### Option 3: Quick Testing (Fewer Trials)

```python
# For testing/debugging, use fewer trials
tuner = BayesianHyperparameterTuner('NR-AhR', n_trials=5)
results = tuner.optimize()
```

### Option 4: Run from Command Line

```bash
# Navigate to project directory
cd c:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN

# Run Bayesian optimization for a single dataset
python src\bayesian_hyperparameter_tuning.py
```

## Using Optimized Hyperparameters

Once you've run the optimization, use the optimized hyperparameters when training:

### Method 1: Automatically Load Tuned Parameters

```python
from src.train_all_toxicities import train_single_toxicity, train_all_toxicities

# Train single dataset with tuned parameters
train_single_toxicity('NR-AhR', use_tuned_params=True)

# Train all datasets with tuned parameters
train_all_toxicities(use_tuned_params=True)
```

### Method 2: Manually Load and Inspect

```python
import json

# Load best hyperparameters
with open('models/tuning_results/NR-AhR/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

print("Best hyperparameters for NR-AhR:")
print(json.dumps(best_params, indent=2))
```

## Output Files

After optimization, the following files are created in `models/tuning_results/{dataset_name}/`:

1. **`best_hyperparameters.json`** - Best hyperparameters and their validation AUC
2. **`all_trials.csv`** - Results from all optimization trials
3. **`study.pkl`** - Complete Optuna study object (for advanced analysis)
4. **`summary.json`** - Summary of optimization results
5. **`trial_{n}/best_model.pt`** - Best model checkpoint for each trial

## Example Output

```json
{
  "dataset": "NR-AhR",
  "best_trial": 15,
  "best_val_auc": 0.8456,
  "best_params": {
    "num_layers": 3,
    "hidden_dim1": 64,
    "hidden_dim2": 128,
    "hidden_dim3": 256,
    "classifier_hidden": 128,
    "dropout": 0.3,
    "learning_rate": 0.001245,
    "weight_decay": 0.00002341
  },
  "n_trials": 32,
  "timestamp": "2025-10-12 14:30:45"
}
```

## Complete Workflow

### Step 1: Run Bayesian Optimization (Once)

```python
from src.bayesian_hyperparameter_tuning import tune_all_datasets

# This finds the best hyperparameters for each dataset
# WARNING: This takes a LONG time (days)
results = tune_all_datasets(n_trials=32, device='cpu')
```

### Step 2: Train with Optimized Parameters

```python
from src.train_all_toxicities import train_all_toxicities

# Now train models using the optimized hyperparameters
results = train_all_toxicities(use_tuned_params=True)
```

### Step 3: Compare Results

```python
import pandas as pd

# Load results with default parameters
default_results = pd.read_csv('results/overall_summary.csv')

# Load results with tuned parameters  
tuned_results = pd.read_csv('results_tuned/overall_summary.csv')

# Compare
comparison = pd.DataFrame({
    'dataset': default_results['dataset'],
    'default_auc': default_results['test_roc_auc'],
    'tuned_auc': tuned_results['test_roc_auc'],
    'improvement': tuned_results['test_roc_auc'] - default_results['test_roc_auc']
})

print(comparison)
```

## Visualization

### Plot Optimization History

```python
import optuna
import pickle

# Load study
with open('models/tuning_results/NR-AhR/study.pkl', 'rb') as f:
    study = pickle.load(f)

# Plot optimization history
from optuna.visualization import plot_optimization_history, plot_param_importances

plot_optimization_history(study)
plot_param_importances(study)
```

## Time Estimates

Approximate time for 32 trials per dataset:

- **CPU (single core)**: 4-8 hours per dataset
- **CPU (multi-core)**: 2-4 hours per dataset  
- **GPU**: 1-2 hours per dataset

For all 12 datasets:
- **CPU**: 2-4 days
- **GPU**: 12-24 hours

## Tips and Best Practices

1. **Start with one dataset** to test your setup:
   ```python
   tuner = BayesianHyperparameterTuner('NR-AhR', n_trials=5)
   results = tuner.optimize()
   ```

2. **Use fewer trials for testing** (5-10 trials instead of 32)

3. **Run on GPU** if available to speed up optimization

4. **Run overnight** for full optimization (32 trials × 12 datasets)

5. **Check intermediate results**:
   ```python
   # View all trials so far
   import pandas as pd
   trials = pd.read_csv('models/tuning_results/NR-AhR/all_trials.csv')
   print(trials.sort_values('value', ascending=False).head(10))
   ```

6. **Resume interrupted optimization**:
   ```python
   # Optuna automatically resumes from saved study
   tuner = BayesianHyperparameterTuner('NR-AhR', n_trials=32)
   results = tuner.optimize()  # Continues from where it left off
   ```

## Comparison with Paper

The implementation follows the paper's methodology:

| Feature | Paper | Our Implementation |
|---------|-------|-------------------|
| Optimization algorithm | Bayesian (TPE) | ✅ Optuna TPE |
| Number of trials | 32 | ✅ 32 (configurable) |
| Early stopping patience | 30 epochs | ✅ 30 epochs |
| Selection metric | ROC-AUC | ✅ Validation ROC-AUC |
| Class imbalance handling | Yes | ✅ Class weights |

## Troubleshooting

### Issue: "Out of memory"
**Solution**: Reduce batch size or use CPU instead of GPU

### Issue: "Takes too long"
**Solution**: Reduce `n_trials` or use GPU

### Issue: "No tuned parameters found"
**Solution**: Run optimization first before training with `use_tuned_params=True`

### Issue: "Study already exists"
**Solution**: Delete existing study files or use a different study name

## Advanced Usage

### Custom Search Space

Modify the `objective` function in `bayesian_hyperparameter_tuning.py`:

```python
# Add custom hyperparameters
batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
activation = trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu'])
```

### Parallel Optimization

```python
# Run multiple trials in parallel
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=32, n_jobs=4)  # 4 parallel jobs
```

### Custom Pruning

```python
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

# More aggressive pruning
pruner = SuccessiveHalvingPruner()
study = optuna.create_study(direction='maximize', pruner=pruner)
```

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- TPE Algorithm: https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- SSL-GCN Paper: [Add paper reference]

## Summary

Bayesian hyperparameter optimization is now fully integrated into your SSL-GCN pipeline. The key advantages are:

1. ✅ **Automatic hyperparameter search** (no manual tuning needed)
2. ✅ **Follows paper methodology** (32 trials, ROC-AUC metric, 30 epoch patience)
3. ✅ **Easy to use** (single function call)
4. ✅ **Results are saved** and automatically loaded for training
5. ✅ **Better model performance** through optimized hyperparameters

Start with a small test, then scale up to full optimization!
