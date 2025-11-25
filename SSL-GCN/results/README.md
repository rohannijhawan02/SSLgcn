# Training Results Directory

This directory contains CSV result files for each toxicity dataset after training.

## Directory Structure

After running `train_all_toxicities.py`, you'll find:

```
results/
├── overall_summary.csv          # Comparison of all datasets
├── overall_comparison.png       # Visual comparison plots
├── metrics_heatmap.png         # Heatmap of all metrics

├── NR-AhR/
│   ├── training_history.csv    # Epoch-by-epoch training metrics
│   ├── test_results.csv        # Final test evaluation
│   ├── summary.csv             # Dataset info and best results
│   └── training_curves.png     # Training visualization
│
├── NR-AR/
│   └── ... (same files)
│
└── ... (one folder per toxicity)
```

## CSV Files

### 1. training_history.csv
Records the training progress for each epoch:
- epoch, train_loss, train_accuracy, train_auc
- val_loss, val_accuracy, val_auc

### 2. test_results.csv
Final evaluation on the test set:
- accuracy, roc_auc, precision, recall, f1_score

### 3. summary.csv
Complete dataset information and results:
- Dataset statistics (sizes, class distribution)
- Best validation results
- Final test metrics

### 4. overall_summary.csv
Comparison table for all datasets with key metrics

## How to Generate Results

Run one of these commands:

```powershell
# Quick test (one dataset, 20 epochs)
python quick_test_results.py

# Train all datasets (12 datasets, up to 100 epochs each)
python train_all_toxicities.py

# Visualize results after training
python visualize_results.py
```

## Analyzing Results

You can open the CSV files in:
- Microsoft Excel
- Python (pandas): `pd.read_csv('results/NR-AhR/training_history.csv')`
- R: `read.csv('results/NR-AhR/training_history.csv')`
- Any CSV viewer

## What the Numbers Mean

- **Accuracy**: Percentage of correct predictions (0-1)
- **ROC-AUC**: Area under ROC curve, main metric for imbalanced data (0.5-1.0)
  - 0.5 = random guessing
  - 1.0 = perfect classifier
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were found
- **F1-Score**: Harmonic mean of precision and recall

## Next Steps

1. ✅ Review `overall_summary.csv` for quick comparison
2. ✅ Check individual dataset folders for detailed results
3. ✅ Use `visualize_results.py` to create plots
4. ✅ Analyze training curves to check for overfitting
5. ✅ Use best models from `checkpoints/` for predictions

For more information, see `RESULTS_GUIDE.md` or `CSV_RESULTS_SUMMARY.md` in the main directory.
