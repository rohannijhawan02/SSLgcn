# Installation and Setup Guide for Baseline ML Models

## Quick Setup Instructions

Follow these steps to set up the baseline ML models environment:

### Step 1: Install XGBoost

The baseline models require XGBoost which may not be installed yet. Install it using:

```bash
pip install xgboost
```

Or if using conda:
```bash
conda install -c conda-forge xgboost
```

### Step 2: Verify Installation

Run the test script to verify all components are working:

```bash
python src/test_baseline_models.py
```

All tests should pass (✓) if the setup is correct.

### Step 3: View Quick Start Guide

```bash
python src/baseline_models_quickstart.py
```

This will show you all available commands and usage examples.

### Step 4: Train Your First Model (Quick Test)

Train KNN for a single toxicity endpoint (fastest option for testing):

```bash
python src/train_model_knn.py --toxicity NR-AhR
```

This should complete in under 5 minutes.

### Step 5: Make Your First Prediction

After training, test prediction:

```bash
python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" --toxicity NR-AhR --model KNN
```

## Complete Installation

To install all required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas>=1.3.0
- numpy>=1.21.0
- scikit-learn>=0.24.0
- rdkit>=2022.3.1
- xgboost>=1.5.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- torch>=1.10.0
- dgl>=0.9.0

## Troubleshooting

### Issue: "No module named 'xgboost'"

**Solution:**
```bash
pip install xgboost
```

### Issue: "No module named 'rdkit'"

**Solution (use conda, pip may not work):**
```bash
conda install -c conda-forge rdkit
```

### Issue: Test still failing after installation

**Solution:** 
Restart your Python kernel/terminal and try again:
```bash
# Exit and restart terminal/PowerShell
python src/test_baseline_models.py
```

### Issue: "Data file not found"

**Solution:**
Ensure your data files are in the correct location:
```
Data/csv/NR-AhR.csv
Data/csv/NR-AR.csv
... (all toxicity CSVs)
```

## Training Options (After Setup)

### Quick Test (5 minutes):
```bash
python src/train_model_knn.py --toxicity NR-AhR
```

### Medium Test (30 minutes):
```bash
python src/train_baseline_models.py --toxicity NR-AhR
```
This trains all 5 models for one toxicity.

### Full Training (several hours):
```bash
python src/train_all_baseline_models.py
```
This trains all 60 models (5 models × 12 toxicities).

## Expected Output

After successful training, you should see:

```
models/baseline_models/
└── NR-AhR/
    ├── KNN_model.pkl
    ├── NN_model.pkl
    ├── RF_model.pkl
    ├── SVM_model.pkl
    └── XGBoost_model.pkl

results/baseline_models/
└── NR-AhR/
    ├── summary.csv
    ├── KNN_results.json
    └── ... (other results)
```

## Next Steps

After setup is complete:

1. **Read the documentation:**
   - `docs/BASELINE_MODELS_README.md`
   - `docs/BASELINE_MODELS_IMPLEMENTATION_SUMMARY.md`

2. **Run quick start guide:**
   ```bash
   python src/baseline_models_quickstart.py
   ```

3. **Train models:**
   Start with a single model to verify everything works, then scale up.

4. **Make predictions:**
   Use trained models to predict toxicity for new compounds.

## Support

If you encounter issues not covered here:
1. Check that all dependencies are installed correctly
2. Verify Python version is 3.8 or higher
3. Ensure data files are in the correct location
4. Review error messages carefully - they usually indicate what's missing

## Summary

The baseline ML models are now ready to use! Follow the steps above to:
✓ Install dependencies
✓ Verify installation
✓ Train models
✓ Make predictions

Happy modeling!
