# Baseline ML Models for Toxicity Prediction

This module implements baseline machine learning models using ECFP4 fingerprints for molecular toxicity prediction, as described in the research methodology.

## Overview

To establish baseline performance, five commonly used ML algorithms were implemented:
- **K-Nearest Neighbor (KNN)**
- **Neural Network (NN)** - Multi-layer Perceptron
- **Random Forest (RF)**
- **Support Vector Machine (SVM)**
- **eXtreme Gradient Boosting (XGBoost)**

## Molecular Encoding

The compounds are encoded using **Extended Connectivity Fingerprints (ECFP4)**, which is a circular topological fingerprint designed for:
- Molecular characterization
- Similarity searching
- Structure-activity modeling

**ECFP4 Configuration:**
- Radius: 2 (ECFP4)
- Number of bits: 2048
- Generated using RDKit library

## Models Overview

### 1. K-Nearest Neighbor (KNN)
Finds the k nearest neighbors in the feature space and predicts based on majority voting.

**Optimized Hyperparameters:**
- `n_neighbors`: [3, 5, 7, 9, 11, 13, 15]
- `weights`: ['uniform', 'distance']
- `metric`: ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
- `algorithm`: ['auto', 'ball_tree', 'kd_tree', 'brute']

### 2. Neural Network (NN)
Multi-layer Perceptron with backpropagation for binary classification.

**Optimized Hyperparameters:**
- `hidden_layer_sizes`: [(64,), (128,), (256,), (64,32), (128,64), etc.]
- `activation`: ['relu', 'tanh', 'logistic']
- `alpha`: [0.0001, 0.001, 0.01]
- `learning_rate`: ['constant', 'adaptive']
- `max_iter`: [1000]

### 3. Random Forest (RF)
Ensemble of decision trees with bootstrap sampling and feature randomness.

**Optimized Hyperparameters:**
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [10, 20, 30, 40, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]
- `class_weight`: ['balanced', 'balanced_subsample', None]

### 4. Support Vector Machine (SVM)
Finds optimal hyperplane with maximum margin separation.

**Optimized Hyperparameters:**
- `C`: [0.1, 1, 10, 100, 1000]
- `kernel`: ['rbf', 'linear', 'poly', 'sigmoid']
- `gamma`: ['scale', 'auto', 0.001, 0.01, 0.1]
- `class_weight`: ['balanced', None]

**Note:** Features are standardized (zero mean, unit variance) for SVM.

### 5. XGBoost
Gradient boosting with regularization and advanced tree construction.

**Optimized Hyperparameters:**
- `n_estimators`: [100, 200, 300, 500]
- `max_depth`: [3, 5, 7, 9, 11]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 0.9, 1.0]
- `colsample_bytree`: [0.6, 0.8, 0.9, 1.0]
- `min_child_weight`: [1, 3, 5]
- `gamma`: [0, 0.1, 0.2]
- `reg_alpha`: [0, 0.1, 0.5]
- `reg_lambda`: [1, 1.5, 2]

## Data Splitting

**Scaffold-based splitting** is used to overcome data bias:
- Train: 80%
- Validation: 10%
- Test: 10%

This ensures structurally different molecules are in different subsets.

## Training Methodology

1. **Data Loading**: Load CSV files with SMILES and toxicity labels
2. **Scaffold Splitting**: Split data based on Bemis-Murcko scaffolds
3. **ECFP4 Encoding**: Convert SMILES to 2048-bit fingerprints
4. **Grid Search**: 5-fold cross-validation on train+val set
5. **Model Selection**: Select model with best ROC-AUC
6. **Test Evaluation**: Evaluate on held-out test set

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Individual Model Types

#### Train KNN for all toxicities:
```bash
python src/train_model_knn.py
```

#### Train KNN for specific toxicity:
```bash
python src/train_model_knn.py --toxicity NR-AhR
```

#### Train Neural Network:
```bash
python src/train_model_nn.py
```

#### Train Random Forest:
```bash
python src/train_model_rf.py
```

#### Train SVM:
```bash
python src/train_model_svm.py
```

#### Train XGBoost:
```bash
python src/train_model_xgboost.py
```

### Train All Models at Once

#### Train single toxicity with all 5 models:
```bash
python src/train_baseline_models.py --toxicity NR-AhR
```

#### Train all 60 models (5 models × 12 toxicities):
```bash
python src/train_all_baseline_models.py
```

This will train **60 different ML models** total!

## Making Predictions

### Predict single SMILES:
```bash
python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
```

### Predict from CSV file:
```bash
python src/predict_baseline_models.py --input test_molecules.csv --output predictions.csv
```

### Predict with specific model:
```bash
python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" --model XGBoost
```

### Predict specific toxicity:
```bash
python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" --toxicity NR-AhR
```

## Directory Structure

```
models/baseline_models/
├── NR-AhR/
│   ├── KNN_model.pkl
│   ├── NN_model.pkl
│   ├── RF_model.pkl
│   ├── SVM_model.pkl
│   └── XGBoost_model.pkl
├── NR-AR/
│   └── ...
└── ... (all 12 toxicity endpoints)

results/baseline_models/
├── overall_summary.csv
├── model_averages.csv
├── detailed_report.txt
├── NR-AhR/
│   ├── summary.csv
│   ├── KNN_results.json
│   ├── KNN_predictions.csv
│   ├── NN_results.json
│   ├── NN_predictions.csv
│   └── ... (all model results)
└── ... (all toxicity endpoints)
```

## Evaluation Metrics

All models are evaluated using:
- **ROC-AUC**: Primary metric (important for imbalanced datasets)
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Toxicity Endpoints

The models are trained for 12 different toxicity endpoints:

**Nuclear Receptors (NR):**
1. NR-AhR - Aryl hydrocarbon Receptor
2. NR-AR - Androgen Receptor
3. NR-AR-LBD - Androgen Receptor Ligand Binding Domain
4. NR-Aromatase - Aromatase
5. NR-ER - Estrogen Receptor
6. NR-ER-LBD - Estrogen Receptor Ligand Binding Domain
7. NR-PPAR-gamma - Peroxisome Proliferator-Activated Receptor Gamma

**Stress Response (SR):**
8. SR-ARE - Antioxidant Response Element
9. SR-ATAD5 - ATPase Family AAA Domain-Containing Protein 5
10. SR-HSE - Heat Shock Factor Response Element
11. SR-MMP - Mitochondrial Membrane Potential
12. SR-p53 - Tumor Suppressor p53

## Model Selection

After training all models:
1. Compare performance across all 60 models
2. Identify best model per toxicity endpoint
3. Select based on Test ROC-AUC score
4. Consider trade-offs (speed vs. accuracy)

## Performance Tips

### KNN
- Fast prediction
- No training required
- Good for small datasets
- Memory-intensive for large datasets

### Neural Network
- Can capture complex patterns
- Requires more data
- Longer training time
- Risk of overfitting

### Random Forest
- Robust to overfitting
- Handles high-dimensional data well
- Fast training and prediction
- Provides feature importance

### SVM
- Effective for high-dimensional data
- Memory-intensive
- Slower training with large datasets
- Requires feature scaling

### XGBoost
- Often best performance
- Handles imbalanced data well
- Provides feature importance
- Longer training time

## Comparison with GCN

After training baseline models, compare with GCN results:

```python
# Example comparison
baseline_best_auc = 0.85  # From baseline models
gcn_auc = 0.92  # From GCN model
improvement = (gcn_auc - baseline_best_auc) / baseline_best_auc * 100
print(f"GCN improvement over baseline: {improvement:.2f}%")
```

## Files Created

### Training Scripts:
- `train_baseline_models.py` - Main baseline training module
- `train_all_baseline_models.py` - Train all 60 models
- `train_model_knn.py` - KNN-specific training
- `train_model_nn.py` - Neural Network training
- `train_model_rf.py` - Random Forest training
- `train_model_svm.py` - SVM training
- `train_model_xgboost.py` - XGBoost training

### Prediction Script:
- `predict_baseline_models.py` - Make predictions with trained models

## Troubleshooting

### Memory Issues
If you encounter memory issues with large grid searches:
- Reduce parameter grid size
- Use fewer cross-validation folds
- Train models one at a time

### Long Training Times
To speed up training:
- Reduce parameter grid
- Use `n_jobs=-1` for parallel processing
- Train specific toxicities only
- Use fewer estimators (RF, XGBoost)

### Installation Issues
If XGBoost installation fails:
```bash
pip install xgboost --no-cache-dir
```

For RDKit issues, use conda:
```bash
conda install -c conda-forge rdkit
```

## Citation

If you use these baseline models in your research, please cite the original paper methodology regarding ECFP4 fingerprints and baseline ML algorithm comparison.

## References

- Extended Connectivity Fingerprints: Rogers & Hahn (2010)
- Scikit-learn: Pedregosa et al. (2011)
- XGBoost: Chen & Guestrin (2016)
- RDKit: Open-source cheminformatics toolkit

## License

This code is provided for research purposes. Please check individual library licenses.
