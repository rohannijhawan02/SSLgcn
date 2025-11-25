# Baseline Models - Quick Reference Card

## 🚀 Quick Commands

### Setup
```bash
pip install xgboost                          # Install XGBoost
python src/test_baseline_models.py          # Verify installation
python src/baseline_models_quickstart.py    # View guide
```

### Train Single Model
```bash
python src/train_model_knn.py --toxicity NR-AhR        # KNN
python src/train_model_nn.py --toxicity NR-AhR         # Neural Network
python src/train_model_rf.py --toxicity NR-AhR         # Random Forest
python src/train_model_svm.py --toxicity NR-AhR        # SVM
python src/train_model_xgboost.py --toxicity NR-AhR    # XGBoost
```

### Train All Models for One Toxicity
```bash
python src/train_baseline_models.py --toxicity NR-AhR
```

### Train All 60 Models
```bash
python src/train_all_baseline_models.py
```

### Make Predictions
```bash
# Single SMILES
python src/predict_baseline_models.py --smiles "CCO"

# From file
python src/predict_baseline_models.py --input test.csv --output pred.csv

# Specific model
python src/predict_baseline_models.py --smiles "CCO" --model XGBoost --toxicity NR-AhR
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `train_model_knn.py` | Train KNN models |
| `train_model_nn.py` | Train Neural Network |
| `train_model_rf.py` | Train Random Forest |
| `train_model_svm.py` | Train SVM |
| `train_model_xgboost.py` | Train XGBoost |
| `train_baseline_models.py` | Core framework |
| `train_all_baseline_models.py` | Train all 60 models |
| `predict_baseline_models.py` | Make predictions |
| `test_baseline_models.py` | System test |
| `baseline_models_quickstart.py` | Quick start guide |

## 📊 Models Overview

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| KNN | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Quick baseline |
| Random Forest | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Robust results |
| XGBoost | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Best performance |
| Neural Network | ⚡⚡ | ⭐⭐⭐⭐ | Complex patterns |
| SVM | ⚡ | ⭐⭐⭐⭐ | High-dim data |

## 🎯 Toxicity Endpoints

**NR (Nuclear Receptors):**
- NR-AhR, NR-AR, NR-AR-LBD, NR-Aromatase
- NR-ER, NR-ER-LBD, NR-PPAR-gamma

**SR (Stress Response):**
- SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

## 📈 Training Time Estimates

| Task | Time |
|------|------|
| 1 model, 1 toxicity | 1-30 min |
| 1 model, all toxicities | 15-300 min |
| All 5 models, 1 toxicity | 20-120 min |
| All 60 models | 5-10 hours |

## 🔍 Output Locations

**Models:**
```
models/baseline_models/{toxicity}/{model}_model.pkl
```

**Results:**
```
results/baseline_models/{toxicity}/summary.csv
results/baseline_models/overall_summary.csv
```

## 📖 Documentation

| Document | Location |
|----------|----------|
| Setup Guide | `SETUP_GUIDE.md` |
| Full README | `docs/BASELINE_MODELS_README.md` |
| Implementation | `docs/BASELINE_MODELS_IMPLEMENTATION_SUMMARY.md` |
| Complete Summary | `BASELINE_MODELS_COMPLETE.md` |

## 💡 Tips

1. **Start small**: Test with KNN on 1 toxicity first
2. **Use XGBoost**: Often gives best results
3. **Check results**: View `summary.csv` files
4. **Compare models**: Use `overall_summary.csv`
5. **Save time**: Train overnight for all models

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| XGBoost not found | `pip install xgboost` |
| RDKit not found | `conda install -c conda-forge rdkit` |
| Tests failing | Restart terminal |
| Data not found | Check `Data/csv/` folder |

## 🎓 ECFP4 Details

- **Type**: Extended Connectivity Fingerprints
- **Radius**: 2 (ECFP4)
- **Bits**: 2048
- **Library**: RDKit

## 📊 Evaluation Metrics

- **ROC-AUC** (primary)
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔄 Workflow

1. Install dependencies
2. Run test
3. Train 1 model (test)
4. Check results
5. Train all models
6. Compare performance
7. Make predictions

---

**Total Models**: 60 (5 types × 12 endpoints)  
**Implementation**: Word-by-word as per paper  
**Status**: Production ready ✅
