"""
Quick Start Script for Baseline ML Models
Demonstrates basic usage of the baseline model training and prediction
"""

import os
import sys

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")

def main():
    print_header("BASELINE ML MODELS - QUICK START GUIDE")
    
    print("This script demonstrates how to use the baseline ML models.\n")
    print("Available models:")
    print("  1. K-Nearest Neighbor (KNN)")
    print("  2. Neural Network (NN)")
    print("  3. Random Forest (RF)")
    print("  4. Support Vector Machine (SVM)")
    print("  5. eXtreme Gradient Boosting (XGBoost)")
    
    print("\n" + "-"*80)
    print("TRAINING OPTIONS")
    print("-"*80 + "\n")
    
    print("Option 1: Train single model for single toxicity")
    print("  Example: python src/train_model_knn.py --toxicity NR-AhR\n")
    
    print("Option 2: Train single model for all toxicities (12 models)")
    print("  Example: python src/train_model_knn.py\n")
    
    print("Option 3: Train all 5 models for single toxicity")
    print("  Example: python src/train_baseline_models.py --toxicity NR-AhR\n")
    
    print("Option 4: Train ALL models (60 models = 5 models × 12 toxicities)")
    print("  Example: python src/train_all_baseline_models.py\n")
    
    print("\n" + "-"*80)
    print("PREDICTION OPTIONS")
    print("-"*80 + "\n")
    
    print("Option 1: Predict single SMILES")
    print('  Example: python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"\n')
    
    print("Option 2: Predict from CSV file")
    print("  Example: python src/predict_baseline_models.py --input test.csv --output predictions.csv\n")
    
    print("Option 3: Predict with specific model")
    print('  Example: python src/predict_baseline_models.py --smiles "CCO" --model XGBoost\n')
    
    print("\n" + "-"*80)
    print("RECOMMENDED WORKFLOW")
    print("-"*80 + "\n")
    
    print("Step 1: Install dependencies")
    print("  pip install -r requirements.txt\n")
    
    print("Step 2: Train a single model to test (fastest)")
    print("  python src/train_model_knn.py --toxicity NR-AhR\n")
    
    print("Step 3: Check results")
    print("  results/baseline_models/NR-AhR/summary.csv\n")
    
    print("Step 4: Make predictions")
    print('  python src/predict_baseline_models.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"\n')
    
    print("Step 5: Train all models (when ready)")
    print("  python src/train_all_baseline_models.py\n")
    
    print("\n" + "-"*80)
    print("EXPECTED OUTPUT STRUCTURE")
    print("-"*80 + "\n")
    
    print("models/baseline_models/")
    print("  ├── NR-AhR/")
    print("  │   ├── KNN_model.pkl")
    print("  │   ├── NN_model.pkl")
    print("  │   ├── RF_model.pkl")
    print("  │   ├── SVM_model.pkl")
    print("  │   └── XGBoost_model.pkl")
    print("  └── ... (other toxicities)\n")
    
    print("results/baseline_models/")
    print("  ├── overall_summary.csv")
    print("  ├── NR-AhR/")
    print("  │   ├── summary.csv")
    print("  │   ├── KNN_results.json")
    print("  │   └── ... (other models)")
    print("  └── ... (other toxicities)\n")
    
    print("\n" + "-"*80)
    print("INDIVIDUAL MODEL TRAINING SCRIPTS")
    print("-"*80 + "\n")
    
    models = ['knn', 'nn', 'rf', 'svm', 'xgboost']
    for model in models:
        print(f"Train {model.upper()}:")
        print(f"  All toxicities: python src/train_model_{model}.py")
        print(f"  Single toxicity: python src/train_model_{model}.py --toxicity NR-AhR\n")
    
    print("\n" + "-"*80)
    print("DATASET INFORMATION")
    print("-"*80 + "\n")
    
    print("Toxicity Endpoints (12 total):")
    print("  Nuclear Receptors (NR):")
    print("    - NR-AhR, NR-AR, NR-AR-LBD, NR-Aromatase")
    print("    - NR-ER, NR-ER-LBD, NR-PPAR-gamma")
    print("  Stress Response (SR):")
    print("    - SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53\n")
    
    print("Data Split:")
    print("  - Train: 80%")
    print("  - Validation: 10%")
    print("  - Test: 10%")
    print("  - Method: Scaffold-based splitting\n")
    
    print("Feature Encoding:")
    print("  - Type: ECFP4 (Extended Connectivity Fingerprints)")
    print("  - Radius: 2")
    print("  - Bits: 2048\n")
    
    print("\n" + "-"*80)
    print("PERFORMANCE TIPS")
    print("-"*80 + "\n")
    
    print("Training Speed (fastest to slowest):")
    print("  1. KNN (instant training, distance-based)")
    print("  2. Random Forest (parallel tree building)")
    print("  3. XGBoost (optimized gradient boosting)")
    print("  4. Neural Network (iterative optimization)")
    print("  5. SVM (quadratic optimization, slowest for large data)\n")
    
    print("Typical Performance (by ROC-AUC):")
    print("  - XGBoost: Often best (0.85-0.95)")
    print("  - Random Forest: Consistent (0.80-0.90)")
    print("  - SVM: Good with scaling (0.80-0.88)")
    print("  - Neural Network: Variable (0.75-0.90)")
    print("  - KNN: Baseline (0.70-0.85)\n")
    
    print("\n" + "-"*80)
    print("EXAMPLE: COMPLETE WORKFLOW")
    print("-"*80 + "\n")
    
    print("# 1. Train XGBoost for NR-AhR")
    print("python src/train_model_xgboost.py --toxicity NR-AhR\n")
    
    print("# 2. Check results")
    print("cat results/baseline_models/NR-AhR/XGBoost_results.json\n")
    
    print("# 3. Make prediction")
    print('python src/predict_baseline_models.py \\')
    print('  --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" \\')
    print('  --toxicity NR-AhR \\')
    print('  --model XGBoost\n')
    
    print("\n" + "-"*80)
    print("TROUBLESHOOTING")
    print("-"*80 + "\n")
    
    print("Issue: ImportError for xgboost")
    print("Solution: pip install xgboost\n")
    
    print("Issue: ImportError for rdkit")
    print("Solution: conda install -c conda-forge rdkit\n")
    
    print("Issue: Out of memory during training")
    print("Solution: Train one toxicity at a time, or use smaller parameter grids\n")
    
    print("Issue: Training is too slow")
    print("Solution: Start with KNN or RF (fastest), reduce parameter grid size\n")
    
    print("\n" + "="*80)
    print("For more details, see: docs/BASELINE_MODELS_README.md")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
