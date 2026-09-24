"""
Test Script for Baseline ML Models
Verifies that all components are working correctly
"""

import sys
import os

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    try:
        from rdkit import Chem
        print("  ✓ rdkit")
    except ImportError as e:
        print(f"  ✗ rdkit: {e}")
        return False
    
    try:
        from xgboost import XGBClassifier
        print("  ✓ xgboost")
    except ImportError as e:
        print(f"  ✗ xgboost: {e}")
        return False
    
    return True


def test_ecfp4_encoding():
    """Test ECFP4 encoding functionality."""
    print("\nTesting ECFP4 encoding...")
    
    try:
        from train_baseline_models import ECFP4Encoder
        
        encoder = ECFP4Encoder(radius=2, n_bits=2048)
        
        # Test valid SMILES
        test_smiles = "CCO"
        fp = encoder.smiles_to_ecfp4(test_smiles)
        
        if fp is not None and len(fp) == 2048:
            print(f"  ✓ ECFP4 encoding works (SMILES: {test_smiles}, fingerprint length: {len(fp)})")
        else:
            print(f"  ✗ ECFP4 encoding failed")
            return False
        
        # Test invalid SMILES
        invalid_smiles = "INVALID"
        fp_invalid = encoder.smiles_to_ecfp4(invalid_smiles)
        
        if fp_invalid is None:
            print(f"  ✓ Invalid SMILES handling works")
        else:
            print(f"  ✗ Invalid SMILES not detected")
            return False
        
        # Test batch encoding
        smiles_list = ["CCO", "CC", "CCC"]
        fps, valid_idx = encoder.encode_dataset(smiles_list)
        
        if fps.shape[0] == len(smiles_list) and fps.shape[1] == 2048:
            print(f"  ✓ Batch encoding works ({fps.shape[0]} molecules)")
        else:
            print(f"  ✗ Batch encoding failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ ECFP4 encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        # Check if data files exist
        data_file = "Data/csv/NR-AhR.csv"
        if not os.path.exists(data_file):
            print(f"  ⚠ Data file not found: {data_file}")
            print(f"    This is expected if data is not yet downloaded")
            return True
        
        df = pd.read_csv(data_file)
        print(f"  ✓ Data loaded successfully ({len(df)} samples)")
        
        # Check required columns
        if 'SMILES' in df.columns and 'NR-AhR' in df.columns:
            print(f"  ✓ Required columns present")
        else:
            print(f"  ✗ Missing required columns")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data loading test failed: {e}")
        return False


def test_scaffold_splitting():
    """Test scaffold splitting functionality."""
    print("\nTesting scaffold splitting...")
    
    try:
        from data_preprocessing import ScaffoldSplitter
        
        splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        
        # Test with sample SMILES
        test_smiles = ["CCO", "CC", "CCC", "CCCC", "CCCCC", 
                       "c1ccccc1", "c1ccccc1C", "c1ccccc1CC"]
        test_labels = [0, 1, 0, 1, 0, 1, 0, 1]
        
        train_idx, val_idx, test_idx = splitter.scaffold_split(test_smiles, test_labels)
        
        total = len(train_idx) + len(val_idx) + len(test_idx)
        
        if total == len(test_smiles):
            print(f"  ✓ Scaffold splitting works")
            print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        else:
            print(f"  ✗ Scaffold splitting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Scaffold splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_classes():
    """Test that model trainer classes can be instantiated."""
    print("\nTesting model trainer classes...")
    
    try:
        from train_baseline_models import BaselineModelTrainer
        
        # Test base trainer
        trainer = BaselineModelTrainer(
            toxicity_name='NR-AhR',
            data_dir='Data/csv',
            results_dir='results/baseline_models',
            models_dir='models/baseline_models',
            seed=42
        )
        print(f"  ✓ BaselineModelTrainer instantiated")
        
        # Check encoder
        if hasattr(trainer, 'encoder'):
            print(f"  ✓ ECFP4Encoder integrated")
        
        # Check model configs
        if len(trainer.models_config) == 5:
            print(f"  ✓ All 5 model configurations present")
            for model_name in ['KNN', 'NN', 'RF', 'SVM', 'XGBoost']:
                if model_name in trainer.models_config:
                    print(f"    - {model_name} configuration found")
        else:
            print(f"  ✗ Missing model configurations")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist or can be created."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'Data/csv',
        'src',
        'docs'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path} exists")
        else:
            print(f"  ⚠ {dir_path} not found")
    
    # Test creating output directories
    try:
        os.makedirs('models/baseline_models', exist_ok=True)
        os.makedirs('results/baseline_models', exist_ok=True)
        print(f"  ✓ Output directories created")
    except Exception as e:
        print(f"  ✗ Cannot create output directories: {e}")
        return False
    
    return True


def test_files_exist():
    """Test that all created files exist."""
    print("\nTesting created files...")
    
    required_files = [
        'src/train_baseline_models.py',
        'src/train_all_baseline_models.py',
        'src/train_model_knn.py',
        'src/train_model_nn.py',
        'src/train_model_rf.py',
        'src/train_model_svm.py',
        'src/train_model_xgboost.py',
        'src/predict_baseline_models.py',
        'src/baseline_models_quickstart.py',
        'docs/BASELINE_MODELS_README.md',
        'docs/BASELINE_MODELS_IMPLEMENTATION_SUMMARY.md',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BASELINE ML MODELS - SYSTEM TEST")
    print("="*80)
    
    tests = [
        ("Files", test_files_exist),
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("ECFP4 Encoding", test_ecfp4_encoding),
        ("Scaffold Splitting", test_scaffold_splitting),
        ("Model Classes", test_model_classes),
        ("Data Loading", test_data_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Install any missing dependencies: pip install -r requirements.txt")
        print("  2. View quick start: python src/baseline_models_quickstart.py")
        print("  3. Train a test model: python src/train_model_knn.py --toxicity NR-AhR")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Install RDKit: conda install -c conda-forge rdkit")
        print("  - Ensure data files are in Data/csv/ directory")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
