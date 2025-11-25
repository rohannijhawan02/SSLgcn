"""
Baseline ML Models Training for Toxicity Prediction
Implements KNN, NN, RF, SVM, and XGBoost using ECFP4 fingerprints

Based on the approach: "To establish the baseline performance, several commonly 
used ML algorithms, namely K-Nearest Neighbor (KNN), Neural Network (NN), 
Random Forest (RF), Support Vector Machine (SVM) and eXtreme Gradient Boosting 
(XGBoost) were tested. The compounds were encoded using the Extended Connectivity 
Fingerprints (ECFP4)."
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from data_preprocessing import ScaffoldSplitter


class ECFP4Encoder:
    """
    Extended Connectivity Fingerprints (ECFP4) encoder for molecular characterization.
    ECFP4 is a circular topological fingerprint designed for molecular characterization,
    similarity searching, and structure-activity modeling.
    """
    
    def __init__(self, radius=2, n_bits=2048):
        """
        Initialize ECFP4 encoder.
        
        Args:
            radius: Radius for circular fingerprint (2 for ECFP4)
            n_bits: Number of bits in fingerprint vector
        """
        self.radius = radius
        self.n_bits = n_bits
        
    def smiles_to_ecfp4(self, smiles):
        """
        Convert SMILES to ECFP4 fingerprint.
        
        Args:
            smiles: SMILES string
            
        Returns:
            numpy array: ECFP4 fingerprint or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 
                self.radius, 
                nBits=self.n_bits
            )
            return np.array(fp)
        except:
            return None
    
    def encode_dataset(self, smiles_list):
        """
        Encode a list of SMILES to ECFP4 fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array: Matrix of fingerprints (n_samples x n_bits)
            list: Valid indices
        """
        fingerprints = []
        valid_indices = []
        
        for idx, smiles in enumerate(smiles_list):
            fp = self.smiles_to_ecfp4(smiles)
            if fp is not None:
                fingerprints.append(fp)
                valid_indices.append(idx)
        
        return np.array(fingerprints), valid_indices


class BaselineModelTrainer:
    """
    Trainer for baseline ML models (KNN, NN, RF, SVM, XGBoost).
    """
    
    def __init__(self, 
                 toxicity_name,
                 data_dir='Data/csv',
                 results_dir='results/baseline_models',
                 models_dir='models/baseline_models',
                 seed=42):
        """
        Initialize baseline model trainer.
        
        Args:
            toxicity_name: Name of toxicity endpoint (e.g., 'NR-AhR')
            data_dir: Directory containing CSV files
            results_dir: Directory to save results
            models_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        self.toxicity_name = toxicity_name
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.seed = seed
        
        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, toxicity_name), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, toxicity_name), exist_ok=True)
        
        # Initialize encoder
        self.encoder = ECFP4Encoder(radius=2, n_bits=2048)
        
        # Model configurations
        self.models_config = {
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'NN': {
                'model': MLPClassifier(random_state=seed, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'RF': {
                'model': RandomForestClassifier(random_state=seed),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(random_state=seed, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=seed, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
        
        print(f"\n{'='*80}")
        print(f"Baseline Model Trainer Initialized")
        print(f"{'='*80}")
        print(f"Toxicity: {toxicity_name}")
        print(f"Models: {list(self.models_config.keys())}")
        print(f"Fingerprint: ECFP4 (radius=2, n_bits=2048)")
        print(f"{'='*80}\n")
    
    def load_and_prepare_data(self):
        """
        Load data and prepare train/val/test splits using scaffold splitting.
        
        Returns:
            dict: Dictionary containing splits with fingerprints and labels
        """
        print(f"Loading data for {self.toxicity_name}...")
        
        # Load CSV
        csv_path = os.path.join(self.data_dir, f"{self.toxicity_name}.csv")
        df = pd.read_csv(csv_path)
        
        # Remove rows with missing labels or SMILES
        df = df.dropna(subset=[self.toxicity_name, 'SMILES'])
        
        print(f"Total samples: {len(df)}")
        print(f"Class distribution: {df[self.toxicity_name].value_counts().to_dict()}")
        
        # Get SMILES and labels
        smiles_list = df['SMILES'].tolist()
        labels = df[self.toxicity_name].values.astype(int)
        
        # Scaffold split
        print("\nPerforming scaffold-based split (0.8:0.1:0.1)...")
        splitter = ScaffoldSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=self.seed)
        train_idx, val_idx, test_idx = splitter.scaffold_split(smiles_list, labels)
        
        print(f"Train size: {len(train_idx)}")
        print(f"Validation size: {len(val_idx)}")
        print(f"Test size: {len(test_idx)}")
        
        # Encode to ECFP4
        print("\nEncoding molecules to ECFP4 fingerprints...")
        all_fingerprints, valid_indices = self.encoder.encode_dataset(smiles_list)
        
        # Filter splits to only include valid molecules
        train_idx = [i for i in train_idx if i in valid_indices]
        val_idx = [i for i in val_idx if i in valid_indices]
        test_idx = [i for i in test_idx if i in valid_indices]
        
        # Create index mapping
        idx_to_fp_idx = {valid_indices[i]: i for i in range(len(valid_indices))}
        
        # Get fingerprints and labels for each split
        X_train = all_fingerprints[[idx_to_fp_idx[i] for i in train_idx]]
        y_train = labels[train_idx]
        
        X_val = all_fingerprints[[idx_to_fp_idx[i] for i in val_idx]]
        y_val = labels[val_idx]
        
        X_test = all_fingerprints[[idx_to_fp_idx[i] for i in test_idx]]
        y_test = labels[test_idx]
        
        print(f"Final train size: {len(X_train)}")
        print(f"Final validation size: {len(X_val)}")
        print(f"Final test size: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def train_model(self, model_name, data):
        """
        Train a single model with hyperparameter optimization.
        
        Args:
            model_name: Name of the model ('KNN', 'NN', 'RF', 'SVM', 'XGBoost')
            data: Dictionary containing train/val/test data
            
        Returns:
            dict: Trained model and results
        """
        print(f"\n{'-'*80}")
        print(f"Training {model_name} for {self.toxicity_name}")
        print(f"{'-'*80}")
        
        # Get model configuration
        config = self.models_config[model_name]
        base_model = config['model']
        param_grid = config['params']
        
        # Combine train and val for grid search (as per paper methodology)
        X_train_val = np.vstack([data['X_train'], data['X_val']])
        y_train_val = np.concatenate([data['y_train'], data['y_val']])
        
        # Grid search with cross-validation
        print(f"Performing grid search with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_val, y_train_val)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on validation set
        y_val_pred = best_model.predict(data['X_val'])
        y_val_proba = best_model.predict_proba(data['X_val'])[:, 1]
        
        val_auc = roc_auc_score(data['y_val'], y_val_proba)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            data['y_val'], y_val_pred, average='binary'
        )
        
        print(f"\nValidation Results:")
        print(f"  ROC-AUC: {val_auc:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}")
        print(f"  Recall: {val_rec:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        
        # Evaluate on test set
        y_test_pred = best_model.predict(data['X_test'])
        y_test_proba = best_model.predict_proba(data['X_test'])[:, 1]
        
        test_auc = roc_auc_score(data['y_test'], y_test_proba)
        test_acc = accuracy_score(data['y_test'], y_test_pred)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            data['y_test'], y_test_pred, average='binary'
        )
        
        print(f"\nTest Results:")
        print(f"  ROC-AUC: {test_auc:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_prec:.4f}")
        print(f"  Recall: {test_rec:.4f}")
        print(f"  F1-Score: {test_f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(data['y_test'], y_test_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'model': best_model,
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'val_results': {
                'auc': val_auc,
                'accuracy': val_acc,
                'precision': val_prec,
                'recall': val_rec,
                'f1': val_f1
            },
            'test_results': {
                'auc': test_auc,
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1': test_f1,
                'predictions': y_test_pred.tolist(),
                'probabilities': y_test_proba.tolist(),
                'true_labels': data['y_test'].tolist(),
                'confusion_matrix': cm.tolist()
            }
        }
    
    def save_model(self, model_name, model_dict):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model
            model_dict: Dictionary containing model and metadata
        """
        model_path = os.path.join(
            self.models_dir, 
            self.toxicity_name, 
            f"{model_name}_model.pkl"
        )
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model saved to: {model_path}")
    
    def save_results(self, model_name, results):
        """
        Save training results to disk.
        
        Args:
            model_name: Name of the model
            results: Dictionary containing results
        """
        # Save detailed results as JSON
        results_path = os.path.join(
            self.results_dir,
            self.toxicity_name,
            f"{model_name}_results.json"
        )
        
        # Prepare results for JSON (exclude model object)
        results_to_save = {
            'model_name': results['model_name'],
            'best_params': results['best_params'],
            'cv_score': results['cv_score'],
            'val_results': results['val_results'],
            'test_results': results['test_results']
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        
        print(f"Results saved to: {results_path}")
        
        # Save test predictions as CSV
        predictions_df = pd.DataFrame({
            'true_label': results['test_results']['true_labels'],
            'predicted_label': results['test_results']['predictions'],
            'probability': results['test_results']['probabilities']
        })
        
        predictions_path = os.path.join(
            self.results_dir,
            self.toxicity_name,
            f"{model_name}_predictions.csv"
        )
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
    
    def train_all_models(self):
        """
        Train all baseline models for this toxicity endpoint.
        
        Returns:
            dict: Results for all models
        """
        print(f"\n{'='*80}")
        print(f"Training All Baseline Models for {self.toxicity_name}")
        print(f"{'='*80}\n")
        
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Train each model
        all_results = {}
        
        for model_name in self.models_config.keys():
            try:
                results = self.train_model(model_name, data)
                all_results[model_name] = results
                
                # Save model and results
                self.save_model(model_name, results)
                self.save_results(model_name, results)
                
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create summary
        self.create_summary(all_results)
        
        return all_results
    
    def create_summary(self, all_results):
        """
        Create summary of all model results.
        
        Args:
            all_results: Dictionary of results for all models
        """
        print(f"\n{'='*80}")
        print(f"Summary for {self.toxicity_name}")
        print(f"{'='*80}\n")
        
        summary_data = []
        
        for model_name, results in all_results.items():
            summary_data.append({
                'Model': model_name,
                'CV_ROC_AUC': results['cv_score'],
                'Val_ROC_AUC': results['val_results']['auc'],
                'Val_Accuracy': results['val_results']['accuracy'],
                'Val_F1': results['val_results']['f1'],
                'Test_ROC_AUC': results['test_results']['auc'],
                'Test_Accuracy': results['test_results']['accuracy'],
                'Test_Precision': results['test_results']['precision'],
                'Test_Recall': results['test_results']['recall'],
                'Test_F1': results['test_results']['f1']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_ROC_AUC', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = os.path.join(
            self.results_dir,
            self.toxicity_name,
            'summary.csv'
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        
        # Identify best model
        best_model = summary_df.iloc[0]['Model']
        best_auc = summary_df.iloc[0]['Test_ROC_AUC']
        
        print(f"\n{'='*80}")
        print(f"Best Model: {best_model} (Test ROC-AUC: {best_auc:.4f})")
        print(f"{'='*80}\n")


def main():
    """
    Main function to train baseline models for a specific toxicity endpoint.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline ML models')
    parser.add_argument('--toxicity', type=str, required=True,
                       help='Toxicity endpoint name (e.g., NR-AhR)')
    parser.add_argument('--data_dir', type=str, default='Data/csv',
                       help='Directory containing CSV files')
    parser.add_argument('--results_dir', type=str, default='results/baseline_models',
                       help='Directory to save results')
    parser.add_argument('--models_dir', type=str, default='models/baseline_models',
                       help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BaselineModelTrainer(
        toxicity_name=args.toxicity,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        seed=args.seed
    )
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
