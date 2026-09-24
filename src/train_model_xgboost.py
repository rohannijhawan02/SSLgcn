"""
eXtreme Gradient Boosting (XGBoost) Model Training for Toxicity Prediction
Uses ECFP4 fingerprints for molecular encoding

Based on: "eXtreme Gradient Boosting (XGBoost) was tested. The compounds were 
encoded using the Extended Connectivity Fingerprints (ECFP4), which is a 
circular topological fingerprint designed for molecular characterization, 
similarity searching, and structure-activity modeling."
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from train_baseline_models import BaselineModelTrainer, ECFP4Encoder
from data_preprocessing import ScaffoldSplitter


class XGBoostTrainer(BaselineModelTrainer):
    """
    Specialized trainer for XGBoost classifier.
    """
    
    def __init__(self, toxicity_name, **kwargs):
        super().__init__(toxicity_name, **kwargs)
        
        # XGBoost-specific configuration
        self.model_name = 'XGBoost'
        self.param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, 11],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        print(f"\n{'='*80}")
        print(f"eXtreme Gradient Boosting (XGBoost) Trainer Initialized")
        print(f"{'='*80}")
        print(f"Toxicity: {toxicity_name}")
        print(f"Fingerprint: ECFP4 (radius=2, n_bits=2048)")
        print(f"Hyperparameters to optimize:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        print(f"{'='*80}\n")
    
    def train_xgboost(self):
        """
        Train XGBoost model with hyperparameter optimization.
        
        Returns:
            dict: Trained model and results
        """
        print(f"\n{'-'*80}")
        print(f"Training XGBoost for {self.toxicity_name}")
        print(f"{'-'*80}")
        
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Combine train and val for grid search
        X_train_val = np.vstack([data['X_train'], data['X_val']])
        y_train_val = np.concatenate([data['y_train'], data['y_val']])
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = np.sum(y_train_val == 0)
        pos_count = np.sum(y_train_val == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        print(f"\nClass distribution:")
        print(f"  Negative samples: {neg_count}")
        print(f"  Positive samples: {pos_count}")
        print(f"  Scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Create XGBoost classifier
        xgb = XGBClassifier(
            random_state=self.seed,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        print(f"\nPerforming grid search with 5-fold cross-validation...")
        print("Note: XGBoost training with extensive grid search may take time...")
        
        grid_search = GridSearchCV(
            xgb,
            self.param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train_val, y_train_val)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Print model info
        print(f"\nBest XGBoost Configuration:")
        print(f"  Number of estimators: {best_model.n_estimators}")
        print(f"  Max depth: {best_model.max_depth}")
        print(f"  Learning rate: {best_model.learning_rate}")
        print(f"  Subsample: {best_model.subsample}")
        print(f"  Colsample bytree: {best_model.colsample_bytree}")
        
        # Feature importance
        feature_importance = best_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        print(f"\nTop 10 Most Important ECFP4 Bits:")
        for i, idx in enumerate(top_features_idx, 1):
            print(f"  {i}. Bit {idx}: {feature_importance[idx]:.6f}")
        
        # Evaluate on validation set
        print(f"\n{'-'*80}")
        print("Validation Set Evaluation")
        print(f"{'-'*80}")
        
        y_val_pred = best_model.predict(data['X_val'])
        y_val_proba = best_model.predict_proba(data['X_val'])[:, 1]
        
        val_auc = roc_auc_score(data['y_val'], y_val_proba)
        val_acc = accuracy_score(data['y_val'], y_val_pred)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            data['y_val'], y_val_pred, average='binary'
        )
        
        print(f"ROC-AUC: {val_auc:.4f}")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_prec:.4f}")
        print(f"Recall: {val_rec:.4f}")
        print(f"F1-Score: {val_f1:.4f}")
        
        # Evaluate on test set
        print(f"\n{'-'*80}")
        print("Test Set Evaluation")
        print(f"{'-'*80}")
        
        y_test_pred = best_model.predict(data['X_test'])
        y_test_proba = best_model.predict_proba(data['X_test'])[:, 1]
        
        test_auc = roc_auc_score(data['y_test'], y_test_proba)
        test_acc = accuracy_score(data['y_test'], y_test_pred)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            data['y_test'], y_test_pred, average='binary'
        )
        
        print(f"ROC-AUC: {test_auc:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_prec:.4f}")
        print(f"Recall: {test_rec:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(data['y_test'], y_test_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClassification Report:")
        print(classification_report(data['y_test'], y_test_pred))
        
        results = {
            'model': best_model,
            'model_name': 'XGBoost',
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'feature_importance': feature_importance.tolist(),
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
        
        # Save model and results
        self.save_model('XGBoost', results)
        self.save_results('XGBoost', results)
        
        return results


def train_xgboost_all_toxicities():
    """
    Train XGBoost for all 12 toxicity endpoints.
    """
    toxicity_endpoints = [
        'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST FOR ALL TOXICITY ENDPOINTS")
    print("="*80)
    print(f"Total endpoints: {len(toxicity_endpoints)}")
    print(f"Endpoints: {', '.join(toxicity_endpoints)}")
    print("="*80 + "\n")
    
    all_results = []
    
    for idx, toxicity in enumerate(toxicity_endpoints, 1):
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(toxicity_endpoints)}] Processing: {toxicity}")
        print(f"{'#'*80}\n")
        
        try:
            trainer = XGBoostTrainer(
                toxicity_name=toxicity,
                results_dir='results/baseline_models',
                models_dir='models/baseline_models',
                seed=42
            )
            
            results = trainer.train_xgboost()
            
            all_results.append({
                'Toxicity': toxicity,
                'Model': 'XGBoost',
                'Best_Params': str(results['best_params']),
                'CV_ROC_AUC': results['cv_score'],
                'Val_ROC_AUC': results['val_results']['auc'],
                'Test_ROC_AUC': results['test_results']['auc'],
                'Test_Accuracy': results['test_results']['accuracy'],
                'Test_F1': results['test_results']['f1']
            })
            
        except Exception as e:
            print(f"\nERROR processing {toxicity}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = 'results/baseline_models/XGBoost_all_toxicities_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*80)
        print("XGBOOST TRAINING COMPLETE - SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        print(f"Average Test ROC-AUC: {summary_df['Test_ROC_AUC'].mean():.4f}")
        print("="*80 + "\n")


def main():
    """
    Main function - train XGBoost for single or all toxicity endpoints.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--toxicity', type=str, default=None,
                       help='Toxicity endpoint (if not provided, trains all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.toxicity:
        # Train single toxicity
        trainer = XGBoostTrainer(
            toxicity_name=args.toxicity,
            results_dir='results/baseline_models',
            models_dir='models/baseline_models',
            seed=args.seed
        )
        trainer.train_xgboost()
    else:
        # Train all toxicities
        train_xgboost_all_toxicities()


if __name__ == '__main__':
    main()
