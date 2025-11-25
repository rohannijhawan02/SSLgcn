"""
Neural Network (NN) Model Training for Toxicity Prediction
Uses ECFP4 fingerprints for molecular encoding

Based on: "Neural Network (NN) was tested. The compounds were encoded using 
the Extended Connectivity Fingerprints (ECFP4), which is a circular topological 
fingerprint designed for molecular characterization, similarity searching, and 
structure-activity modeling."
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from train_baseline_models import BaselineModelTrainer, ECFP4Encoder
from data_preprocessing import ScaffoldSplitter


class NNTrainer(BaselineModelTrainer):
    """
    Specialized trainer for Neural Network (Multi-layer Perceptron) classifier.
    """
    
    def __init__(self, toxicity_name, **kwargs):
        super().__init__(toxicity_name, **kwargs)
        
        # NN-specific configuration
        self.model_name = 'NN'
        self.param_grid = {
            'hidden_layer_sizes': [
                (64,), (128,), (256,),
                (64, 32), (128, 64), (256, 128),
                (128, 64, 32), (256, 128, 64)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [1000],
            'early_stopping': [True],
            'validation_fraction': [0.1]
        }
        
        print(f"\n{'='*80}")
        print(f"Neural Network (NN/MLP) Trainer Initialized")
        print(f"{'='*80}")
        print(f"Toxicity: {toxicity_name}")
        print(f"Fingerprint: ECFP4 (radius=2, n_bits=2048)")
        print(f"Hyperparameters to optimize:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        print(f"{'='*80}\n")
    
    def train_nn(self):
        """
        Train Neural Network model with hyperparameter optimization.
        
        Returns:
            dict: Trained model and results
        """
        print(f"\n{'-'*80}")
        print(f"Training Neural Network for {self.toxicity_name}")
        print(f"{'-'*80}")
        
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        # Combine train and val for grid search
        X_train_val = np.vstack([data['X_train'], data['X_val']])
        y_train_val = np.concatenate([data['y_train'], data['y_val']])
        
        # Create NN classifier
        nn = MLPClassifier(random_state=self.seed)
        
        # Grid search with cross-validation
        print(f"\nPerforming grid search with 5-fold cross-validation...")
        print("Note: This may take a while for neural networks...")
        
        grid_search = GridSearchCV(
            nn,
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
        
        # Print network architecture
        print(f"\nBest Network Architecture:")
        print(f"  Input layer: {X_train_val.shape[1]} nodes (ECFP4 fingerprint)")
        print(f"  Hidden layers: {best_model.hidden_layer_sizes}")
        print(f"  Output layer: 2 nodes (binary classification)")
        print(f"  Activation: {best_model.activation}")
        print(f"  Total parameters: ~{self._estimate_parameters(best_model, X_train_val.shape[1])}")
        
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
        
        # Training info
        if hasattr(best_model, 'n_iter_'):
            print(f"\nTraining iterations: {best_model.n_iter_}")
        if hasattr(best_model, 'loss_'):
            print(f"Final loss: {best_model.loss_:.6f}")
        
        results = {
            'model': best_model,
            'model_name': 'NN',
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
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
        self.save_model('NN', results)
        self.save_results('NN', results)
        
        return results
    
    def _estimate_parameters(self, model, input_size):
        """
        Estimate number of parameters in the neural network.
        
        Args:
            model: Trained MLPClassifier
            input_size: Size of input layer
            
        Returns:
            int: Estimated number of parameters
        """
        layers = [input_size] + list(model.hidden_layer_sizes) + [2]  # 2 output classes
        params = 0
        for i in range(len(layers) - 1):
            params += layers[i] * layers[i+1] + layers[i+1]  # weights + biases
        return params


def train_nn_all_toxicities():
    """
    Train Neural Network for all 12 toxicity endpoints.
    """
    toxicity_endpoints = [
        'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
        'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    print("\n" + "="*80)
    print("TRAINING NEURAL NETWORK FOR ALL TOXICITY ENDPOINTS")
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
            trainer = NNTrainer(
                toxicity_name=toxicity,
                results_dir='results/baseline_models',
                models_dir='models/baseline_models',
                seed=42
            )
            
            results = trainer.train_nn()
            
            all_results.append({
                'Toxicity': toxicity,
                'Model': 'NN',
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
        summary_path = 'results/baseline_models/NN_all_toxicities_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*80)
        print("NEURAL NETWORK TRAINING COMPLETE - SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        print(f"Average Test ROC-AUC: {summary_df['Test_ROC_AUC'].mean():.4f}")
        print("="*80 + "\n")


def main():
    """
    Main function - train NN for single or all toxicity endpoints.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Neural Network model')
    parser.add_argument('--toxicity', type=str, default=None,
                       help='Toxicity endpoint (if not provided, trains all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.toxicity:
        # Train single toxicity
        trainer = NNTrainer(
            toxicity_name=args.toxicity,
            results_dir='results/baseline_models',
            models_dir='models/baseline_models',
            seed=args.seed
        )
        trainer.train_nn()
    else:
        # Train all toxicities
        train_nn_all_toxicities()


if __name__ == '__main__':
    main()
