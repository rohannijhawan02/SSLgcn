"""
Bayesian Hyperparameter Optimization for GCN Model
Following the paper's methodology:
- Bayesian optimization algorithm with 32 trials
- ROC-AUC as the main metric for hyperparameter selection
- Early stopping with patience of 30 epochs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle
import json
from datetime import datetime
import pandas as pd

from model import create_gcn_model
from train import ToxicityTrainer
from train_all_toxicities import load_dataset


class BayesianHyperparameterTuner:
    """
    Bayesian Optimization for Hyperparameter Search using Optuna
    
    Uses Tree-structured Parzen Estimator (TPE) algorithm for efficient
    hyperparameter search with 32 trials maximum.
    """
    
    def __init__(self, 
                 dataset_name,
                 device='cpu',
                 n_trials=32,
                 patience=30,
                 results_dir='models/tuning_results'):
        """
        Initialize Bayesian hyperparameter tuner
        
        Args:
            dataset_name: Name of the toxicity dataset
            device: Device to run on ('cpu' or 'cuda')
            n_trials: Number of optimization trials (default: 32 as per paper)
            patience: Early stopping patience (default: 30 as per paper)
            results_dir: Directory to save tuning results
        """
        self.dataset_name = dataset_name
        self.device = device
        self.n_trials = n_trials
        self.patience = patience
        self.results_dir = os.path.join(results_dir, dataset_name)
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load dataset
        print(f"\n{'='*80}")
        print(f"Bayesian Hyperparameter Optimization for {dataset_name}")
        print(f"{'='*80}")
        print(f"Number of trials: {n_trials}")
        print(f"Early stopping patience: {patience} epochs")
        print(f"Optimization metric: Validation ROC-AUC")
        print(f"{'='*80}\n")
        
        self.train_loader, self.val_loader, self.test_loader, self.dataset_info = load_dataset(dataset_name)
        
        # Calculate class weights
        total_train = self.dataset_info['train_size']
        pos_count = max(self.dataset_info['train_pos'], 1)
        neg_count = max(self.dataset_info['train_neg'], 1)
        
        pos_weight = total_train / (2.0 * pos_count)
        neg_weight = total_train / (2.0 * neg_count)
        
        # Boost for extreme imbalance
        imbalance_ratio = neg_count / pos_count
        if imbalance_ratio > 20:
            boost_factor = min(1.5, (imbalance_ratio / 20) ** 0.5)
            pos_weight *= boost_factor
        
        self.class_weights = torch.FloatTensor([neg_weight, pos_weight]).to(device)
        
        print(f"Dataset Info:")
        print(f"  Train samples: {self.dataset_info['train_size']}")
        print(f"  Val samples: {self.dataset_info['val_size']}")
        print(f"  Class imbalance: 1:{imbalance_ratio:.1f}")
        print(f"  Class weights: [{neg_weight:.3f}, {pos_weight:.3f}]\n")
        
    def objective(self, trial):
        """
        Objective function for Bayesian optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation ROC-AUC score (to be maximized)
        """
        # Sample hyperparameters from search space
        num_layers = trial.suggest_int('num_layers', 2, 4)
        
        # Hidden dimensions for each layer
        hidden_dim1 = trial.suggest_categorical('hidden_dim1', [32, 64, 128])
        hidden_dim2 = trial.suggest_categorical('hidden_dim2', [64, 128, 256])
        hidden_dim3 = trial.suggest_categorical('hidden_dim3', [128, 256, 512])
        
        # Classifier hidden dimension
        classifier_hidden = trial.suggest_categorical('classifier_hidden', [64, 128, 256])
        
        # Regularization
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        
        # Learning rate
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        
        # Build hidden_dims based on num_layers
        if num_layers == 2:
            hidden_dims = [hidden_dim1, hidden_dim2]
        elif num_layers == 3:
            hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3]
        else:  # num_layers == 4
            hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3, 256]
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{self.n_trials}")
        print(f"{'='*70}")
        print(f"Hyperparameters:")
        print(f"  Layers: {num_layers}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Classifier hidden: {classifier_hidden}")
        print(f"  Dropout: {dropout}")
        print(f"  Learning rate: {learning_rate:.6f}")
        print(f"  Weight decay: {weight_decay:.6f}")
        print(f"{'='*70}\n")
        
        # Create model with sampled hyperparameters
        model = create_gcn_model(
            in_feats=self.dataset_info['in_feats'],
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            classifier_hidden=classifier_hidden,
            num_classes=2,
            dropout=dropout
        )
        
        # Create checkpoint directory for this trial
        trial_checkpoint_dir = os.path.join(self.results_dir, f'trial_{trial.number}')
        
        # Create trainer
        trainer = ToxicityTrainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=self.patience,
            checkpoint_dir=trial_checkpoint_dir,
            class_weights=self.class_weights
        )
        
        # Train model
        try:
            history = trainer.train(
                self.train_loader,
                self.val_loader,
                num_epochs=100  # Max epochs, but early stopping at self.patience
            )
            
            # Return best validation ROC-AUC
            best_val_auc = history['best_val_auc']
            
            print(f"\nTrial {trial.number + 1} Result:")
            print(f"  Best Validation ROC-AUC: {best_val_auc:.4f}")
            print(f"  Best Epoch: {history['best_epoch'] + 1}")
            
            return best_val_auc
            
        except Exception as e:
            print(f"\n❌ Trial {trial.number + 1} failed: {str(e)}")
            return 0.0  # Return worst score if trial fails
    
    def optimize(self):
        """
        Run Bayesian optimization to find best hyperparameters
        
        Returns:
            dict: Best hyperparameters and their performance
        """
        # Create Optuna study with TPE sampler (Bayesian optimization)
        study = optuna.create_study(
            direction='maximize',  # Maximize validation ROC-AUC
            sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=f'{self.dataset_name}_hyperopt'
        )
        
        print(f"\n🔍 Starting Bayesian Optimization with {self.n_trials} trials...")
        print(f"Algorithm: Tree-structured Parzen Estimator (TPE)")
        print(f"Objective: Maximize Validation ROC-AUC\n")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best trial
        best_trial = study.best_trial
        
        print(f"\n{'='*80}")
        print(f"Optimization Complete!")
        print(f"{'='*80}")
        print(f"Best Trial: #{best_trial.number + 1}")
        print(f"Best Validation ROC-AUC: {best_trial.value:.4f}")
        print(f"\nBest Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
        
        # Save optimization results
        self._save_results(study)
        
        # Return best hyperparameters
        return {
            'best_params': best_trial.params,
            'best_val_auc': best_trial.value,
            'best_trial_number': best_trial.number,
            'study': study
        }
    
    def _save_results(self, study):
        """
        Save optimization results to files
        
        Args:
            study: Optuna study object
        """
        # Save study object
        study_file = os.path.join(self.results_dir, 'study.pkl')
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        print(f"✓ Study saved to: {study_file}")
        
        # Save best hyperparameters as JSON
        best_params_file = os.path.join(self.results_dir, 'best_hyperparameters.json')
        with open(best_params_file, 'w') as f:
            json.dump({
                'dataset': self.dataset_name,
                'best_trial': study.best_trial.number,
                'best_val_auc': study.best_trial.value,
                'best_params': study.best_trial.params,
                'n_trials': self.n_trials,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        print(f"✓ Best hyperparameters saved to: {best_params_file}")
        
        # Save all trials as CSV
        trials_df = study.trials_dataframe()
        trials_csv = os.path.join(self.results_dir, 'all_trials.csv')
        trials_df.to_csv(trials_csv, index=False)
        print(f"✓ All trials saved to: {trials_csv}")
        
        # Create summary report
        summary = {
            'dataset': self.dataset_name,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number,
            'best_val_auc': study.best_trial.value,
            'optimization_time': str(study.trials[-1].datetime_complete - study.trials[0].datetime_start),
            'best_hyperparameters': study.best_trial.params
        }
        
        summary_file = os.path.join(self.results_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to: {summary_file}")


def tune_all_datasets(toxicity_list=None, n_trials=32, device='cpu'):
    """
    Run Bayesian hyperparameter optimization for all toxicity datasets
    
    Args:
        toxicity_list: List of toxicity dataset names (None for all)
        n_trials: Number of optimization trials per dataset
        device: Device to run on
    """
    # Default list of toxicity datasets
    if toxicity_list is None:
        toxicity_list = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
            'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
    
    print(f"\n{'='*80}")
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION FOR ALL DATASETS")
    print(f"{'='*80}")
    print(f"Datasets: {len(toxicity_list)}")
    print(f"Trials per dataset: {n_trials}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Store results for all datasets
    all_results = {}
    
    for idx, dataset_name in enumerate(toxicity_list, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {idx}/{len(toxicity_list)} datasets")
        print(f"Current: {dataset_name}")
        print(f"{'='*80}\n")
        
        try:
            # Create tuner
            tuner = BayesianHyperparameterTuner(
                dataset_name=dataset_name,
                device=device,
                n_trials=n_trials,
                patience=30
            )
            
            # Run optimization
            result = tuner.optimize()
            all_results[dataset_name] = result
            
            print(f"\n✓ Completed hyperparameter tuning for {dataset_name}")
            
        except Exception as e:
            print(f"\n❌ Failed to tune {dataset_name}: {str(e)}")
            all_results[dataset_name] = {'error': str(e)}
            continue
    
    # Save overall summary
    summary_df = pd.DataFrame([
        {
            'dataset': name,
            'best_val_auc': results.get('best_val_auc', 0.0),
            'best_trial': results.get('best_trial_number', -1),
            'status': 'success' if 'error' not in results else 'failed'
        }
        for name, results in all_results.items()
    ])
    
    summary_csv = os.path.join('models/tuning_results', 'overall_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Overall summary saved to: {summary_csv}")
    print(f"{'='*80}\n")
    
    # Automatically train with optimized parameters
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print(f"{'='*80}\n")
    
    from train_all_toxicities import train_all_toxicities
    training_results = train_all_toxicities(
        toxicity_list=toxicity_list,
        num_epochs=100,
        device=device,
        results_dir='results_optimized',
        use_tuned_params=True  # Automatically use the tuned parameters
    )
    
    print(f"\n{'='*80}")
    print(f"COMPLETE WORKFLOW FINISHED!")
    print(f"{'='*80}")
    print(f"1. Hyperparameter tuning results: {summary_csv}")
    print(f"2. Training results with optimized params: results_optimized/")
    print(f"{'='*80}\n")
    
    return {
        'tuning_results': all_results,
        'training_results': training_results
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization for SSL-GCN')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='Single dataset to tune (e.g., NR-AhR). If not specified, tunes all datasets.')
    parser.add_argument('--n_trials', type=int, default=5,
                        help='Number of optimization trials (default: 5 for testing, use 32 for full optimization)')
    parser.add_argument('--auto_train', action='store_true', default=True,
                        help='Automatically train with optimized parameters after tuning (default: True)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training after optimization (only tune hyperparameters)')
    
    args = parser.parse_args()
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Number of trials: {args.n_trials}")
    
    if args.n_trials < 32:
        print(f"⚠️  WARNING: Using {args.n_trials} trials for quick testing.")
        print(f"   For full optimization as per paper, use: --n_trials 32\n")
    
    if args.dataset:
        # Tune single dataset
        print(f"\n{'='*80}")
        print(f"Tuning hyperparameters for: {args.dataset}")
        print(f"{'='*80}\n")
        
        tuner = BayesianHyperparameterTuner(args.dataset, device=device, n_trials=args.n_trials)
        results = tuner.optimize()
        
        # Automatically train with optimized parameters
        if not args.skip_training:
            print(f"\n{'='*80}")
            print(f"Training {args.dataset} with optimized hyperparameters")
            print(f"{'='*80}\n")
            
            from train_all_toxicities import train_single_toxicity
            train_result = train_single_toxicity(
                dataset_name=args.dataset,
                num_epochs=100,
                device=device,
                results_dir='results_optimized',
                use_tuned_params=True
            )
            
            print(f"\n{'='*80}")
            print(f"✅ COMPLETE!")
            print(f"{'='*80}")
            print(f"Tuning results: models/tuning_results/{args.dataset}/")
            print(f"Training results: results_optimized/{args.dataset}/")
            print(f"{'='*80}\n")
    else:
        # Tune all datasets
        print(f"\n{'='*80}")
        print(f"Tuning hyperparameters for ALL datasets")
        print(f"{'='*80}\n")
        
        if not args.skip_training:
            # This function will automatically train after tuning
            results = tune_all_datasets(n_trials=args.n_trials, device=device)
        else:
            # Only tune, don't train
            print("⚠️  Skipping training (--skip_training flag set)")
            toxicity_list = [
                'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
                'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
            ]
            
            all_results = {}
            for idx, dataset_name in enumerate(toxicity_list, 1):
                print(f"\nProgress: {idx}/{len(toxicity_list)} - {dataset_name}")
                try:
                    tuner = BayesianHyperparameterTuner(dataset_name, device=device, n_trials=args.n_trials)
                    result = tuner.optimize()
                    all_results[dataset_name] = result
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    all_results[dataset_name] = {'error': str(e)}
            
            print(f"\n{'='*80}")
            print(f"Tuning complete! Run training manually with:")
            print(f"python src/train_all_toxicities.py --use_tuned_params")
            print(f"{'='*80}\n")
