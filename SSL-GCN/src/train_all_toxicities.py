"""
Train and Evaluate GCN Models for All Toxicity Datasets
Saves detailed results in CSV format for each toxicity separately
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import dgl
import pickle
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm

from model import create_gcn_model
from train import ToxicityTrainer


class MoleculeDataset(Dataset):
    """Dataset class for molecular graphs"""
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_fn(batch):
    """Collate function for batching DGL graphs"""
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.LongTensor(labels)


def convert_to_dgl_graph(data):
    """Convert graph data dictionary to DGL graph"""
    adj_matrix = data['adjacency_matrix']
    features = data['feature_matrix']
    
    # Create graph from adjacency matrix
    src, dst = np.where(adj_matrix > 0)
    g = dgl.graph((src, dst), num_nodes=adj_matrix.shape[0])
    
    # Add self-loops to handle 0-in-degree nodes
    g = dgl.add_self_loop(g)
    
    # Add node features
    g.ndata['feat'] = torch.FloatTensor(features)
    
    return g


def load_dataset(dataset_name, data_dir='Data/cache'):
    """
    Load complete dataset for a toxicity
    
    Args:
        dataset_name: Name of the toxicity dataset (e.g., 'NR-AhR')
        data_dir: Directory containing cached data
    
    Returns:
        train_loader, val_loader, test_loader, in_feats, dataset_info
    """
    print(f"\nLoading {dataset_name} dataset...")
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Load splits
    with open(os.path.join(dataset_path, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)
    
    # Load graph data
    print("  Loading graphs...")
    with open(os.path.join(dataset_path, 'train_graphs.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(dataset_path, 'val_graphs.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(dataset_path, 'test_graphs.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    # Convert to DGL graphs
    print("  Converting to DGL format...")
    train_graphs = [convert_to_dgl_graph(d) for d in tqdm(train_data, desc="    Train", leave=False)]
    val_graphs = [convert_to_dgl_graph(d) for d in tqdm(val_data, desc="    Val", leave=False)]
    test_graphs = [convert_to_dgl_graph(d) for d in tqdm(test_data, desc="    Test", leave=False)]
    
    # Get labels
    train_labels = [int(label) for label in splits['train']['labels']]
    val_labels = [int(label) for label in splits['val']['labels']]
    test_labels = [int(label) for label in splits['test']['labels']]
    
    # Dataset info
    dataset_info = {
        'name': dataset_name,
        'train_size': len(train_graphs),
        'val_size': len(val_graphs),
        'test_size': len(test_graphs),
        'train_pos': sum(train_labels),
        'train_neg': len(train_labels) - sum(train_labels),
        'val_pos': sum(val_labels),
        'val_neg': len(val_labels) - sum(val_labels),
        'test_pos': sum(test_labels),
        'test_neg': len(test_labels) - sum(test_labels),
        'in_feats': train_graphs[0].ndata['feat'].shape[1]
    }
    
    print(f"  Train: {dataset_info['train_size']} samples ({dataset_info['train_pos']} positive, {dataset_info['train_neg']} negative)")
    print(f"  Val:   {dataset_info['val_size']} samples ({dataset_info['val_pos']} positive, {dataset_info['val_neg']} negative)")
    print(f"  Test:  {dataset_info['test_size']} samples ({dataset_info['test_pos']} positive, {dataset_info['test_neg']} negative)")
    print(f"  Features: {dataset_info['in_feats']}")
    
    # Create data loaders
    train_dataset = MoleculeDataset(train_graphs, train_labels)
    val_dataset = MoleculeDataset(val_graphs, val_labels)
    test_dataset = MoleculeDataset(test_graphs, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, dataset_info


def save_results_to_csv(dataset_name, training_history, test_metrics, dataset_info, results_dir='results'):
    """
    Save training and testing results to CSV files
    
    Args:
        dataset_name: Name of the toxicity dataset
        training_history: Training history dictionary
        test_metrics: Test evaluation metrics dictionary
        dataset_info: Dataset information dictionary
        results_dir: Directory to save results
    """
    # Create results directory for this dataset
    dataset_results_dir = os.path.join(results_dir, dataset_name)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # 1. Save training history (epoch-by-epoch)
    training_df = pd.DataFrame({
        'epoch': range(1, len(training_history['train_loss']) + 1),
        'train_loss': training_history['train_loss'],
        'train_accuracy': training_history['train_acc'],
        'train_auc': training_history['train_auc'],
        'val_loss': training_history['val_loss'],
        'val_accuracy': training_history['val_acc'],
        'val_auc': training_history['val_auc']
    })
    training_csv = os.path.join(dataset_results_dir, 'training_history.csv')
    training_df.to_csv(training_csv, index=False)
    print(f"  ✓ Training history saved to: {training_csv}")
    
    # 2. Save test results with diagnostic info
    test_df = pd.DataFrame({
        'metric': ['accuracy', 'roc_auc', 'precision', 'recall', 'f1_score', 
                   'predicted_positive', 'actual_positive', 'predicted_negative', 'actual_negative'],
        'value': [
            test_metrics['accuracy'],
            test_metrics['auc'],
            test_metrics.get('precision', 0.0),
            test_metrics.get('recall', 0.0),
            test_metrics.get('f1', 0.0),
            test_metrics.get('num_pred_positive', 0),
            dataset_info['test_pos'],
            test_metrics.get('num_pred_negative', 0),
            dataset_info['test_neg']
        ]
    })
    test_csv = os.path.join(dataset_results_dir, 'test_results.csv')
    test_df.to_csv(test_csv, index=False)
    print(f"  ✓ Test results saved to: {test_csv}")
    
    # Print warning if model predicts all negative
    if test_metrics.get('num_pred_positive', 0) == 0:
        print(f"  ⚠️  WARNING: Model predicted ALL samples as NEGATIVE!")
        print(f"      This means the model is NOT learning to identify toxic compounds.")
        print(f"      Actual positive samples in test: {dataset_info['test_pos']}")
    
    # 3. Save dataset info and summary
    summary_df = pd.DataFrame({
        'attribute': [
            'dataset_name',
            'train_samples',
            'val_samples',
            'test_samples',
            'train_positive',
            'train_negative',
            'val_positive',
            'val_negative',
            'test_positive',
            'test_negative',
            'input_features',
            'best_epoch',
            'best_val_auc',
            'test_accuracy',
            'test_auc',
            'test_precision',
            'test_recall',
            'test_f1'
        ],
        'value': [
            dataset_info['name'],
            dataset_info['train_size'],
            dataset_info['val_size'],
            dataset_info['test_size'],
            dataset_info['train_pos'],
            dataset_info['train_neg'],
            dataset_info['val_pos'],
            dataset_info['val_neg'],
            dataset_info['test_pos'],
            dataset_info['test_neg'],
            dataset_info['in_feats'],
            training_history['best_epoch'] + 1,
            f"{training_history['best_val_auc']:.4f}",
            f"{test_metrics['accuracy']:.4f}",
            f"{test_metrics['auc']:.4f}",
            f"{test_metrics['precision']:.4f}",
            f"{test_metrics['recall']:.4f}",
            f"{test_metrics['f1']:.4f}"
        ]
    })
    summary_csv = os.path.join(dataset_results_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Summary saved to: {summary_csv}")


def load_best_hyperparameters(dataset_name, tuning_results_dir='models/tuning_results'):
    """
    Load best hyperparameters from Bayesian optimization results
    
    Args:
        dataset_name: Name of the toxicity dataset
        tuning_results_dir: Directory containing tuning results
        
    Returns:
        dict: Best hyperparameters or None if not found
    """
    best_params_file = os.path.join(tuning_results_dir, dataset_name, 'best_hyperparameters.json')
    
    if os.path.exists(best_params_file):
        with open(best_params_file, 'r') as f:
            data = json.load(f)
            return data.get('best_params')
    return None


def train_single_toxicity(dataset_name, num_epochs=100, device='cpu', results_dir='results', use_tuned_params=False):
    """
    Train and evaluate model for a single toxicity dataset
    
    Args:
        dataset_name: Name of the toxicity dataset
        num_epochs: Maximum number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        results_dir: Directory to save results
        use_tuned_params: Whether to use hyperparameters from Bayesian optimization
    
    Returns:
        dict: Results including test metrics
    """
    print("\n" + "=" * 80)
    print(f"Training Model for {dataset_name}")
    print("=" * 80)
    
    # Load dataset
    train_loader, val_loader, test_loader, dataset_info = load_dataset(dataset_name)
    
    # Calculate class weights for imbalanced data
    # Use effective number of samples method for extreme imbalance
    total_train = dataset_info['train_size']
    pos_count = max(dataset_info['train_pos'], 1)
    neg_count = max(dataset_info['train_neg'], 1)
    
    # Method 1: Balanced class weights (used in sklearn)
    pos_weight = total_train / (2.0 * pos_count)
    neg_weight = total_train / (2.0 * neg_count)
    
    # For extremely imbalanced datasets (ratio > 1:20), boost positive weight more
    imbalance_ratio = neg_count / pos_count
    if imbalance_ratio > 20:
        # Apply additional scaling for extreme imbalance
        boost_factor = min(1.5, (imbalance_ratio / 20) ** 0.5)
        pos_weight *= boost_factor
        print(f"  Extreme imbalance detected (1:{imbalance_ratio:.1f}), boosting positive weight by {boost_factor:.2f}x")
    
    class_weights = torch.FloatTensor([neg_weight, pos_weight]).to(device)
    
    print(f"\n⚖️  Class Imbalance Info:")
    print(f"  Train - Positive: {dataset_info['train_pos']} ({dataset_info['train_pos']/total_train*100:.1f}%)")
    print(f"  Train - Negative: {dataset_info['train_neg']} ({dataset_info['train_neg']/total_train*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
    print(f"  Class weights - Negative: {neg_weight:.3f}, Positive: {pos_weight:.3f}")
    
    # Load hyperparameters (either from tuning or use defaults)
    if use_tuned_params:
        best_params = load_best_hyperparameters(dataset_name)
        if best_params:
            print(f"\n🎯 Using optimized hyperparameters from Bayesian search:")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            
            # Extract hyperparameters
            num_layers = best_params.get('num_layers', 3)
            hidden_dim1 = best_params.get('hidden_dim1', 64)
            hidden_dim2 = best_params.get('hidden_dim2', 128)
            hidden_dim3 = best_params.get('hidden_dim3', 256)
            
            if num_layers == 2:
                hidden_dims = [hidden_dim1, hidden_dim2]
            elif num_layers == 3:
                hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3]
            else:  # num_layers == 4
                hidden_dims = [hidden_dim1, hidden_dim2, hidden_dim3, 256]
            
            classifier_hidden = best_params.get('classifier_hidden', 128)
            dropout = best_params.get('dropout', 0.3)
            learning_rate = best_params.get('learning_rate', 0.001)
            weight_decay = best_params.get('weight_decay', 1e-5)
        else:
            print(f"\n⚠️  No tuned hyperparameters found for {dataset_name}, using defaults")
            use_tuned_params = False
    
    if not use_tuned_params:
        # Default hyperparameters
        hidden_dims = [64, 128, 256]
        num_layers = 3
        classifier_hidden = 128
        dropout = 0.3
        learning_rate = 0.001
        weight_decay = 1e-5
        print(f"\n📋 Using default hyperparameters")
    
    # Create model
    print("\nCreating model...")
    model = create_gcn_model(
        in_feats=dataset_info['in_feats'],
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        classifier_hidden=classifier_hidden,
        num_classes=2,
        dropout=dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Create trainer with class weights
    checkpoint_dir = os.path.join('checkpoints', dataset_name)
    trainer = ToxicityTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=30,  # As per paper: early stopping at 30 epochs
        checkpoint_dir=checkpoint_dir,
        class_weights=class_weights  # Pass class weights to handle imbalance
    )
    
    # Train model
    print("\nStarting training...")
    training_history = trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Load best model and evaluate on test set
    print("\nLoading best model for testing...")
    trainer.load_checkpoint('best_model.pt')
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results to CSV
    print(f"\nSaving results to {results_dir}/{dataset_name}/...")
    save_results_to_csv(dataset_name, training_history, test_metrics, dataset_info, results_dir)
    
    print("\n" + "=" * 80)
    print(f"Completed {dataset_name}")
    print("=" * 80)
    
    return {
        'dataset_name': dataset_name,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'dataset_info': dataset_info
    }


def train_all_toxicities(toxicity_list=None, num_epochs=100, device='cpu', results_dir='results', use_tuned_params=False):
    """
    Train and evaluate models for all toxicity datasets
    
    Args:
        toxicity_list: List of toxicity dataset names (if None, uses all available)
        num_epochs: Maximum number of training epochs per dataset
        device: Device to train on ('cpu' or 'cuda')
        results_dir: Directory to save results
        use_tuned_params: Whether to use hyperparameters from Bayesian optimization
    
    Returns:
        dict: Results for all datasets
    """
    # Default list of all toxicity datasets
    if toxicity_list is None:
        toxicity_list = [
            'NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
            'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
    
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("TRAINING GCN MODELS FOR ALL TOXICITY DATASETS")
    print("=" * 80)
    print(f"Datasets: {len(toxicity_list)}")
    print(f"Max epochs per dataset: {num_epochs}")
    print(f"Device: {device}")
    print(f"Results directory: {results_dir}")
    print(f"Using tuned hyperparameters: {'Yes' if use_tuned_params else 'No'}")
    print("=" * 80)
    
    all_results = []
    start_time = datetime.now()
    
    # Train each dataset
    for i, dataset_name in enumerate(toxicity_list, 1):
        print(f"\n\n{'=' * 80}")
        print(f"Progress: {i}/{len(toxicity_list)} datasets")
        print(f"Current: {dataset_name}")
        print("=" * 80)
        
        try:
            result = train_single_toxicity(dataset_name, num_epochs, device, results_dir, use_tuned_params)
            all_results.append(result)
            print(f"✓ Successfully completed {dataset_name}")
        except Exception as e:
            print(f"✗ Error training {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary CSV for all datasets
    print("\n" + "=" * 80)
    print("Creating overall summary...")
    print("=" * 80)
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'dataset': result['dataset_name'],
            'train_samples': result['dataset_info']['train_size'],
            'test_samples': result['dataset_info']['test_size'],
            'test_accuracy': result['test_metrics']['accuracy'],
            'test_roc_auc': result['test_metrics']['auc'],
            'test_precision': result['test_metrics']['precision'],
            'test_recall': result['test_metrics']['recall'],
            'test_f1': result['test_metrics']['f1'],
            'best_val_auc': result['training_history']['best_val_auc'],
            'best_epoch': result['training_history']['best_epoch'] + 1
        })
    
    overall_summary_df = pd.DataFrame(summary_data)
    overall_summary_csv = os.path.join(results_dir, 'overall_summary.csv')
    overall_summary_df.to_csv(overall_summary_csv, index=False)
    
    print(f"\n✓ Overall summary saved to: {overall_summary_csv}")
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETED!")
    print("=" * 80)
    print(f"Total datasets processed: {len(all_results)}/{len(toxicity_list)}")
    print(f"Total time: {duration}")
    print(f"Results saved in: {results_dir}/")
    print("\nPer-dataset results:")
    for result in all_results:
        print(f"  {result['dataset_name']}: Test AUC = {result['test_metrics']['auc']:.4f}")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GCN models for toxicity prediction')
    parser.add_argument('--use_tuned_params', action='store_true',
                        help='Use hyperparameters from Bayesian optimization (if available)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Train single dataset (e.g., NR-AhR). If not specified, trains all.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs (default: 100)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Auto-detect if tuned parameters exist
    if not args.use_tuned_params:
        import os
        if os.path.exists('models/tuning_results'):
            tuned_datasets = [d for d in os.listdir('models/tuning_results') 
                            if os.path.isdir(os.path.join('models/tuning_results', d))]
            if tuned_datasets:
                print(f"\n✅ Found optimized hyperparameters for {len(tuned_datasets)} datasets!")
                print(f"   Automatically using tuned parameters.")
                print(f"   To use default parameters, run with: --use_tuned_params=False\n")
                args.use_tuned_params = True
    
    if args.dataset:
        # Train single dataset
        print(f"\nTraining single dataset: {args.dataset}")
        result = train_single_toxicity(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            device=device,
            results_dir=args.results_dir,
            use_tuned_params=args.use_tuned_params
        )
    else:
        # Train all datasets
        results = train_all_toxicities(
            toxicity_list=None,  # Use all datasets
            num_epochs=args.epochs,
            device=device,
            results_dir=args.results_dir,
            use_tuned_params=args.use_tuned_params
        )
