"""
Training Pipeline for GCN Toxicity Prediction
Handles model training, validation, and evaluation with ROC-AUC metric
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import dgl
from tqdm import tqdm
import os
import pickle
import json
from datetime import datetime


class ToxicityTrainer:
    """
    Trainer class for GCN toxicity prediction model
    
    Handles training, validation, and testing with early stopping
    based on ROC-AUC metric (important for imbalanced datasets)
    """
    
    def __init__(self, 
                 model, 
                 device='cpu',
                 learning_rate=0.001,
                 weight_decay=1e-5,
                 patience=10,
                 checkpoint_dir='checkpoints',
                 class_weights=None):
        """
        Initialize the trainer
        
        Args:
            model: GCN model instance
            device: Device to run training on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            patience: Early stopping patience (epochs)
            checkpoint_dir: Directory to save model checkpoints
            class_weights: Tensor of class weights for imbalanced datasets
        """
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (handles class imbalance with weights)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Using weighted loss with class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_auc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'best_epoch': 0,
            'best_val_auc': 0.0
        }
        
    def compute_class_weights(self, labels):
        """
        Compute class weights for imbalanced dataset
        
        Args:
            labels: Training labels
            
        Returns:
            Tensor: Class weights
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = torch.FloatTensor([total / (len(unique) * count) for count in counts])
        return weights.to(self.device)
    
    def collate_fn(self, batch):
        """
        Collate function for batching DGL graphs
        
        Args:
            batch: List of (graph, label) tuples
            
        Returns:
            Batched graph and labels
        """
        graphs, labels = zip(*batch)
        batched_graph = dgl.batch(graphs)
        labels = torch.LongTensor(labels)
        return batched_graph, labels
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            dict: Training metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch_graphs, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_graphs = batch_graphs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_features = batch_graphs.ndata['feat'].float()
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_graphs, batch_features)
            loss = self.criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # ROC-AUC (main metric for imbalanced data)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_graphs, batch_labels in tqdm(val_loader, desc="Validation", leave=False):
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_features = batch_graphs.ndata['feat'].float()
                
                # Forward pass
                logits = self.model(batch_graphs, batch_features)
                loss = self.criterion(logits, batch_labels)
                
                # Record metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # ROC-AUC (main metric for imbalanced data)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """
        Complete training loop with early stopping
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs
            
        Returns:
            dict: Training history
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Max epochs: {num_epochs}")
        print(f"Early stopping patience: {self.patience}")
        print("=" * 70)
        
        best_val_auc = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['train_auc'].append(train_metrics['auc'])
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_auc'].append(val_metrics['auc'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}")
            
            # Early stopping based on validation ROC-AUC
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self.history['best_epoch'] = epoch
                self.history['best_val_auc'] = best_val_auc
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"✓ New best model saved! (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{self.patience})")
                
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {self.history['best_epoch'] + 1}")
                    break
        
        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"Best Validation AUC: {self.history['best_val_auc']:.4f}")
        print(f"Best Epoch: {self.history['best_epoch'] + 1}")
        
        return self.history
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            dict: Test metrics
        """
        print("\n" + "=" * 70)
        print("Evaluating on Test Set")
        print("=" * 70)
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_graphs, batch_labels in tqdm(test_loader, desc="Testing"):
                batch_graphs = batch_graphs.to(self.device)
                batch_features = batch_graphs.ndata['feat'].float()
                
                # Forward pass
                logits = self.model(batch_graphs, batch_features)
                
                # Record predictions
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        # Count predictions for diagnostics
        num_pred_positive = np.sum(all_preds)
        num_pred_negative = len(all_preds) - num_pred_positive
        num_actual_positive = np.sum(all_labels)
        num_actual_negative = len(all_labels) - num_actual_positive
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_pred_positive': num_pred_positive,
            'num_pred_negative': num_pred_negative,
            'num_total_samples': len(all_labels)
        }
        
        print(f"\nTest Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nPrediction Distribution:")
        print(f"  Predicted Positive: {num_pred_positive}/{len(all_labels)} ({num_pred_positive/len(all_labels)*100:.1f}%)")
        print(f"  Predicted Negative: {num_pred_negative}/{len(all_labels)} ({num_pred_negative/len(all_labels)*100:.1f}%)")
        print(f"  Actual Positive:    {num_actual_positive}/{len(all_labels)} ({num_actual_positive/len(all_labels)*100:.1f}%)")
        print(f"  Actual Negative:    {num_actual_negative}/{len(all_labels)} ({num_actual_negative/len(all_labels)*100:.1f}%)")
        
        if num_pred_positive == 0:
            print(f"\n⚠️  WARNING: Model predicted ALL samples as NEGATIVE!")
            print(f"    This means the model failed to learn to identify toxic compounds.")
        
        print("=" * 70)
        
        return metrics
    
    def save_checkpoint(self, filename, epoch, metrics):
        """
        Save model checkpoint
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch
            metrics: Current metrics
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """
        Load model checkpoint
        
        Args:
            filename: Name of checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint


def run_multiple_trials(model_fn, 
                       train_loader, 
                       val_loader, 
                       test_loader,
                       num_trials=5,
                       num_epochs=100,
                       device='cpu',
                       results_dir='results'):
    """
    Run multiple training trials to get reliable performance estimates
    
    Args:
        model_fn: Function that creates a new model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_trials: Number of independent trials
        num_epochs: Max epochs per trial
        device: Device to run on
        results_dir: Directory to save results
        
    Returns:
        dict: Aggregated results across trials
    """
    os.makedirs(results_dir, exist_ok=True)
    
    all_test_metrics = []
    all_histories = []
    
    print("=" * 70)
    print(f"Running {num_trials} Independent Trials")
    print("=" * 70)
    
    for trial in range(num_trials):
        print(f"\n{'=' * 70}")
        print(f"Trial {trial + 1}/{num_trials}")
        print("=" * 70)
        
        # Create new model for this trial
        model = model_fn()
        
        # Create trainer
        checkpoint_dir = os.path.join(results_dir, f'trial_{trial + 1}')
        trainer = ToxicityTrainer(
            model=model,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        # Train
        history = trainer.train(train_loader, val_loader, num_epochs)
        all_histories.append(history)
        
        # Load best model and evaluate on test set
        trainer.load_checkpoint('best_model.pt')
        test_metrics = trainer.evaluate(test_loader)
        all_test_metrics.append(test_metrics)
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregated Results Across All Trials")
    print("=" * 70)
    
    aggregated = {}
    for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
        values = [m[metric] for m in all_test_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'all_values': values
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {aggregated[metric]['mean']:.4f} ± {aggregated[metric]['std']:.4f}")
        print(f"  Min:  {aggregated[metric]['min']:.4f}")
        print(f"  Max:  {aggregated[metric]['max']:.4f}")
    
    # Save aggregated results
    results_file = os.path.join(results_dir, 'aggregated_results.json')
    with open(results_file, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        json_aggregated = {}
        for metric, stats in aggregated.items():
            json_aggregated[metric] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'all_values': [float(v) for v in stats['all_values']]
            }
        json.dump(json_aggregated, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 70)
    
    return aggregated, all_test_metrics, all_histories


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use this module to train GCN models for toxicity prediction")
