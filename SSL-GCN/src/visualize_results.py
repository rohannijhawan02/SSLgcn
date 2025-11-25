"""
Visualize Training Results from CSV Files
Creates plots for training history and performance comparison
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_training_history(dataset_name, results_dir='results'):
    """
    Plot training history for a single dataset
    
    Args:
        dataset_name: Name of the toxicity dataset
        results_dir: Directory containing results
    """
    # Load training history
    csv_path = os.path.join(results_dir, dataset_name, 'training_history.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Loss
    axes[0].plot(df['epoch'], df['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(df['epoch'], df['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(df['epoch'], df['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[1].plot(df['epoch'], df['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: ROC-AUC
    axes[2].plot(df['epoch'], df['train_auc'], 'b-', label='Train', linewidth=2)
    axes[2].plot(df['epoch'], df['val_auc'], 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('ROC-AUC', fontsize=12)
    axes[2].set_title('Training and Validation ROC-AUC', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch_idx = df['val_auc'].idxmax()
    best_epoch = df.loc[best_epoch_idx, 'epoch']
    best_auc = df.loc[best_epoch_idx, 'val_auc']
    axes[2].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (Epoch {best_epoch})')
    axes[2].plot(best_epoch, best_auc, 'g*', markersize=15, label=f'Best AUC: {best_auc:.4f}')
    axes[2].legend(fontsize=10)
    
    plt.suptitle(f'{dataset_name} - Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(results_dir, dataset_name, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


def plot_overall_comparison(results_dir='results'):
    """
    Plot comparison of all datasets
    
    Args:
        results_dir: Directory containing results
    """
    # Load overall summary
    summary_path = os.path.join(results_dir, 'overall_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found!")
        return
    
    df = pd.read_csv(summary_path)
    
    # Sort by test AUC
    df = df.sort_values('test_roc_auc', ascending=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Test AUC comparison
    axes[0, 0].barh(df['dataset'], df['test_roc_auc'], color='steelblue')
    axes[0, 0].set_xlabel('ROC-AUC', fontsize=12)
    axes[0, 0].set_title('Test ROC-AUC by Dataset', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
    axes[0, 0].set_xlim(0, 1.0)
    for i, (dataset, auc) in enumerate(zip(df['dataset'], df['test_roc_auc'])):
        axes[0, 0].text(auc + 0.01, i, f'{auc:.3f}', va='center', fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Multiple metrics comparison
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    x = np.arange(len(df))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        axes[0, 1].bar(x + i*width, df[metric], width, 
                       label=metric.replace('test_', '').title())
    
    axes[0, 1].set_xlabel('Dataset', fontsize=12)
    axes[0, 1].set_ylabel('Score', fontsize=12)
    axes[0, 1].set_title('Test Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x + width * 1.5)
    axes[0, 1].set_xticklabels(df['dataset'], rotation=45, ha='right', fontsize=9)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Dataset size comparison
    axes[1, 0].bar(df['dataset'], df['train_samples'], alpha=0.7, label='Train', color='blue')
    axes[1, 0].bar(df['dataset'], df['test_samples'], alpha=0.7, label='Test', color='orange')
    axes[1, 0].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 0].set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticklabels(df['dataset'], rotation=45, ha='right', fontsize=9)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training efficiency (epochs needed)
    axes[1, 1].bar(df['dataset'], df['best_epoch'], color='green', alpha=0.7)
    axes[1, 1].set_ylabel('Best Epoch', fontsize=12)
    axes[1, 1].set_title('Training Epochs to Best Model', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticklabels(df['dataset'], rotation=45, ha='right', fontsize=9)
    for i, (dataset, epoch) in enumerate(zip(df['dataset'], df['best_epoch'])):
        axes[1, 1].text(i, epoch + 1, str(int(epoch)), ha='center', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Overall Performance Comparison Across All Datasets', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(results_dir, 'overall_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overall comparison saved to: {output_path}")
    plt.close()


def create_metrics_heatmap(results_dir='results'):
    """
    Create heatmap of all metrics across datasets
    
    Args:
        results_dir: Directory containing results
    """
    # Load overall summary
    summary_path = os.path.join(results_dir, 'overall_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found!")
        return
    
    df = pd.read_csv(summary_path)
    
    # Select metrics for heatmap
    metrics = ['test_accuracy', 'test_roc_auc', 'test_precision', 'test_recall', 'test_f1', 'best_val_auc']
    heatmap_data = df[['dataset'] + metrics].set_index('dataset')
    
    # Rename columns for better display
    heatmap_data.columns = ['Accuracy', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'Best Val AUC']
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'Score'},
                linewidths=0.5, linecolor='gray')
    plt.title('Performance Metrics Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(results_dir, 'metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics heatmap saved to: {output_path}")
    plt.close()


def visualize_all_results(results_dir='results'):
    """
    Create all visualization plots
    
    Args:
        results_dir: Directory containing results
    """
    print("=" * 80)
    print("VISUALIZING TRAINING RESULTS")
    print("=" * 80)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        print("Please run training first: python train_all_toxicities.py")
        return
    
    # Get list of datasets
    datasets = [d for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))]
    
    if not datasets:
        print(f"Error: No dataset folders found in '{results_dir}'")
        return
    
    print(f"\nFound {len(datasets)} datasets")
    print("=" * 80)
    
    # Plot training history for each dataset
    print("\nCreating training curve plots...")
    for i, dataset in enumerate(datasets, 1):
        print(f"  [{i}/{len(datasets)}] Processing {dataset}...")
        try:
            plot_training_history(dataset, results_dir)
        except Exception as e:
            print(f"    Error: {str(e)}")
    
    # Plot overall comparison
    print("\nCreating overall comparison plot...")
    try:
        plot_overall_comparison(results_dir)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Create heatmap
    print("\nCreating metrics heatmap...")
    try:
        create_metrics_heatmap(results_dir)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {results_dir}/<dataset>/training_curves.png (for each dataset)")
    print(f"  - {results_dir}/overall_comparison.png")
    print(f"  - {results_dir}/metrics_heatmap.png")
    print("=" * 80)


if __name__ == "__main__":
    # Visualize all results
    visualize_all_results(results_dir='results')
    
    # You can also visualize a single dataset:
    # plot_training_history('NR-AhR', results_dir='results')
