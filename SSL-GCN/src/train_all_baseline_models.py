"""
Train Baseline Models for All Toxicity Endpoints

This script trains all 5 baseline ML models (KNN, NN, RF, SVM, XGBoost) 
across all 12 toxicity endpoints using ECFP4 fingerprints.

Total: 60 models (5 models × 12 toxicity endpoints)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

from train_baseline_models import BaselineModelTrainer


def train_all_toxicities():
    """
    Train baseline models for all 12 toxicity endpoints.
    """
    # All toxicity endpoints
    toxicity_endpoints = [
        'NR-AhR',
        'NR-AR',
        'NR-AR-LBD',
        'NR-Aromatase',
        'NR-ER',
        'NR-ER-LBD',
        'NR-PPAR-gamma',
        'SR-ARE',
        'SR-ATAD5',
        'SR-HSE',
        'SR-MMP',
        'SR-p53'
    ]
    
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS FOR ALL TOXICITY ENDPOINTS")
    print("="*80)
    print(f"\nTotal toxicity endpoints: {len(toxicity_endpoints)}")
    print(f"Models per endpoint: 5 (KNN, NN, RF, SVM, XGBoost)")
    print(f"Total models to train: {len(toxicity_endpoints) * 5}")
    print(f"\nEndpoints: {', '.join(toxicity_endpoints)}")
    print("="*80 + "\n")
    
    # Track overall results
    overall_results = []
    start_time = time.time()
    
    # Train each toxicity endpoint
    for idx, toxicity in enumerate(toxicity_endpoints, 1):
        print(f"\n{'#'*80}")
        print(f"# [{idx}/{len(toxicity_endpoints)}] PROCESSING: {toxicity}")
        print(f"{'#'*80}\n")
        
        toxicity_start_time = time.time()
        
        try:
            # Create trainer
            trainer = BaselineModelTrainer(
                toxicity_name=toxicity,
                data_dir='Data/csv',
                results_dir='results/baseline_models',
                models_dir='models/baseline_models',
                seed=42
            )
            
            # Train all models for this toxicity
            results = trainer.train_all_models()
            
            # Collect results
            for model_name, model_results in results.items():
                overall_results.append({
                    'Toxicity': toxicity,
                    'Model': model_name,
                    'CV_ROC_AUC': model_results['cv_score'],
                    'Val_ROC_AUC': model_results['val_results']['auc'],
                    'Test_ROC_AUC': model_results['test_results']['auc'],
                    'Test_Accuracy': model_results['test_results']['accuracy'],
                    'Test_Precision': model_results['test_results']['precision'],
                    'Test_Recall': model_results['test_results']['recall'],
                    'Test_F1': model_results['test_results']['f1']
                })
            
            toxicity_elapsed = time.time() - toxicity_start_time
            print(f"\n{toxicity} completed in {toxicity_elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"\nERROR processing {toxicity}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create overall summary
    print("\n" + "="*80)
    print("CREATING OVERALL SUMMARY")
    print("="*80 + "\n")
    
    if overall_results:
        overall_df = pd.DataFrame(overall_results)
        
        # Save overall summary
        summary_path = 'results/baseline_models/overall_summary.csv'
        overall_df.to_csv(summary_path, index=False)
        print(f"Overall summary saved to: {summary_path}")
        
        # Display summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80 + "\n")
        
        # Best model per toxicity
        print("Best Model per Toxicity (by Test ROC-AUC):")
        print("-" * 80)
        best_per_toxicity = overall_df.loc[overall_df.groupby('Toxicity')['Test_ROC_AUC'].idxmax()]
        print(best_per_toxicity[['Toxicity', 'Model', 'Test_ROC_AUC']].to_string(index=False))
        
        # Average performance by model type
        print("\n" + "-" * 80)
        print("Average Performance by Model Type:")
        print("-" * 80)
        model_avg = overall_df.groupby('Model')[
            ['CV_ROC_AUC', 'Val_ROC_AUC', 'Test_ROC_AUC', 'Test_Accuracy', 'Test_F1']
        ].mean()
        model_avg = model_avg.sort_values('Test_ROC_AUC', ascending=False)
        print(model_avg.to_string())
        
        # Overall best model
        print("\n" + "-" * 80)
        print("Top 10 Overall Best Models:")
        print("-" * 80)
        top_models = overall_df.nlargest(10, 'Test_ROC_AUC')
        print(top_models[['Toxicity', 'Model', 'Test_ROC_AUC', 'Test_F1']].to_string(index=False))
        
        # Save model averages
        model_avg_path = 'results/baseline_models/model_averages.csv'
        model_avg.to_csv(model_avg_path)
        print(f"\nModel averages saved to: {model_avg_path}")
        
        # Create detailed report
        create_detailed_report(overall_df)
    
    # Calculate total time
    total_elapsed = time.time() - start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal models trained: {len(overall_results)}")
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Results directory: results/baseline_models/")
    print(f"Models directory: models/baseline_models/")
    print("="*80 + "\n")


def create_detailed_report(overall_df):
    """
    Create a detailed report of all results.
    
    Args:
        overall_df: DataFrame with all results
    """
    report_path = 'results/baseline_models/detailed_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE MODELS DETAILED REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models: {len(overall_df)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS BY TOXICITY ENDPOINT\n")
        f.write("="*80 + "\n\n")
        
        for toxicity in overall_df['Toxicity'].unique():
            f.write(f"\n{toxicity}\n")
            f.write("-" * 80 + "\n")
            toxicity_df = overall_df[overall_df['Toxicity'] == toxicity]
            toxicity_df = toxicity_df.sort_values('Test_ROC_AUC', ascending=False)
            f.write(toxicity_df[['Model', 'Test_ROC_AUC', 'Test_Accuracy', 'Test_F1']].to_string(index=False))
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RESULTS BY MODEL TYPE\n")
        f.write("="*80 + "\n\n")
        
        for model in overall_df['Model'].unique():
            f.write(f"\n{model}\n")
            f.write("-" * 80 + "\n")
            model_df = overall_df[overall_df['Model'] == model]
            model_df = model_df.sort_values('Test_ROC_AUC', ascending=False)
            f.write(model_df[['Toxicity', 'Test_ROC_AUC', 'Test_Accuracy', 'Test_F1']].to_string(index=False))
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Best model per toxicity
        f.write("Best Model per Toxicity (by Test ROC-AUC):\n")
        f.write("-" * 80 + "\n")
        best_per_toxicity = overall_df.loc[overall_df.groupby('Toxicity')['Test_ROC_AUC'].idxmax()]
        f.write(best_per_toxicity[['Toxicity', 'Model', 'Test_ROC_AUC']].to_string(index=False))
        f.write("\n\n")
        
        # Average by model
        f.write("Average Performance by Model Type:\n")
        f.write("-" * 80 + "\n")
        model_avg = overall_df.groupby('Model')[
            ['Test_ROC_AUC', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
        ].mean()
        model_avg = model_avg.sort_values('Test_ROC_AUC', ascending=False)
        f.write(model_avg.to_string())
        f.write("\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Test ROC-AUC: {overall_df['Test_ROC_AUC'].mean():.4f} ± {overall_df['Test_ROC_AUC'].std():.4f}\n")
        f.write(f"Average Test Accuracy: {overall_df['Test_Accuracy'].mean():.4f} ± {overall_df['Test_Accuracy'].std():.4f}\n")
        f.write(f"Average Test F1: {overall_df['Test_F1'].mean():.4f} ± {overall_df['Test_F1'].std():.4f}\n")
        f.write(f"Best Overall ROC-AUC: {overall_df['Test_ROC_AUC'].max():.4f}\n")
        f.write(f"Worst Overall ROC-AUC: {overall_df['Test_ROC_AUC'].min():.4f}\n")
    
    print(f"Detailed report saved to: {report_path}")


if __name__ == '__main__':
    train_all_toxicities()
