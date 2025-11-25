"""
Quick test script to verify the research metrics endpoint logic
"""

import json
from pathlib import Path

def test_metrics_loading():
    results_dir = Path(__file__).parent.parent.parent / "results"
    
    print(f"Results directory: {results_dir}")
    print(f"Results directory exists: {results_dir.exists()}")
    
    # Test GCN results loading
    overall_summary_path = results_dir / "overall_summary.csv"
    print(f"\nGCN Summary: {overall_summary_path.exists()}")
    
    # Test baseline results
    baseline_dir = results_dir / "baseline_models"
    print(f"Baseline dir exists: {baseline_dir.exists()}")
    
    if baseline_dir.exists():
        toxicities = [d.name for d in baseline_dir.iterdir() if d.is_dir()]
        print(f"Baseline toxicities: {toxicities}")
        
        # Test ROC data loading
        roc_count = 0
        for toxicity in toxicities:
            for model in ['RF', 'XGBoost', 'SVM', 'NN', 'KNN']:
                json_file = baseline_dir / toxicity / f"{model}_results.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'test_results' in data and 'probabilities' in data['test_results']:
                            roc_count += 1
                            print(f"  ✓ {toxicity}/{model}: {len(data['test_results']['probabilities'])} samples")
        
        print(f"\nTotal ROC data files: {roc_count}")

if __name__ == "__main__":
    test_metrics_loading()
