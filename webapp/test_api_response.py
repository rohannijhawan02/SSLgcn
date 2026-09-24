# Test the API endpoint response structure
# Run this after the backend is up

import requests
import json

url = "http://localhost:8000/api/predict"
payload = {
    "smiles": "CCO",
    "endpoints": ["NR-AhR", "NR-AR"],
    "compare_baseline": True
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    data = response.json()
    
    print("="*60)
    print("API Response Structure")
    print("="*60)
    print(f"\nSMILES: {data.get('smiles')}")
    print(f"Compare Baseline: {data.get('compare_baseline')}")
    print(f"\nNumber of GCN Predictions: {len(data.get('predictions', []))}")
    
    print("\n--- GCN Predictions ---")
    for pred in data.get('predictions', []):
        print(f"  Endpoint: {pred['endpoint']} -> {pred['prediction']}")
    
    print("\n--- Baseline Predictions ---")
    baseline = data.get('baseline_predictions', {})
    if baseline:
        print(f"Type: {type(baseline)}")
        print(f"Keys: {list(baseline.keys())}")
        print(f"\nStructure:")
        for endpoint, models in baseline.items():
            print(f"\n  {endpoint}:")
            for model, prediction in models.items():
                print(f"    {model}: {prediction}")
    else:
        print("  None or empty")
    
    print("\n" + "="*60)
    print("Full JSON Response:")
    print("="*60)
    print(json.dumps(data, indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
