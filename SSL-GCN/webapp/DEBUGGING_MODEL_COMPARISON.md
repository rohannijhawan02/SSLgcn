# Debugging Model Comparison Issue

## Steps to Debug

### 1. Make Sure Backend is Running
Open a PowerShell terminal and run:
```powershell
.\webapp\start-backend.ps1
```
Keep this terminal open.

### 2. Make Sure Frontend is Running  
Open another PowerShell terminal and run:
```powershell
.\webapp\start-frontend.ps1
```
Keep this terminal open.

### 3. Test in Browser
1. Go to http://localhost:3000
2. Open Browser DevTools (Press F12)
3. Go to the Console tab
4. Enter a SMILES string (e.g., `CCO`)
5. Click "Validate SMILES"
6. Select some endpoints (e.g., NR-AhR, NR-AR)
7. **Make sure to CHECK the "Compare with Baseline Models" checkbox**
8. Click "Predict Toxicity"

### 4. Check Console Output
Look for these console.log messages:

```
PredictionResults received: {
  compare_baseline: true,
  baseline_predictions: { ... },
  predictions: [...]
}
```

**Expected structure of baseline_predictions:**
```javascript
{
  "NR-AhR": {
    "knn": "Toxic",
    "neural_network": "Non-toxic",
    "random_forest": "Toxic",
    "svm": "Non-toxic",
    "xgboost": "Toxic"
  },
  "NR-AR": {
    "knn": "Non-toxic",
    ...
  }
}
```

Then look for:
```
First prediction endpoint: NR-AhR
Baseline predictions keys: ["NR-AhR", "NR-AR"]
Baseline pred for this endpoint: { knn: "Toxic", neural_network: "Non-toxic", ... }
```

### 5. Common Issues to Check

#### Issue 1: Checkbox not checked
- **Symptom:** `compare_baseline: false` in console
- **Fix:** Make sure you checked the "Compare with Baseline Models" checkbox before clicking Predict

#### Issue 2: baseline_predictions is null
- **Symptom:** `baseline_predictions: null` in console
- **Cause:** Backend didn't return baseline predictions
- **Fix:** Check backend terminal for errors

#### Issue 3: baseline_predictions is empty object {}
- **Symptom:** `baseline_predictions: {}` with no keys
- **Cause:** Backend loop didn't execute
- **Fix:** Check if endpoints array is populated

#### Issue 4: Endpoint mismatch
- **Symptom:** `Baseline pred for this endpoint: undefined`
- **Cause:** The endpoint ID from predictions doesn't match the keys in baseline_predictions
- **Example:** pred.endpoint = "NR_AhR" but baseline key is "NR-AhR"
- **Fix:** Check that endpoints are consistent (hyphen vs underscore)

### 6. Manual API Test
If the frontend isn't working, test the API directly:

Open a terminal (with backend running) and run:
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"smiles": "CCO", "endpoints": ["NR-AhR", "NR-AR"], "compare_baseline": true}' `
  | Select-Object -ExpandProperty Content
```

This should return JSON with baseline_predictions populated.

### 7. Check Network Tab
1. Open DevTools -> Network tab
2. Make a prediction request
3. Click on the `/api/predict` request
4. Go to "Response" tab
5. Look at the `baseline_predictions` field
6. Should see:
```json
{
  "baseline_predictions": {
    "NR-AhR": {
      "knn": "Toxic",
      "neural_network": "Non-toxic",
      ...
    }
  },
  "compare_baseline": true
}
```

### 8. Expected UI Behavior
When everything works:
- ✅ "Model Comparison" section appears below the GCN predictions
- ✅ Table has 7 columns: Endpoint, GCN, XGBoost, Random Forest, SVM, Neural Network, KNN
- ✅ All cells show either "Toxic" (red) or "Non-toxic" (green)
- ✅ No cells show "-" (dash)
- ✅ Button says "Hide Comparison" (since it's shown by default)

### 9. If Still Seeing Dashes (-)
The dashes appear when `baselinePred` is undefined or falsy. This means:
- Either `baseline_predictions[pred.endpoint]` is returning undefined
- Which means the endpoint key doesn't exist in the baseline_predictions object

**Check this in console:**
```javascript
// In the console, after making a prediction, you should see:
// First prediction endpoint: [some endpoint ID]
// Baseline predictions keys: [array of endpoint IDs]
// These should match!
```

### 10. Quick Fix Test
If you're still seeing issues, try adding this temporary debugging code in the browser console after predictions load:

```javascript
// Copy the results data and inspect it
console.log('Checking data structure...');
const results = /* your results object */;
console.log('Predictions:', results.predictions.map(p => p.endpoint));
console.log('Baseline keys:', Object.keys(results.baseline_predictions || {}));
console.log('Do they match?', 
  results.predictions.every(p => 
    results.baseline_predictions && results.baseline_predictions[p.endpoint]
  )
);
```

## Current Code Changes Made

1. **Backend** (`webapp/backend/app.py` line 408-424):
   - Changed baseline_predictions structure from model-based to endpoint-based
   - Now returns predictions organized by endpoint ID

2. **Frontend** (`webapp/frontend/src/components/PredictionResults.jsx`):
   - Added KNN column to comparison table
   - Added debug console.log statements
   - Added fallback rendering for missing data (dashes)
   - Set showComparison to true by default

3. **Frontend** (`webapp/frontend/src/pages/HomePage.jsx`):
   - Improved error handling for validation errors
   - Added getErrorMessage utility

## Need More Help?
Share the console output from the browser DevTools and I can help diagnose further!
