# Quick Start: Launch ToxPredict Web Application

## Prerequisites
- Python 3.8+ installed
- Node.js 16+ installed
- Dependencies installed (see below)

## Installation (First Time Only)

### Backend Setup
```powershell
# Navigate to backend
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\backend

# Install dependencies
pip install fastapi uvicorn[standard] Pillow python-multipart rdkit-pypi
```

### Frontend Setup
```powershell
# Navigate to frontend
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\frontend

# Install dependencies (already done)
npm install
```

## Launch the Application

### Terminal 1: Start Backend Server
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ Backend is ready at: **http://localhost:8000**
📚 API Docs available at: **http://localhost:8000/docs**

### Terminal 2: Start Frontend Server
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\frontend
npm run dev
```

**Expected Output:**
```
VITE v4.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

✅ Frontend is ready at: **http://localhost:5173**

### Terminal 3: Open in Browser
```powershell
# Open the application
start http://localhost:5173
```

## Testing the Application

### 1. Test SMILES Validation

**Valid SMILES Examples:**
- `CCO` (Ethanol)
- `c1ccccc1` (Benzene)
- `CC(=O)Oc1ccccc1C(=O)O` (Aspirin)
- `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` (Caffeine)

**Invalid SMILES (Should show error, not crash):**
- `XXXXX`
- `invalid123`
- `12345`

### 2. Test Endpoint Presets

Click these preset buttons to select multiple endpoints:
- **All Toxicities** → Selects all 12 endpoints
- **Nuclear Receptor Panel** → 7 endpoints
- **Stress Response Panel** → 5 endpoints
- **Environmental Toxicants** → 4 endpoints (NR-AhR, NR-ER, NR-AR, SR-ARE)
- **Endocrine Disruption** → 4 endpoints (NR-AR, NR-ER, NR-Aromatase, NR-PPAR-gamma)

### 3. Test Baseline Model Comparison

1. Enter valid SMILES: `CCO`
2. Click "Validate SMILES"
3. Select endpoints (use a preset)
4. ✅ **Check "Compare with Baseline Models"**
5. Click "Run Prediction"
6. After results load, click **"Show Comparison"** button
7. See all 5 baseline models (KNN, NN, RF, SVM, XGBoost) compared with GCN

### 4. Test Export Functionality

After predictions load:
- Click **"CSV"** button → Downloads `toxicity_prediction_[timestamp].csv`
- Click **"JSON"** button → Downloads `toxicity_prediction_[timestamp].json`

## Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'rdkit'`
```powershell
pip install rdkit-pypi
```

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`
```powershell
pip install fastapi uvicorn[standard]
```

**Problem:** Port 8000 already in use
```powershell
# Kill process on port 8000
$process = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
if ($process) { Stop-Process -Id $process -Force }

# Or use different port
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### Frontend Issues

**Problem:** `npm: command not found`
- Install Node.js from https://nodejs.org/

**Problem:** Port 5173 already in use
```powershell
# Frontend will automatically try next available port (5174, 5175, etc.)
# Or manually specify port:
npm run dev -- --port 3000
```

**Problem:** Blank page or errors
```powershell
# Clear cache and reinstall
rm -r node_modules
rm package-lock.json
npm install
npm run dev
```

### Connection Issues

**Problem:** Frontend can't connect to backend
1. Check backend is running on http://localhost:8000
2. Visit http://localhost:8000/api/health
3. Should see: `{"status":"healthy","timestamp":"...","models_loaded":false,"available_endpoints":12}`

## Features Overview

### ✅ Working Features
- ✅ SMILES validation with RDKit
- ✅ Molecular property calculation
- ✅ 2D molecule visualization
- ✅ 12 toxicity endpoint predictions
- ✅ GCN model predictions
- ✅ Baseline model comparison (KNN, NN, RF, SVM, XGBoost)
- ✅ Endpoint presets (all working!)
- ✅ CSV export with all predictions
- ✅ JSON export with full data
- ✅ Error handling (no crashes on invalid SMILES)
- ✅ Responsive design
- ✅ Dark academic theme

### 🚧 Coming Soon
- 🚧 Batch SMILES upload (CSV/SMI files)
- 🚧 Molecule drawing tool
- 🚧 Model explainability (attention maps, SHAP)
- 🚧 Real ML models (currently using mock predictions)

## Quick Command Reference

### Start Everything (Copy-Paste Ready)

**Terminal 1 - Backend:**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\backend; uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp\frontend; npm run dev
```

**Terminal 3 - Open Browser:**
```powershell
start http://localhost:5173
```

## Keyboard Shortcuts

While in the application:
- **Enter** in SMILES input → Validates SMILES
- **Ctrl+C** in terminals → Stops servers
- **F5** in browser → Refreshes page
- **Ctrl+Shift+I** → Open browser DevTools for debugging

## Development Notes

- Backend auto-reloads on file changes (`--reload` flag)
- Frontend auto-reloads with HMR (Hot Module Replacement)
- Changes to Python code reflected immediately
- Changes to React components reflected immediately
- No need to restart servers for code changes!

## API Endpoints

Visit **http://localhost:8000/docs** for interactive API documentation.

Key endpoints:
- `GET /api/health` - Health check
- `GET /api/endpoints` - List all toxicity endpoints
- `POST /api/validate` - Validate SMILES
- `POST /api/predict` - Predict toxicity
- `GET /api/models` - List available models

## File Locations

- **Backend:** `webapp/backend/app.py`
- **Frontend Components:** `webapp/frontend/src/components/`
- **Frontend Pages:** `webapp/frontend/src/pages/`
- **API Utils:** `webapp/frontend/src/utils/api.js`
- **Styles:** `webapp/frontend/src/index.css`

## Support

For issues or questions:
1. Check terminal output for errors
2. Check browser console (F12) for frontend errors
3. Visit http://localhost:8000/docs for API testing
4. Check `IMPLEMENTATION_NOTES.md` for technical details

---

**Ready to go!** 🚀

Just run the two terminals (backend & frontend) and open http://localhost:5173 in your browser.
