# ✅ ToxPredict Setup Checklist

Use this checklist to set up and verify your ToxPredict web application.

---

## 📋 Pre-Installation Checklist

### System Requirements

- [ ] **Operating System**: Windows 10/11
- [ ] **Python 3.8+** installed
  - [ ] Check: Open PowerShell and run `python --version`
  - [ ] If not installed: Download from https://python.org/downloads/
  - [ ] ⚠️ Make sure "Add Python to PATH" is checked during installation

- [ ] **Node.js 16+** installed
  - [ ] Check: Run `node --version` in PowerShell
  - [ ] If not installed: Download from https://nodejs.org/

- [ ] **PowerShell** available (default on Windows)
  - [ ] Check: Search "PowerShell" in Start Menu

- [ ] **Git** (optional, for version control)
  - [ ] Check: Run `git --version`
  - [ ] Download from https://git-scm.com/

---

## 🚀 Installation Steps

### Step 1: Navigate to Project
- [ ] Open PowerShell
- [ ] Navigate to webapp directory:
  ```powershell
  cd C:\Users\geeta\OneDrive\Attachments\Desktop\SSL-GCN\webapp
  ```

### Step 2: Enable Script Execution (if needed)
- [ ] Run the following command if you get "script execution disabled" error:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- [ ] Type `Y` to confirm

### Step 3: Run Setup Script
- [ ] Execute setup script:
  ```powershell
  .\setup.ps1
  ```
- [ ] Wait for:
  - [ ] Python virtual environment creation
  - [ ] Backend dependency installation (~2-3 minutes)
  - [ ] Node.js dependency installation (~2-3 minutes)

### Step 4: Verify Installation
- [ ] Setup script completed without errors
- [ ] `webapp/backend/venv` folder exists
- [ ] `webapp/frontend/node_modules` folder exists

---

## 🎮 Running the Application

### Terminal 1: Backend Server

- [ ] Open PowerShell in `webapp` directory
- [ ] Run backend:
  ```powershell
  .\start-backend.ps1
  ```
- [ ] Verify you see:
  ```
  INFO:     Uvicorn running on http://0.0.0.0:8000
  INFO:     Application startup complete
  ```
- [ ] Test backend health:
  - [ ] Open browser to http://localhost:8000/api/health
  - [ ] Should see: `{"status": "healthy"}`

### Terminal 2: Frontend Server

- [ ] Open **NEW** PowerShell window in `webapp` directory
- [ ] Run frontend:
  ```powershell
  .\start-frontend.ps1
  ```
- [ ] Verify you see:
  ```
  VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
  ```
- [ ] Browser should auto-open to http://localhost:5173

---

## 🧪 Testing the Application

### Basic Functionality Tests

#### Test 1: Homepage Loads
- [ ] Navigate to http://localhost:5173
- [ ] Page loads without errors
- [ ] See "ToxPredict" in the navbar
- [ ] See SMILES input section

#### Test 2: SMILES Validation
- [ ] Click on "Ethanol" example button
- [ ] SMILES field shows: `CCO`
- [ ] Click "Validate SMILES"
- [ ] See green checkmark
- [ ] Molecular properties appear:
  - [ ] Molecular Weight: ~46.07
  - [ ] LogP: ~-0.07
  - [ ] 2D molecule image shows

#### Test 3: Endpoint Selection
- [ ] Scroll to "Select Toxicity Endpoints" section
- [ ] Click "All Endpoints" preset
- [ ] All 12 endpoints selected (checkboxes checked)
- [ ] Counter shows "12 endpoints selected"

#### Test 4: Prediction
- [ ] With validated SMILES and endpoints selected
- [ ] Click "Predict Toxicity" button
- [ ] Wait for prediction (~1-2 seconds)
- [ ] Results section appears with:
  - [ ] Overall summary cards (Total, Toxic, Risk Level)
  - [ ] Detailed predictions table
  - [ ] Molecular properties grid

#### Test 5: Download Results
- [ ] In results section, click "Download Results"
- [ ] JSON file downloads
- [ ] File contains SMILES, predictions, properties

### Navigation Tests

- [ ] Click "Explainability" in navbar
  - [ ] Shows "Under Development" message
  - [ ] Lists planned features

- [ ] Click "Research" in navbar
  - [ ] Shows research portal placeholder
  - [ ] Lists model architecture info

- [ ] Click "About" in navbar
  - [ ] Shows project information
  - [ ] Technology stack listed
  - [ ] Mission statement visible

### API Documentation Test

- [ ] Navigate to http://localhost:8000/docs
- [ ] Swagger UI loads
- [ ] See list of 11 API endpoints
- [ ] Try "Validate SMILES" endpoint:
  - [ ] Click "Try it out"
  - [ ] Enter SMILES: `CCO`
  - [ ] Click "Execute"
  - [ ] See 200 response with validation result

---

## 🐛 Troubleshooting Checklist

### Backend Issues

#### Problem: "Python not found"
- [ ] Install Python from https://python.org/downloads/
- [ ] Make sure "Add to PATH" is checked
- [ ] Restart PowerShell

#### Problem: "Cannot activate virtual environment"
- [ ] Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- [ ] Confirm with `Y`
- [ ] Try setup again

#### Problem: "Port 8000 in use"
- [ ] Find what's using port 8000:
  ```powershell
  netstat -ano | findstr :8000
  ```
- [ ] Kill the process or change port in `start-backend.ps1`

#### Problem: "RDKit import error"
- [ ] Activate virtual environment:
  ```powershell
  cd webapp\backend
  .\venv\Scripts\Activate.ps1
  ```
- [ ] Install RDKit:
  ```powershell
  pip install rdkit-pypi
  ```

### Frontend Issues

#### Problem: "Node not found"
- [ ] Install Node.js from https://nodejs.org/
- [ ] Restart PowerShell

#### Problem: "Module not found" errors
- [ ] Navigate to frontend:
  ```powershell
  cd webapp\frontend
  ```
- [ ] Reinstall dependencies:
  ```powershell
  npm install
  ```

#### Problem: "Blank page" in browser
- [ ] Open browser developer tools (F12)
- [ ] Check Console tab for errors
- [ ] Verify backend is running (http://localhost:8000/api/health)

#### Problem: "API connection refused"
- [ ] Check backend is running on port 8000
- [ ] Check `frontend/vite.config.js` proxy settings
- [ ] Try accessing backend directly: http://localhost:8000/api/health

---

## 📊 Verification Checklist

### Backend Verification

- [ ] Health endpoint responds: http://localhost:8000/api/health
- [ ] Endpoints list loads: http://localhost:8000/api/endpoints
- [ ] Models list loads: http://localhost:8000/api/models
- [ ] API docs load: http://localhost:8000/docs
- [ ] CORS headers present (check Network tab in browser)

### Frontend Verification

- [ ] Homepage renders correctly
- [ ] Navigation works (all 4 pages accessible)
- [ ] SMILES validation works
- [ ] Endpoint selection works
- [ ] Predictions display (mock data)
- [ ] Responsive design (resize browser window)
- [ ] No console errors in browser (F12)

---

## 🎯 Success Criteria

You've successfully set up ToxPredict if:

✅ Both backend and frontend servers run without errors
✅ You can validate SMILES strings
✅ You can select toxicity endpoints
✅ You can see prediction results (mock data)
✅ All navigation links work
✅ API documentation is accessible
✅ No errors in browser console

---

## 📝 Next Steps After Setup

Once everything is working:

### 1. Explore the Application
- [ ] Try different SMILES examples
- [ ] Test different endpoint combinations
- [ ] Download prediction results
- [ ] Check all pages

### 2. Review Documentation
- [ ] Read `webapp/README.md` for full documentation
- [ ] Review `webapp/BUILD_SUMMARY.md` for what was built
- [ ] Check API documentation at http://localhost:8000/docs

### 3. Train Models (When Ready)
- [ ] Navigate to project root
- [ ] Train GCN models:
  ```powershell
  python src/train.py --toxicity NR-AhR --model gcn
  ```
- [ ] Train baseline models:
  ```powershell
  python src/train_model_knn.py --toxicity NR-AhR
  python src/train_model_nn.py --toxicity NR-AhR
  # etc.
  ```

### 4. Integrate Real Models (Advanced)
- [ ] Update `backend/app.py` to load trained models
- [ ] Replace `mock_prediction()` function
- [ ] Test with real predictions
- [ ] See `webapp/README.md` section "Integrating Trained Models"

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check Browser Console** (F12)
   - Look for JavaScript errors
   - Check Network tab for failed API calls

2. **Check Backend Terminal**
   - Look for Python errors
   - Check for missing dependencies

3. **Review Documentation**
   - `webapp/GETTING_STARTED.md` - Quick start guide
   - `webapp/README.md` - Full documentation
   - `webapp/BUILD_SUMMARY.md` - What was built

4. **Common Solutions**
   - Restart both servers
   - Clear browser cache (Ctrl+Shift+Delete)
   - Reinstall dependencies
   - Check firewall settings

---

## 📅 Maintenance Checklist

### Daily Use
- [ ] Keep both servers running while using
- [ ] Don't close terminal windows
- [ ] Use Ctrl+C to stop servers gracefully

### Updates
- [ ] Update Python packages: `pip install --upgrade -r requirements.txt`
- [ ] Update npm packages: `npm update` (in frontend directory)

### Backups
- [ ] Save trained models regularly
- [ ] Backup prediction results
- [ ] Version control with Git (recommended)

---

## ✨ Congratulations!

If you've checked all the boxes, your ToxPredict application is ready to use! 🎉

**Quick Access URLs:**
- 🌐 Frontend: http://localhost:5173
- 🔌 Backend API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

**Happy Predicting! 🧪🔬**
