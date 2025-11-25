# 🚀 ToxPredict Web Application - Getting Started

Welcome to ToxPredict! This guide will help you set up and run the web application on your Windows machine.

## 📋 Prerequisites

Before you begin, make sure you have:

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download here](https://nodejs.org/)
3. **Git** (optional) - For cloning the repository
4. **PowerShell** - Available by default on Windows

## 🎯 Quick Start (3 Steps)

### Step 1: Run Setup Script

Open PowerShell in the `webapp` directory and run:

```powershell
.\setup.ps1
```

This script will:
- Check Python and Node.js installation
- Create a Python virtual environment
- Install backend dependencies (FastAPI, RDKit, etc.)
- Install frontend dependencies (React, TailwindCSS, etc.)

### Step 2: Start Backend Server

In the same PowerShell window (or a new one):

```powershell
.\start-backend.ps1
```

This will start the FastAPI server at: **http://localhost:8000**

### Step 3: Start Frontend (in a NEW PowerShell window)

Open a **new** PowerShell window in the `webapp` directory:

```powershell
.\start-frontend.ps1
```

This will start the React app at: **http://localhost:5173**

### 🎉 Done!

Open your browser and go to: **http://localhost:5173**

## 📱 Manual Setup (Alternative)

If you prefer to set up manually or the scripts don't work:

### Backend Manual Setup

```powershell
# Navigate to backend directory
cd webapp/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Manual Setup

```powershell
# Navigate to frontend directory (in a NEW terminal)
cd webapp/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🧪 Testing the Application

Once both servers are running:

1. **Open Browser**: Go to http://localhost:5173

2. **Enter a SMILES String**: Try one of these examples:
   - Ethanol: `CCO`
   - Benzene: `c1ccccc1`
   - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
   - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`

3. **Validate SMILES**: Click "Validate SMILES"
   - You should see molecular properties
   - A 2D structure image will appear

4. **Select Endpoints**: Choose toxicity endpoints to test
   - Try "All Endpoints" preset
   - Or select individual endpoints

5. **Predict**: Click "Predict Toxicity"
   - Results will show predictions for each endpoint
   - View confidence scores and molecular properties

## 🔍 API Testing

You can also test the backend API directly:

1. **API Documentation**: http://localhost:8000/docs
   - Interactive Swagger UI
   - Try out all endpoints

2. **Health Check**:
   ```powershell
   curl http://localhost:8000/api/health
   ```

3. **Validate SMILES**:
   ```powershell
   curl -X POST "http://localhost:8000/api/validate" `
   -H "Content-Type: application/json" `
   -d '{\"smiles\": \"CCO\"}'
   ```

## 🎨 Features Available

### ✅ Implemented
- SMILES input and validation
- Real-time molecular property calculation
- 2D structure visualization
- 12 toxicity endpoint selection
- Mock predictions (for testing UI)
- Dark academic theme
- Responsive design
- API documentation

### 🚧 Coming Soon
- Batch upload (CSV/Excel)
- Molecule drawing tool
- Real model predictions (GCN + baselines)
- Attention visualization
- SHAP value explainability
- Prediction history
- User authentication

## 📂 Project Structure

```
webapp/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── venv/              # Virtual environment (created by setup)
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── utils/         # Utilities (API client)
│   │   ├── App.jsx        # Main app component
│   │   └── main.jsx       # Entry point
│   ├── package.json
│   ├── node_modules/      # Node dependencies (created by setup)
│   └── vite.config.js
├── README.md              # Full documentation
├── GETTING_STARTED.md     # This file
├── setup.ps1              # Setup script
├── start-backend.ps1      # Start backend
└── start-frontend.ps1     # Start frontend
```

## 🐛 Troubleshooting

### Issue: Python not found
**Solution**: Install Python 3.8+ from https://www.python.org/downloads/
Make sure to check "Add Python to PATH" during installation.

### Issue: Node not found
**Solution**: Install Node.js 16+ from https://nodejs.org/

### Issue: Cannot activate virtual environment
**Solution**: You may need to enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Port already in use
**Backend (8000)**: Change port in start-backend.ps1:
```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

**Frontend (5173)**: Vite will automatically use the next available port.

### Issue: RDKit import error
**Solution**: 
```powershell
cd webapp\backend
.\venv\Scripts\Activate.ps1
pip install rdkit-pypi
```

### Issue: Frontend blank page
**Solution**: Check browser console (F12) for errors. Make sure backend is running.

### Issue: API connection refused
**Solution**: 
1. Make sure backend is running on port 8000
2. Check `frontend/vite.config.js` proxy settings
3. Try accessing http://localhost:8000/api/health directly

## 🔧 Configuration

### Change API Port

Edit `start-backend.ps1` and `frontend/vite.config.js`:

```javascript
// frontend/vite.config.js
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:YOUR_PORT',  // Change this
        // ...
      }
    }
  }
})
```

### Change Frontend Port

Edit `frontend/vite.config.js`:

```javascript
export default defineConfig({
  server: {
    port: YOUR_PORT,  // Change this
    // ...
  }
})
```

### Customize Theme

Edit `frontend/tailwind.config.js` to change colors, fonts, etc.

## 📚 Next Steps

1. **Explore the UI**: Try all the example SMILES strings

2. **Check API Docs**: Visit http://localhost:8000/docs

3. **Train Models**: Run the training scripts in `src/` to train real models
   ```powershell
   cd ../..  # Go to project root
   python src/train.py --toxicity NR-AhR --model gcn
   ```

4. **Integrate Models**: Update `backend/app.py` to load trained models
   (See webapp/README.md for details)

5. **Customize**: Modify colors, add features, or extend functionality

## 📖 Additional Resources

- [Full Documentation](./README.md)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [TailwindCSS Docs](https://tailwindcss.com/)
- [RDKit Docs](https://www.rdkit.org/docs/)

## 💡 Tips

- **Keep both terminals open**: Backend and frontend run in separate processes
- **Use Chrome DevTools**: F12 to debug frontend issues
- **Check backend logs**: Look at the terminal running the backend for API errors
- **Hot reload**: Both servers support hot reload - changes are reflected automatically
- **API testing**: Use Postman or curl to test API endpoints directly

## 🎓 Learn More

The web application is built with:
- **Backend**: FastAPI (Python) - Modern, fast web framework
- **Frontend**: React 18 - Popular UI library
- **Styling**: TailwindCSS - Utility-first CSS framework
- **Chemistry**: RDKit - Cheminformatics toolkit
- **Build Tool**: Vite - Fast frontend build tool

## 📧 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review the full README.md
3. Check browser console (F12) for frontend errors
4. Check backend terminal for API errors
5. Make sure all dependencies are installed

---

**Happy predicting! 🧪🔬**
