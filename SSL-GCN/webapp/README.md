# ToxPredict Web Application

A research-grade toxicity prediction web application using Graph Convolutional Networks (GCN) and Baseline Machine Learning models.

## 🚀 QUICK START

**Fastest way to start the application:**

1. **Double-click:** `START_ALL.bat` (Windows)
2. **Wait** for both servers to start
3. **Open browser:** http://localhost:3000

📖 **Need more details?** See:
- [QUICK_START.md](QUICK_START.md) - Quick reference card
- [HOW_TO_START.md](HOW_TO_START.md) - Complete guide with troubleshooting
- [START_GUIDE.txt](START_GUIDE.txt) - Visual guide

---

## 🎯 Overview

This web application provides an intuitive interface for predicting molecular toxicity across 12 biological endpoints using state-of-the-art machine learning models.

### Features
- ✅ **SMILES Input**: Enter molecular structures as SMILES strings
- ✅ **12 Toxicity Endpoints**: Predict across Nuclear Receptor and Stress Response pathways
- ✅ **Multiple Models**: GCN, KNN, MLP, Random Forest, SVM, XGBoost
- ✅ **Real-time Validation**: Instant SMILES validation with molecular properties
- ✅ **Dark Academic Theme**: Professional research-grade interface
- 🚧 **Batch Upload**: Coming soon
- 🚧 **Molecule Drawing**: Coming soon
- 🚧 **Model Explainability**: Attention visualization, SHAP values (coming soon)

## 📁 Project Structure

```
webapp/
├── backend/
│   ├── app.py                 # FastAPI application
│   └── requirements.txt       # Python dependencies
└── frontend/
    ├── src/
    │   ├── components/        # React components
    │   │   ├── Navbar.jsx
    │   │   ├── SMILESInput.jsx
    │   │   ├── EndpointSelector.jsx
    │   │   └── PredictionResults.jsx
    │   ├── pages/            # Page components
    │   │   ├── HomePage.jsx
    │   │   ├── ExplainabilityPage.jsx
    │   │   ├── ResearchPage.jsx
    │   │   └── AboutPage.jsx
    │   ├── utils/
    │   │   └── api.js        # API client
    │   ├── App.jsx           # Root component
    │   ├── main.jsx          # React entry point
    │   └── index.css         # TailwindCSS styles
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Backend Setup

1. **Navigate to backend directory**:
   ```powershell
   cd webapp/backend
   ```

2. **Create and activate virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```powershell
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at: http://localhost:8000
   Interactive API docs: http://localhost:8000/docs

### Frontend Setup

1. **Navigate to frontend directory** (in a new terminal):
   ```powershell
   cd webapp/frontend
   ```

2. **Install dependencies**:
   ```powershell
   npm install
   ```

3. **Start the development server**:
   ```powershell
   npm run dev
   ```

   The app will be available at: http://localhost:5173

## 🔗 API Endpoints

### Health Check
```
GET /api/health
```

### Get Available Endpoints
```
GET /api/endpoints
```

### Get Available Models
```
GET /api/models
```

### Validate SMILES
```
POST /api/validate
Body: { "smiles": "CCO" }
```

### Predict Toxicity
```
POST /api/predict
Body: {
  "smiles": "CCO",
  "endpoints": ["NR-AhR", "NR-ER"],
  "model": "gcn"
}
```

### Batch Prediction
```
POST /api/batch-predict
Body: {
  "smiles_list": ["CCO", "c1ccccc1"],
  "endpoints": ["NR-AhR"],
  "model": "gcn"
}
```

## 🎨 Technology Stack

### Frontend
- **React 18.2**: Modern UI library
- **Vite 5.0**: Fast build tool
- **TailwindCSS 3.3**: Utility-first CSS framework
- **React Router 6.20**: Client-side routing
- **Axios**: HTTP client
- **Lucide React**: Beautiful icons
- **React Hot Toast**: Toast notifications

### Backend
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation
- **RDKit**: Cheminformatics toolkit
- **Uvicorn**: ASGI server
- **Pillow**: Image generation

### Machine Learning (Ready for Integration)
- **PyTorch**: Deep learning framework for GCN
- **scikit-learn**: Baseline ML models
- **XGBoost**: Gradient boosting
- **NumPy/Pandas**: Data manipulation

## 🧪 Current State

### ✅ Completed
- Full backend API with 11 endpoints
- Complete frontend structure with routing
- SMILES input and validation
- Endpoint selection interface
- Results display with molecular properties
- Dark academic theme implementation
- Responsive design

### 🔄 In Progress
- Model integration (models trained separately)
- Database setup for predictions storage

### 🚧 Coming Soon
- Batch upload from CSV/Excel
- Molecule drawing tool
- Model explainability dashboard
- Attention heatmaps
- SHAP value visualization
- Performance metrics dashboard
- User authentication
- Prediction history

## 🔧 Configuration

### Backend Configuration
Edit `backend/app.py` to modify:
- CORS settings
- Model paths
- Endpoint definitions
- Validation rules

### Frontend Configuration
Edit `frontend/vite.config.js` to modify:
- API proxy settings
- Build options
- Port configuration

Edit `frontend/tailwind.config.js` to customize:
- Color scheme
- Typography
- Spacing
- Custom utilities

## 📊 Toxicity Endpoints

### Nuclear Receptor (NR)
1. **NR-AhR**: Aryl hydrocarbon Receptor
2. **NR-AR**: Androgen Receptor
3. **NR-AR-LBD**: Androgen Receptor Ligand Binding Domain
4. **NR-Aromatase**: Aromatase enzyme
5. **NR-ER**: Estrogen Receptor
6. **NR-ER-LBD**: Estrogen Receptor Ligand Binding Domain
7. **NR-PPAR-gamma**: Peroxisome Proliferator-Activated Receptor Gamma

### Stress Response (SR)
8. **SR-ARE**: Antioxidant Response Element
9. **SR-ATAD5**: ATPase Family AAA Domain-Containing Protein 5
10. **SR-HSE**: Heat Shock Element
11. **SR-MMP**: Mitochondrial Membrane Potential
12. **SR-p53**: Tumor Protein p53

## 🐛 Troubleshooting

### Backend Issues

**Problem**: RDKit import error
```
Solution: pip install rdkit-pypi
```

**Problem**: Port 8000 already in use
```
Solution: uvicorn app:app --reload --port 8001
```

### Frontend Issues

**Problem**: Module not found errors
```
Solution: 
cd frontend
npm install
```

**Problem**: API connection refused
```
Solution: Ensure backend is running on http://localhost:8000
Check vite.config.js proxy settings
```

**Problem**: TailwindCSS styles not loading
```
Solution:
npm run build
npm run dev
```

## 📝 Development Notes

### Adding New Endpoints
1. Update `TOXICITY_ENDPOINTS` in `backend/app.py`
2. Add endpoint data (CSV) to `Data/csv/`
3. Train models for the new endpoint
4. Update frontend endpoint list (auto-fetched from API)

### Integrating Trained Models

Currently using mock predictions. To integrate real models:

1. **Train models** using the training scripts in `src/`:
   ```powershell
   python src/train.py --toxicity NR-AhR --model gcn
   python src/train_model_knn.py --toxicity NR-AhR
   ```

2. **Update prediction functions** in `backend/app.py`:
   ```python
   # Replace mock_prediction() with:
   from model import GCNModel  # or other model classes
   
   def load_models():
       models = {}
       for endpoint in TOXICITY_ENDPOINTS:
           model_path = f"checkpoints/{endpoint['id']}/best_model.pt"
           models[endpoint['id']] = torch.load(model_path)
       return models
   
   def predict_toxicity(smiles, endpoint, model_type):
       # Load model
       # Generate features
       # Make prediction
       # Return result
   ```

3. **Test predictions**:
   ```powershell
   curl -X POST "http://localhost:8000/api/predict" \
   -H "Content-Type: application/json" \
   -d '{"smiles": "CCO", "endpoints": ["NR-AhR"], "model": "gcn"}'
   ```

## 🎓 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TailwindCSS Documentation](https://tailwindcss.com/docs)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Tox21 Challenge](https://tripod.nih.gov/tox21/challenge/)

## 📜 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Tox21 Challenge for providing the dataset
- RDKit community for cheminformatics tools
- Open source ML/DL libraries

## 📧 Contact

For questions or collaboration opportunities, please reach out through the About page in the application.

---

**Note**: This application provides computational predictions for research purposes only. Always validate with experimental data and consult toxicology experts for critical applications.
