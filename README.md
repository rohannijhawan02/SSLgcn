# SSL-GCN Molecular Toxicity Prediction

A graph-learning pipeline for binary toxicity prediction from molecular SMILES. The repository contains a DGL/PyTorch GCN, classical fingerprint baselines, preprocessing utilities, notebooks, saved evaluation tables, and an optional FastAPI/React interface.

## Problem statement

Toxicity screening is expensive and slow. This project evaluates whether molecular structure can be used to predict activity on the 12 Tox21 challenge assays covering nuclear-receptor and stress-response pathways.

## Objectives

- Convert SMILES strings into molecular graphs with 74-dimensional atom features.
- Train one binary GCN classifier per toxicity endpoint.
- Compare graph predictions with KNN, neural network, random forest, SVM, and XGBoost baselines.
- Expose trained predictions and result tables through an optional web interface.

## Proposed solution

The implemented system is a supervised, per-endpoint GCN pipeline. It reads the bundled assay CSVs, creates DGL graph objects, performs train/validation/test splitting, trains a GCN with class-weighted loss, selects the best validation ROC-AUC checkpoint, and writes test metrics and training history. Fingerprint-based baseline trainers are available separately.

The attached project document describes a broader semi-supervised teacher–student workflow. That workflow is retained as a reference diagram only: the current source does not implement pseudo-labeling, Mean Teacher updates, or additional unlabeled datasets, so this README does not claim those features as shipped functionality.

## Architecture and workflow

```text
Tox21 assay CSVs
        │
        ▼
SMILES validation and graph conversion (RDKit + DGL)
        │
        ▼
Per-endpoint train/validation/test splits
        │
        ├── GCN training → checkpoints/<endpoint>/best_model.pt
        ├── Baseline training → results/baseline_models/
        └── Evaluation → results/<endpoint>/ and results/overall_summary.csv
                                      │
                                      ▼
                         Optional FastAPI + React interface
```

Reference diagrams from `labfinal.docx` are included in [assets/](assets/):

- [Workflow reference](assets/workflow-reference.png)
- [Data-flow reference](assets/data-flow-reference.png)
- [Dataset overview](assets/dataset-overview.png)
- [Prediction-screen reference](assets/prediction-screen-reference.png)

These diagrams are documentation assets, not additional executable components.

## Methodology

1. Read one CSV per assay. Each file contains an endpoint label, molecule identifier, and SMILES string.
2. Validate and convert molecules to undirected DGL graphs using RDKit-derived atom and bond information.
3. Split each endpoint into train, validation, and test sets during preprocessing.
4. Train the GCN and monitor ROC-AUC, which is more informative than accuracy for imbalanced assays.
5. Save the best checkpoint and CSV metrics for each endpoint.
6. Train optional ECFP/fingerprint baselines and compare their saved result tables.

This implementation is not a multi-task model and does not perform semi-supervised learning yet.

## Dataset

The bundled `Data/csv/` files are the 12 endpoint tables used by this project, with 7,832 rows per assay including the header. The endpoints are:

`NR-AhR`, `NR-AR`, `NR-AR-LBD`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, and `SR-p53`.

Source links:

- [Official Tox21 Challenge data page](https://tripod.nih.gov/tox21/challenge/data.jsp)
- [Official Tox21 data and tools portal](https://tox21.gov/data-and-tools/)
- [MoleculeNet/DeepChem Tox21 CSV mirror](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz)

The repository does not claim that its per-assay CSVs are byte-for-byte identical to any one mirror. Check the source release and assay definitions before using the data for a new study.

## Technologies

- Python, pandas, NumPy, scikit-learn
- RDKit for chemical parsing and molecular features
- PyTorch and DGL for graph learning
- XGBoost, matplotlib, and seaborn for baselines and analysis
- FastAPI/Uvicorn backend and React/Vite frontend (optional)

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── assets/                 # Reference diagrams and project image
├── Data/csv/               # Bundled assay CSVs
├── src/                    # Preprocessing, models, training, prediction, analysis
├── notebooks/              # Exploratory and training notebooks
├── checkpoints/            # Existing GCN checkpoints (<1 MB each)
├── results/                # Existing metrics, histories, predictions, and comparisons
└── webapp/
    ├── backend/            # FastAPI service and model loaders
    └── frontend/           # React/Vite client; install dependencies locally
```

Generated graph caches, baseline model pickles, Python environments, `node_modules`, and frontend build output are intentionally not committed. Running preprocessing or training recreates generated outputs.

## Installation and execution

Use Python 3.10–3.12 with a working RDKit and DGL installation. On some platforms RDKit/DGL are easier to install with Conda than pip.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Preprocess all bundled assays:

```bash
python src/data_preprocessing.py
```

Train one endpoint (the default output paths are relative to the repository root):

```bash
python src/train_all_toxicities.py --dataset NR-AhR --epochs 100
```

Train all endpoints:

```bash
python src/train_all_toxicities.py --epochs 100
```

Run a saved-checkpoint prediction:

```bash
python src/predict.py --smiles "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
```

The baseline trainers are in `src/train_baseline_models.py` and `src/train_all_baseline_models.py`. They create the ignored `models/baseline_models/` directory and write comparison outputs below `results/baseline_models/`.

### Optional web application

Install backend dependencies from the root requirements file, then run the API from its directory:

```bash
cd webapp/backend
uvicorn app:app --reload --port 8000
```

In another terminal:

```bash
cd webapp/frontend
npm install
npm run dev
```

The API expects RDKit, DGL, PyTorch, and available checkpoints. Without those dependencies or model artifacts, only the parts that do not require model inference can run.

## Results

The checked-in result tables are historical outputs produced by the repository. They are not regenerated during cleanup. The GCN aggregate file is [results/overall_summary.csv](results/overall_summary.csv); per-endpoint histories and test metrics are under `results/<endpoint>/`, and baseline comparisons are under `results/baseline_models/`.

The saved GCN test ROC-AUC values in the aggregate file range from 0.6744 to 0.8459 across the 12 endpoints. These are reported repository outputs, not a claim of a newly reproduced experiment; reproduce them after installing the compatible scientific stack and rerunning the pipeline.

## Limitations

- The source implements supervised per-endpoint GCN training, not the SSL/Mean Teacher design shown in the reference document.
- No unlabeled ClinTox, SIDER, ToxCast, or HIV training pipeline is included.
- Assay imbalance produces weak precision/F1 for some endpoints even when ROC-AUC is reasonable.
- Checkpoints and cached graphs are generated artifacts, not a portable model release.
- The web application has optional integrations and is not a substitute for experimental toxicology.

## Future improvements

- Implement and evaluate the documented teacher–student semi-supervised objective.
- Add reproducible configuration files, fixed seeds, and automated tests for graph conversion and data splits.
- Publish versioned model artifacts and provenance for each result table.
- Add calibration, uncertainty estimates, applicability-domain checks, and external validation.
- Make the frontend/backend contract and deployment configuration platform-independent.

## References

- [Tox21 Challenge](https://tripod.nih.gov/tox21/challenge/)
- [Tox21 data and tools](https://tox21.gov/data-and-tools/)
- [MoleculeNet: A Benchmark for Molecular Machine Learning](https://deepchem.readthedocs.io/en/latest/moleculenet.html)
- [DGL documentation](https://www.dgl.ai/pages/about.html)
- [RDKit documentation](https://www.rdkit.org/docs/)
