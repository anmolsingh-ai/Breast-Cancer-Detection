# Breast Cancer Preprocessing & Voting Ensemble

This repository preprocesses a Breast Cancer dataset and trains a voting ensemble model.

**Project structure**
- data/: raw and processed datasets
- features/features.py: preprocessing script
- models/: trained model and training script
- dvc.yaml: DVC pipeline stages
- data.dvc: DVC-tracked data directory

**Quick start**
1. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate.bat
```

2. Install required packages:

```bash
pip install pandas scikit-learn joblib dvc
```

3. Run preprocessing (creates `data/processed/processed_breast_cancer.csv`):

```bash
python -m features.features
```

4. Train the voting ensemble (saves `models/voting_model.joblib`):

```bash
python -m models.voting_ensemble
```

5. Or run the full pipeline with DVC:

```bash
dvc repro
```

**DVC tracking**
- The dataset directory is tracked via `data.dvc` (tracks `data/`).
- The trained model can be tracked either via a `.dvc` file (`models/voting_model.joblib.dvc`) or declared as an output in `dvc.yaml`'s `train` stage — this repository currently keeps data tracked via `data.dvc` and the model tracked as configured in `dvc.yaml`/`.dvc` files.

**Outputs**
- Processed data: `data/processed/processed_breast_cancer.csv`
- Trained model: `models/voting_model.joblib`

**Notes**
- If you change which files DVC tracks, avoid overlapping tracked directories (e.g., do not track `data/` and also `data/processed/` separately).
- If you want, I can generate a `requirements.txt` or run the pipeline and show results.
