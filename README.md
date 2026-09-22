# Credit Score Classifier — Local Deployment

Individual project: a credit-tier classifier (Poor / Standard / Good) trained inside a
leakage-safe pipeline, then deployed as a local Streamlit app.

## Open this first
- [`notebook.ipynb`](./notebook.ipynb) — full EDA, preprocessing, model comparison, and training.
- [`app.py`](./app.py) — the Streamlit app (loads `pipeline.py` + `artifacts/`).

## Pipeline
- `pipeline.py` / `train.py` — the training pipeline (imputation, encoding, scaling, SMOTE-Tomek, all fit inside the training fold only).
- `evaluation.py` — held-out evaluation.
- `data_ingestion.py` — data loading utilities.
- `artifacts/` — the trained pipeline (`credit_score_pipeline.pkl`).

## Result
Best model (LightGBM): **Macro-F1 0.722**, ROC-AUC 0.885.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

No dataset is committed here — see the notebook for the source.
