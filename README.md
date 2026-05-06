# cell_anomaly_detection_demo

This repository provides a complete workflow for blood-cell data analysis:
1. Scripted EDA (tables + figures),
2. Disease-category model training and comparison,
3. SHAP-based feature importance,
4. Interactive localhost web app for exploration and prediction.

## 1. Repository purpose

Use this project to:
- Explore the dataset structure and class behavior.
- Train and compare classification models for `disease_category`.
- Inspect model performance and feature importance.
- Interactively explore EDA and 3D PCA in a web interface.

## 2. Data files

- `cell_anomaly_dataset/blood_cell_anomaly_detection.csv` (main dataset)
- `cell_anomaly_dataset/cell_type_reference.csv` (reference metadata)

## 3. Project structure

- `eda/01_overview.py` - schema, quality checks, summary tables
- `eda/02_univariate_figures.py` - univariate distributions and category plots
- `eda/03_bivariate_figures.py` - correlations, feature-vs-target plots, PCA figures
- `eda/04_modeling.py` - train/test modeling, model comparison, SHAP, `summary.md`
- `eda/run_eda.py` - orchestrator for full pipeline
- `webapp/app.py` - Streamlit web app (EDA + PCA + model results + prediction UI)

## 4. Installation

From repository root:

```bash
python -m pip install streamlit plotly
```

If your environment is missing ML/EDA libraries, install the standard stack used by the scripts (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `shap`).

## 5. Run the analysis pipeline

Run EDA only:

```bash
python eda/run_eda.py
```

Run EDA + modeling:

```bash
python eda/run_eda.py --include-modeling
```

Run with explicit paths/output:

```bash
python eda/run_eda.py \
  --main-csv cell_anomaly_dataset/blood_cell_anomaly_detection.csv \
  --reference-csv cell_anomaly_dataset/cell_type_reference.csv \
  --output-dir outputs/eda/manual_run \
  --include-modeling
```

## 6. Run the localhost web app

Start the app:

```bash
streamlit run webapp/app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

App tabs:
- **EDA Explorer**: interactive distributions, scatter plots, pivot tables.
- **PCA 3D Explorer**: choose X/Y/Z principal components, choose color label, zoom/rotate/pan.
- **Modeling Results**: loads saved run artifacts from `outputs/eda/*`.
- **Predict**: enter feature values and get disease-category probabilities.

If Predict is empty at first, click **Train models now** in the sidebar.

## 7. Output artifacts

Each run writes to `outputs/eda/<timestamp>/` (or your custom `--output-dir`), including:
- `tables/` and `figures/` from EDA
- `tables/modeling/` and `figures/modeling/` from model evaluation + SHAP
- `summary.md` with main modeling figures and findings

## 8. Notes

- Current modeling target is `disease_category` (multiclass).
- The modeling stage excludes leakage-prone/non-generalizable columns before training.
- Class color convention in comparison plots:
  - **Normal**: `#1f77b4`
  - **Anomaly**: `#FA8072`
