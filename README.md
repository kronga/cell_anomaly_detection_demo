# cell_anomaly_detection_demo

Script-based EDA workflow for blood cell anomaly data.

## Data

- `cell_anomaly_dataset/blood_cell_anomaly_detection.csv`
- `cell_anomaly_dataset/cell_type_reference.csv`

## EDA scripts

- `eda/01_overview.py` - schema, quality checks, summary tables
- `eda/02_univariate_figures.py` - univariate distributions and category plots
- `eda/03_bivariate_figures.py` - correlations, feature-vs-target plots, and PCA plots
- `eda/04_modeling.py` - disease-category train/test modeling, model comparison, SHAP, and summary report
- `eda/run_eda.py` - runs the full pipeline end-to-end

## Run

From repository root:

```bash
python eda/run_eda.py
```

Optional explicit paths/output directory:

```bash
python eda/run_eda.py \
  --main-csv cell_anomaly_dataset/blood_cell_anomaly_detection.csv \
  --reference-csv cell_anomaly_dataset/cell_type_reference.csv \
  --output-dir outputs/eda/manual_run
```

Run individual stages:

```bash
python eda/01_overview.py
python eda/02_univariate_figures.py
python eda/03_bivariate_figures.py
python eda/04_modeling.py
```

Run EDA + modeling in one command:

```bash
python eda/run_eda.py --include-modeling
```

## Interactive website (localhost)

Install Streamlit if needed:

```bash
python -m pip install streamlit plotly
```

Run the web app:

```bash
streamlit run webapp/app.py
```

If the Predict tab is empty initially, click **Train models now** in the sidebar (or **Train models for Predict tab** inside the tab).

The website includes:
- **EDA Explorer**: interactive distribution, scatter, and pivot analysis.
- **PCA 3D Explorer**: interactive 3D PCA with selectable PC axes, selectable color label, and built-in zoom/rotate/pan.
- **Modeling Results**: view generated run summaries/figures from `outputs/eda/*`.
- **Predict**: enter feature values and get disease-category prediction + probabilities.

## Outputs

By default each run creates `outputs/eda/<timestamp>/` with:

- `tables/` for CSV/TXT summaries
- `figures/univariate/` for univariate plots
- `figures/bivariate/` for bivariate plots
- `figures/univariate/aggregated/` for large multi-subplot comparison pages
- `figures/bivariate/aggregated/` for large multi-subplot Normal vs Anomaly violin comparisons
- `figures/bivariate/pca/` for PCA projections colored by:
  - `anomaly_group` (Normal vs Anomaly)
  - `cell_type`
  - `disease_category`
  - `dataset_source`
  - `patient_age_group`
  - `patient_sex`
- `tables/modeling/` for model metrics, confusion matrices, reports, and SHAP feature importance
- `figures/modeling/` for model comparison, best confusion matrix, and SHAP figures
- `summary.md` with main modeling figures and concise analysis

Class color convention is fixed across comparison plots:
- **Normal**: blue (`#1f77b4`)
- **Anomaly**: salmon pink (`#FA8072`)
