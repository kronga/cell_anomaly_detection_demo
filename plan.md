# EDA Implementation Plan

## Problem and approach
Build a reusable, script-based EDA workflow that profiles the dataset(s), generates publication-ready figures, and saves all executed logic as `.py` files so it can be rerun later without notebook dependency.

Approach:
1. Create a small EDA script package under the repo (input + output paths configurable).
2. Generate tabular summaries and figures into a timestamped output folder.
3. Keep scripts modular (overview, univariate plots, bivariate plots, optional target-focused plots) and runnable independently.

## Planned todos
1. **Define EDA scope and outputs**
   - Use `blood_cell_anomaly_detection.csv` and `cell_type_reference.csv`.
   - Include reference-table joins/validation where relevant for labels and metadata checks.
   - Implement a **comprehensive** EDA figure set in the first pass.
   - Lock output directory structure and naming conventions.

2. **Set up reusable script structure**
   - Add a dedicated folder (for example `eda/`) with:
     - `config.py` (paths, style, constants),
     - `utils.py` (loading, type grouping, save helpers),
     - executable entry scripts for each EDA stage.

3. **Implement dataset overview script**
   - Script: `eda/01_overview.py`
   - Outputs:
     - shape, dtypes, missingness, duplicates, cardinality,
     - descriptive statistics tables,
     - class-balance and category-frequency tables.
   - Save outputs as CSV/TXT artifacts in `outputs/eda/...`.

4. **Implement figure generation scripts**
   - Script: `eda/02_univariate_figures.py`
     - numeric distributions (hist/KDE/box),
     - categorical count plots,
     - target distribution plots,
     - reference-informed category plots using `cell_type_reference.csv` where useful.
   - Script: `eda/03_bivariate_figures.py`
     - correlation heatmap,
     - numeric-vs-target comparisons,
     - PCA 2D projections from scaled numeric features, colored by relevant columns (`anomaly_label`, `cell_type`, `disease_category`, `dataset_source`, `patient_age_group`, `patient_sex`),
     - selected pairplots/scatter plots for key features,
     - additional comprehensive figures (violin/boxen/facet-based comparisons) for key groupings.
   - Save figures as PNG files with deterministic filenames.

5. **Implement orchestration runner**
   - Script: `eda/run_eda.py`
   - Runs all stages in order and prints final artifact paths.

6. **Document execution**
   - Update `README.md` with exact commands to run each script and the full pipeline.

## Notes and considerations
- Prefer `pandas + matplotlib + seaborn` to keep dependencies standard.
- Ensure scripts are idempotent and create output directories automatically.
- Avoid leakage-prone interpretation in modeling comments; keep EDA descriptive.
- Keep all script logic in `.py` files (no one-off transient commands as the final workflow).
- Refinement added: aggregate multi-subplot figures, class-focused Normal vs Anomaly overlays, and improved anomaly PCA visualizations using fixed class colors (blue for Normal, salmon for Anomaly).
