from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "cell_anomaly_dataset"
MAIN_DATA_PATH = DATA_DIR / "blood_cell_anomaly_detection.csv"
REFERENCE_DATA_PATH = DATA_DIR / "cell_type_reference.csv"

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "eda"
ANOMALY_LABEL_COL = "anomaly_label"
ANOMALY_GROUP_COL = "anomaly_group"
ANOMALY_LABEL_MAP = {0: "Normal", 1: "Anomaly"}
CLASS_PALETTE = {"Normal": "#1f77b4", "Anomaly": "#FA8072"}
RELEVANT_PCA_COLOR_COLUMNS = [
    ANOMALY_GROUP_COL,
    "cell_type",
    "disease_category",
    "dataset_source",
    "patient_age_group",
    "patient_sex",
]

# Keep legends readable for high-cardinality columns.
MAX_CLASSES_FOR_PLOT_LEGEND = 25
