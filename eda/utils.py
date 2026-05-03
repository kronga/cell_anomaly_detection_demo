from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    ANOMALY_GROUP_COL,
    ANOMALY_LABEL_COL,
    ANOMALY_LABEL_MAP,
    DEFAULT_OUTPUT_ROOT,
    MAIN_DATA_PATH,
    MAX_CLASSES_FOR_PLOT_LEGEND,
    REFERENCE_DATA_PATH,
)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--main-csv",
        type=Path,
        default=MAIN_DATA_PATH,
        help="Path to blood_cell_anomaly_detection.csv",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=REFERENCE_DATA_PATH,
        help="Path to cell_type_reference.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for artifacts. If omitted, a timestamped run dir is created.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="whitegrid",
        help="Seaborn style.",
    )
    return parser


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = DEFAULT_OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_data(main_csv: Path, reference_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_df = pd.read_csv(main_csv)
    ref_df = pd.read_csv(reference_csv)
    return main_df, ref_df


def merge_reference(main_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    return main_df.merge(
        ref_df,
        on=["cell_type", "disease_category", "anomaly_label"],
        how="left",
        suffixes=("", "_ref"),
    )


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def save_df(df: pd.DataFrame, out_path: Path, index: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)


def save_text(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def setup_plot_style(style: str) -> None:
    sns.set_theme(style=style)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def safe_hue_order(series: pd.Series) -> list[str] | None:
    values = series.dropna().astype(str)
    if values.nunique() > MAX_CLASSES_FOR_PLOT_LEGEND:
        return None
    return sorted(values.unique().tolist())


def add_anomaly_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[ANOMALY_GROUP_COL] = out[ANOMALY_LABEL_COL].map(ANOMALY_LABEL_MAP).fillna("Unknown")
    return out


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]
