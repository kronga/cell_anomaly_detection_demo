from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from config import ANOMALY_GROUP_COL, CLASS_PALETTE
from utils import (
    add_anomaly_group,
    base_parser,
    categorical_columns,
    chunked,
    load_data,
    merge_reference,
    numeric_columns,
    resolve_output_dir,
    save_figure,
    setup_plot_style,
)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _drop_unused_axes(fig: plt.Figure, axes, used: int) -> None:
    for ax in axes.flat[used:]:
        fig.delaxes(ax)


def main() -> None:
    parser = base_parser("Generate univariate EDA figures.")
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)
    fig_dir = out_dir / "figures" / "univariate"
    setup_plot_style(args.style)

    df, ref_df = load_data(args.main_csv, args.reference_csv)
    df = add_anomaly_group(df)
    merged = merge_reference(df, ref_df)

    num_cols = [c for c in numeric_columns(df) if c != "anomaly_label"]
    cat_cols = categorical_columns(df)

    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=ANOMALY_GROUP_COL,
        hue=ANOMALY_GROUP_COL,
        order=["Normal", "Anomaly"],
        hue_order=["Normal", "Anomaly"],
        palette=CLASS_PALETTE,
        legend=False,
    )
    plt.title("Anomaly Label Distribution")
    save_figure(fig_dir / "target_anomaly_label_count.png")

    # Aggregated overlay histograms (Normal vs Anomaly)
    for page_idx, cols in enumerate(chunked(num_cols, 9), start=1):
        fig, axes = plt.subplots(3, 3, figsize=(19, 14))
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            sns.histplot(
                data=df,
                x=col,
                hue=ANOMALY_GROUP_COL,
                hue_order=["Normal", "Anomaly"],
                palette=CLASS_PALETTE,
                bins=35,
                stat="density",
                common_norm=False,
                multiple="layer",
                alpha=0.35,
                kde=True,
                ax=ax,
            )
            ax.set_title(col)
        _drop_unused_axes(fig, axes, len(cols))
        fig.suptitle("Overlay Histograms: Normal vs Anomaly", fontsize=16, y=1.01)
        save_figure(fig_dir / "aggregated" / f"overlay_histograms_page_{page_idx:02d}.png")

    # Aggregated distribution comparison as boxplots
    for page_idx, cols in enumerate(chunked(num_cols, 9), start=1):
        fig, axes = plt.subplots(3, 3, figsize=(19, 14))
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            sns.boxplot(
                data=df,
                x=ANOMALY_GROUP_COL,
                y=col,
                hue=ANOMALY_GROUP_COL,
                order=["Normal", "Anomaly"],
                hue_order=["Normal", "Anomaly"],
                palette=CLASS_PALETTE,
                legend=False,
                ax=ax,
            )
            ax.set_title(col)
            ax.set_xlabel("")
        _drop_unused_axes(fig, axes, len(cols))
        fig.suptitle("Boxplot Comparison: Normal vs Anomaly", fontsize=16, y=1.01)
        save_figure(fig_dir / "aggregated" / f"boxplot_comparison_page_{page_idx:02d}.png")

    # Categorical counts
    for page_idx, cols in enumerate(chunked(cat_cols, 4), start=1):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            order = df[col].value_counts().index.tolist()[:15]
            sns.countplot(
                data=df,
                x=col,
                hue=ANOMALY_GROUP_COL,
                order=order,
                hue_order=["Normal", "Anomaly"],
                palette=CLASS_PALETTE,
                ax=ax,
            )
            ax.set_title(f"{col} (top 15)")
            ax.tick_params(axis="x", rotation=45)
        _drop_unused_axes(fig, axes, len(cols))
        fig.suptitle("Categorical Comparison by Anomaly Group", fontsize=16, y=1.01)
        save_figure(fig_dir / "aggregated" / f"categorical_comparison_page_{page_idx:02d}.png")

    # Reference-informed plot if available
    if "clinical_significance" in merged.columns:
        plt.figure(figsize=(12, 5))
        order = merged["clinical_significance"].value_counts().index.tolist()
        sns.countplot(
            data=merged,
            x="clinical_significance",
            order=order,
            hue=ANOMALY_GROUP_COL,
            hue_order=["Normal", "Anomaly"],
            palette=CLASS_PALETTE,
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Clinical Significance Counts (reference)")
        save_figure(fig_dir / "reference_clinical_significance_count.png")

    print(f"Univariate figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
