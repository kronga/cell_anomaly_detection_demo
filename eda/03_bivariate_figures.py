from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import ANOMALY_GROUP_COL, CLASS_PALETTE, RELEVANT_PCA_COLOR_COLUMNS
from utils import (
    add_anomaly_group,
    base_parser,
    chunked,
    load_data,
    merge_reference,
    numeric_columns,
    resolve_output_dir,
    safe_hue_order,
    save_df,
    save_figure,
    setup_plot_style,
)


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name))


def _drop_unused_axes(fig: plt.Figure, axes, used: int) -> None:
    for ax in axes.flat[used:]:
        fig.delaxes(ax)


def compute_pca_2d(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[feature_cols].astype(float).to_numpy()
    mu = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), mu, x)
    std = np.nanstd(x, axis=0, ddof=0)
    std[std == 0] = 1.0
    z = (x - mu) / std

    u, s, vt = np.linalg.svd(z, full_matrices=False)
    comp = z @ vt[:2].T

    total_var = np.sum(s**2)
    explained = (s[:2] ** 2) / total_var
    explained_df = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": explained,
        }
    )

    pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
    return pca_df, explained_df


def main() -> None:
    parser = base_parser("Generate bivariate EDA figures including PCA.")
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)
    fig_dir = out_dir / "figures" / "bivariate"
    table_dir = out_dir / "tables"
    setup_plot_style(args.style)

    df, ref_df = load_data(args.main_csv, args.reference_csv)
    df = add_anomaly_group(df)
    merged = merge_reference(df, ref_df)

    numeric_cols = numeric_columns(df)
    model_feature_cols = [c for c in numeric_cols if c != "anomaly_label"]

    # Correlation heatmap
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap (numeric columns)")
    save_figure(fig_dir / "correlation_heatmap.png")
    save_df(corr.reset_index().rename(columns={"index": "feature"}), table_dir / "correlation_matrix.csv")

    # Aggregated numeric vs anomaly comparison (multi-subplot pages)
    for page_idx, cols in enumerate(chunked(model_feature_cols, 9), start=1):
        fig, axes = plt.subplots(3, 3, figsize=(19, 14))
        for i, col in enumerate(cols):
            ax = axes.flat[i]
            sns.violinplot(
                data=df,
                x=ANOMALY_GROUP_COL,
                y=col,
                hue=ANOMALY_GROUP_COL,
                order=["Normal", "Anomaly"],
                hue_order=["Normal", "Anomaly"],
                palette=CLASS_PALETTE,
                inner="quartile",
                cut=0,
                legend=False,
                ax=ax,
            )
            ax.set_title(col)
            ax.set_xlabel("")
        _drop_unused_axes(fig, axes, len(cols))
        fig.suptitle("Violin Comparison: Normal vs Anomaly", fontsize=16, y=1.01)
        save_figure(fig_dir / "aggregated" / f"violin_comparison_page_{page_idx:02d}.png")

    # PCA
    pca_df, explained_df = compute_pca_2d(df, model_feature_cols)
    pca_full = pd.concat([pca_df, merged.reset_index(drop=True)], axis=1)
    save_df(pca_full, table_dir / "pca_projection_2d.csv")
    save_df(explained_df, table_dir / "pca_explained_variance.csv")

    exp1 = explained_df.loc[explained_df["component"] == "PC1", "explained_variance_ratio"].iloc[0]
    exp2 = explained_df.loc[explained_df["component"] == "PC2", "explained_variance_ratio"].iloc[0]
    for col in RELEVANT_PCA_COLOR_COLUMNS:
        if col not in pca_full.columns:
            continue
        hue_order = safe_hue_order(pca_full[col])
        plt.figure(figsize=(9, 7))
        palette = CLASS_PALETTE if col == ANOMALY_GROUP_COL else None
        sns.scatterplot(
            data=pca_full,
            x="PC1",
            y="PC2",
            hue=col,
            hue_order=hue_order,
            palette=palette,
            alpha=0.8,
            s=26,
        )
        plt.title(f"PCA Projection colored by {col}")
        plt.xlabel(f"PC1 ({exp1:.1%} var)")
        plt.ylabel(f"PC2 ({exp2:.1%} var)")
        save_figure(fig_dir / "pca" / f"pca_colored_by_{_sanitize(col)}.png")

    # Dedicated anomaly PCA view (improved readability)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=pca_full,
        x="PC1",
        y="PC2",
        hue=ANOMALY_GROUP_COL,
        hue_order=["Normal", "Anomaly"],
        palette=CLASS_PALETTE,
        alpha=0.65,
        s=30,
        edgecolor="none",
        ax=axes[0],
    )
    axes[0].set_title("PCA Scatter: Normal vs Anomaly")
    axes[0].set_xlabel(f"PC1 ({exp1:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({exp2:.1%} var)")

    sns.kdeplot(
        data=pca_full,
        x="PC1",
        y="PC2",
        hue=ANOMALY_GROUP_COL,
        hue_order=["Normal", "Anomaly"],
        palette=CLASS_PALETTE,
        levels=6,
        linewidths=1.2,
        fill=False,
        ax=axes[1],
    )
    axes[1].set_title("PCA Density Contours: Normal vs Anomaly")
    axes[1].set_xlabel(f"PC1 ({exp1:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({exp2:.1%} var)")
    fig.suptitle("PCA Comparison by Anomaly Label", fontsize=16, y=1.02)
    save_figure(fig_dir / "pca" / "pca_anomaly_comparison.png")

    # Aggregated PCA panel for relevant color columns
    available_color_cols = [c for c in RELEVANT_PCA_COLOR_COLUMNS if c in pca_full.columns][:6]
    if available_color_cols:
        ncols = 3
        nrows = int(np.ceil(len(available_color_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
        axes_arr = np.array(axes).reshape(-1)
        for i, col in enumerate(available_color_cols):
            ax = axes_arr[i]
            hue_order = safe_hue_order(pca_full[col])
            palette = CLASS_PALETTE if col == ANOMALY_GROUP_COL else None
            sns.scatterplot(
                data=pca_full,
                x="PC1",
                y="PC2",
                hue=col,
                hue_order=hue_order,
                palette=palette,
                alpha=0.6,
                s=18,
                linewidth=0,
                ax=ax,
            )
            ax.set_title(f"PCA colored by {col}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        for ax in axes_arr[len(available_color_cols):]:
            fig.delaxes(ax)
        fig.suptitle("PCA multi-view comparison", fontsize=16, y=1.01)
        save_figure(fig_dir / "pca" / "pca_multiview_relevant_columns.png")

    # Pairplot on selected features for readability
    var_series = df[model_feature_cols].var(numeric_only=True).sort_values(ascending=False)
    pair_cols = var_series.head(5).index.tolist()
    pair_df = df[pair_cols + [ANOMALY_GROUP_COL]].copy()
    sample_n = min(1500, len(pair_df))
    pair_df = pair_df.sample(sample_n, random_state=42)
    grid = sns.pairplot(
        pair_df,
        vars=pair_cols,
        hue=ANOMALY_GROUP_COL,
        hue_order=["Normal", "Anomaly"],
        palette=CLASS_PALETTE,
        corner=True,
        diag_kind="hist",
    )
    grid.fig.suptitle("Pairplot: top variance numeric features", y=1.02)
    grid.savefig(fig_dir / "pairplot_top_variance_features.png", bbox_inches="tight")
    plt.close("all")

    print(f"Bivariate figures and PCA artifacts saved to: {fig_dir}")


if __name__ == "__main__":
    main()
