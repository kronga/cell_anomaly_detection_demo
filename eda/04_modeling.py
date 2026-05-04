from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError as exc:
    raise ImportError("Missing dependency: shap. Install with `pip install shap`.") from exc

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize

from utils import base_parser, load_data, resolve_output_dir, save_df, save_text, setup_plot_style

TARGET_COL = "disease_category"
PRIMARY_METRIC = "macro_f1"
EXCLUDED_FEATURES = {
    "cell_id",
    "anomaly_label",
    "cytodiffusion_anomaly_score",
    "cytodiffusion_classification_confidence",
    "labeller_confidence_score",
    "dataset_source",
    "staining_protocol",
    "microscope_model",
    "image_resolution_px",
    "magnification_x",
}


def markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in headers:
            value = row[col]
            if col in float_cols:
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = base_parser("Train and compare disease_category prediction models.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for split/models.")
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=400,
        help="Maximum number of holdout rows used for SHAP plots.",
    )
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)
    table_dir = out_dir / "tables" / "modeling"
    fig_dir = out_dir / "figures" / "modeling"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style(args.style)

    df, _ = load_data(args.main_csv, args.reference_csv)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURES and c != TARGET_COL]
    x = df[feature_cols].copy()
    y = df[TARGET_COL].astype(str).copy()
    classes = sorted(y.unique().tolist())

    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        sparse_threshold=0.0,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model_specs: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=args.random_state,
        ),
    }

    rows: list[dict[str, object]] = []
    trained_pipelines: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}

    for model_name, estimator in model_specs.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipe.fit(x_train, y_train)

        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)
        y_test_bin = label_binarize(y_test, classes=classes)

        metrics_row = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            "macro_roc_auc_ovr": float(
                roc_auc_score(
                    y_test_bin,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            ),
        }
        rows.append(metrics_row)
        trained_pipelines[model_name] = pipe
        predictions[model_name] = y_pred

        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
        report_df = report_df.rename(columns={"index": "label"})
        save_df(report_df, table_dir / f"classification_report_{model_name}.csv")

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes).reset_index().rename(columns={"index": "actual"})
        save_df(cm_df, table_dir / f"confusion_matrix_{model_name}.csv")

    comparison_df = pd.DataFrame(rows).sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    save_df(comparison_df, table_dir / "model_comparison.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=comparison_df, x=PRIMARY_METRIC, y="model", color="#4C72B0")
    plt.title("Model comparison by macro-F1")
    plt.xlabel("Macro-F1")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(fig_dir / "model_macro_f1_comparison.png")
    plt.close()

    best_model_name = str(comparison_df.loc[0, "model"])
    best_pipe = trained_pipelines[best_model_name]
    best_pred = predictions[best_model_name]
    best_cm = confusion_matrix(y_test, best_pred, labels=classes)

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(f"Best model confusion matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_best_model.png")
    plt.close()

    transformer = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["model"]
    x_train_trans = transformer.transform(x_train)
    x_test_trans = transformer.transform(x_test)
    feature_names = transformer.get_feature_names_out().tolist()

    rng = np.random.default_rng(args.random_state)
    bg_n = min(len(x_train_trans), args.shap_sample_size)
    ex_n = min(len(x_test_trans), args.shap_sample_size)
    bg_idx = rng.choice(len(x_train_trans), size=bg_n, replace=False)
    ex_idx = rng.choice(len(x_test_trans), size=ex_n, replace=False)
    x_bg = x_train_trans[bg_idx]
    x_explain = x_test_trans[ex_idx]

    explainer = shap.Explainer(model, x_bg, feature_names=feature_names)
    shap_values = explainer(x_explain)

    values = shap_values.values
    if values.ndim == 3:
        # Multiclass output: average absolute contribution across classes.
        values_2d = np.mean(np.abs(values), axis=2)
    else:
        values_2d = values

    mean_abs_shap = np.mean(np.abs(values_2d), axis=0)
    shap_importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    save_df(shap_importance_df, table_dir / "shap_feature_importance.csv")

    shap_explanation = shap.Explanation(
        values=values_2d,
        data=x_explain,
        feature_names=feature_names,
    )
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_beeswarm.png")
    plt.close()

    top_shap = shap_importance_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_shap, x="mean_abs_shap", y="feature", color="#DD8452")
    plt.title("Top SHAP feature importance (mean |SHAP|)")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_top_features_bar.png")
    plt.close()

    top_features = shap_importance_df.head(5)["feature"].tolist()
    summary_lines = [
        "# Disease-category prediction summary",
        "",
        "## Setup",
        f"- Target: `{TARGET_COL}` (multiclass)",
        f"- Train/test split: stratified {1.0 - args.test_size:.0%}/{args.test_size:.0%} (`random_state={args.random_state}`)",
        f"- Primary model-ranking metric: `{PRIMARY_METRIC}`",
        f"- Excluded columns: {', '.join(sorted(EXCLUDED_FEATURES))}",
        "",
        "## Model ranking",
        markdown_table(
            comparison_df,
            float_cols={"accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr"},
        ),
        "",
        "## Main figures",
        "![Model comparison](figures/modeling/model_macro_f1_comparison.png)",
        "",
        f"![Best model confusion matrix: {best_model_name}](figures/modeling/confusion_matrix_best_model.png)",
        "",
        "![SHAP beeswarm](figures/modeling/shap_summary_beeswarm.png)",
        "",
        "![SHAP top features](figures/modeling/shap_top_features_bar.png)",
        "",
        "## Key findings",
        f"- Best model: **{best_model_name}** (macro-F1={comparison_df.loc[0, 'macro_f1']:.4f})",
        f"- Top SHAP features: {', '.join(top_features)}",
        "",
        "## Artifact locations",
        "- Tables: `tables/modeling/`",
        "- Figures: `figures/modeling/`",
    ]
    save_text("\n".join(summary_lines) + "\n", out_dir / "summary.md")

    print(f"Modeling artifacts saved to: {out_dir}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
