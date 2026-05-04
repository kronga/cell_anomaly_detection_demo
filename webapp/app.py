from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT / "cell_anomaly_dataset" / "blood_cell_anomaly_detection.csv"
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


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(MAIN_CSV)


def build_feature_frame(df: pd.DataFrame) -> dict[str, object]:
    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURES and c != TARGET_COL]
    x = df[feature_cols].copy()
    y = df[TARGET_COL].astype(str).copy()
    classes = sorted(y.unique().tolist())

    numeric_cols = x.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    defaults = {}
    for col in feature_cols:
        if col in numeric_cols:
            defaults[col] = float(x[col].median())
        else:
            mode = x[col].mode(dropna=True)
            defaults[col] = str(mode.iloc[0]) if not mode.empty else str(x[col].dropna().astype(str).iloc[0])

    return {
        "x": x,
        "y": y,
        "classes": classes,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "defaults": defaults,
    }


@st.cache_data(show_spinner=False)
def compute_pca_projection(max_components: int = 8) -> dict[str, object]:
    df = load_dataset()
    frame = build_feature_frame(df)
    x: pd.DataFrame = frame["x"]  # type: ignore[assignment]
    numeric_cols: list[str] = frame["numeric_cols"]  # type: ignore[assignment]
    categorical_cols: list[str] = frame["categorical_cols"]  # type: ignore[assignment]

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

    x_transformed = preprocessor.fit_transform(x)
    n_components = int(min(max_components, x_transformed.shape[1], x_transformed.shape[0]))
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(x_transformed)

    pca_cols = [f"PC{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, columns=pca_cols, index=df.index)

    meta_cols = [TARGET_COL] + [c for c in df.columns if df[c].dtype == "object" and c != TARGET_COL]
    if "anomaly_label" in df.columns:
        meta_cols.append("anomaly_label")
    meta_cols = list(dict.fromkeys(meta_cols))
    pca_df = pd.concat([pca_df, df[meta_cols]], axis=1)

    explained_df = pd.DataFrame(
        {
            "component": pca_cols,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    return {"pca_df": pca_df, "explained_df": explained_df, "pca_cols": pca_cols, "meta_cols": meta_cols}


@st.cache_resource(show_spinner=False)
def train_models(test_size: float, random_state: int) -> dict[str, object]:
    df = load_dataset()
    frame = build_feature_frame(df)
    x: pd.DataFrame = frame["x"]  # type: ignore[assignment]
    y: pd.Series = frame["y"]  # type: ignore[assignment]
    classes: list[str] = frame["classes"]  # type: ignore[assignment]
    feature_cols: list[str] = frame["feature_cols"]  # type: ignore[assignment]
    numeric_cols: list[str] = frame["numeric_cols"]  # type: ignore[assignment]
    categorical_cols: list[str] = frame["categorical_cols"]  # type: ignore[assignment]
    defaults: dict[str, object] = frame["defaults"]  # type: ignore[assignment]

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
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model_specs: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }

    metrics_rows: list[dict[str, object]] = []
    pipelines: dict[str, Pipeline] = {}
    y_pred_map: dict[str, pd.Series] = {}

    for model_name, estimator in model_specs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
            }
        )
        pipelines[model_name] = pipe
        y_pred_map[model_name] = pd.Series(y_pred, index=x_test.index)

    comparison_df = pd.DataFrame(metrics_rows).sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    best_model_name = str(comparison_df.loc[0, "model"])

    return {
        "df": df,
        "x": x,
        "y": y,
        "x_test": x_test,
        "y_test": y_test,
        "classes": classes,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "defaults": defaults,
        "comparison_df": comparison_df,
        "pipelines": pipelines,
        "y_pred_map": y_pred_map,
        "best_model_name": best_model_name,
    }


def render_pca_tab() -> None:
    st.subheader("Interactive 3D PCA")
    st.caption("Rotate, pan, and zoom directly in the 3D plot.")

    pca_state = compute_pca_projection(max_components=10)
    pca_df: pd.DataFrame = pca_state["pca_df"]  # type: ignore[assignment]
    explained_df: pd.DataFrame = pca_state["explained_df"]  # type: ignore[assignment]
    pca_cols: list[str] = pca_state["pca_cols"]  # type: ignore[assignment]
    meta_cols: list[str] = pca_state["meta_cols"]  # type: ignore[assignment]

    if len(pca_cols) < 3:
        st.warning("Not enough principal components to render 3D PCA.")
        return

    c1, c2, c3 = st.columns(3)
    x_pc = c1.selectbox("X component", pca_cols, index=0)
    y_pc = c2.selectbox("Y component", pca_cols, index=1)
    z_pc = c3.selectbox("Z component", pca_cols, index=2)

    color_col = st.selectbox("Color by label", meta_cols, index=0)
    sample_n = st.slider(
        "Points to display",
        min_value=500,
        max_value=min(len(pca_df), 5880),
        value=min(len(pca_df), 2500),
        step=100,
    )
    marker_size = st.slider("Marker size", min_value=2, max_value=10, value=4, step=1)

    plot_df = pca_df.sample(sample_n, random_state=42) if len(pca_df) > sample_n else pca_df.copy()
    plot_df[color_col] = plot_df[color_col].astype(str)

    fig = px.scatter_3d(
        plot_df,
        x=x_pc,
        y=y_pc,
        z=z_pc,
        color=color_col,
        opacity=0.75,
        hover_data={TARGET_COL: True},
        title=f"3D PCA colored by {color_col}",
    )
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), legend_title_text=color_col)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explained variance")
    st.dataframe(explained_df, use_container_width=True)
    st.bar_chart(explained_df.set_index("component"))


def render_eda_tab(df: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Target classes", f"{df[TARGET_COL].nunique():,}")

    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Class balance")
    class_counts = df[TARGET_COL].value_counts().rename_axis(TARGET_COL).reset_index(name="count")
    st.bar_chart(class_counts.set_index(TARGET_COL))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    st.subheader("Interactive distributions")
    plot_col = st.selectbox("Pick a column", df.columns.tolist(), index=0)
    if plot_col in numeric_cols:
        bins = st.slider("Histogram bins", min_value=10, max_value=120, value=40, step=5)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[plot_col], bins=bins, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution: {plot_col}")
        st.pyplot(fig)
    else:
        top_n = st.slider("Top categories", min_value=5, max_value=30, value=15, step=1)
        vc = df[plot_col].value_counts().head(top_n)
        st.bar_chart(vc)

    st.subheader("Scatter explorer")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
        y_col = st.selectbox("Y axis", numeric_cols, index=1, key="scatter_y")
        sample_n = st.slider("Sample points", min_value=500, max_value=min(len(df), 5000), value=min(len(df), 2000), step=100)
        plot_df = df.sample(sample_n, random_state=42) if len(df) > sample_n else df
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=TARGET_COL, alpha=0.55, s=20, ax=ax2)
        ax2.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig2)
    else:
        st.info("Not enough numeric columns for scatter plots.")

    if cat_cols:
        st.subheader("Pivot table")
        idx = st.selectbox("Row category", cat_cols, key="pivot_row")
        col = st.selectbox("Column category", cat_cols, key="pivot_col")
        pivot = pd.crosstab(df[idx], df[col])
        st.dataframe(pivot, use_container_width=True)


def render_results_tab(artifacts_root: Path) -> None:
    st.subheader("Saved modeling artifacts")
    run_dirs = sorted(
        [p for p in artifacts_root.glob("*") if p.is_dir() and (p / "summary.md").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        st.info("No run outputs with summary.md found under outputs/eda/.")
        return

    selected = st.selectbox("Select run folder", run_dirs, format_func=lambda p: p.name)
    st.caption(f"Using: `{selected}`")

    summary_md = selected / "summary.md"
    st.markdown(summary_md.read_text(encoding="utf-8"))

    model_table = selected / "tables" / "modeling" / "model_comparison.csv"
    shap_table = selected / "tables" / "modeling" / "shap_feature_importance.csv"
    if model_table.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(model_table), use_container_width=True)
    if shap_table.exists():
        st.subheader("Top SHAP features")
        st.dataframe(pd.read_csv(shap_table).head(20), use_container_width=True)

    for img_path, title in [
        (selected / "figures" / "modeling" / "model_macro_f1_comparison.png", "Macro-F1 model comparison"),
        (selected / "figures" / "modeling" / "confusion_matrix_best_model.png", "Best model confusion matrix"),
        (selected / "figures" / "modeling" / "shap_summary_beeswarm.png", "SHAP beeswarm"),
        (selected / "figures" / "modeling" / "shap_top_features_bar.png", "SHAP top features"),
    ]:
        if img_path.exists():
            st.image(str(img_path), caption=title, use_container_width=True)


def render_predict_tab(train_state: dict[str, object]) -> None:
    st.subheader("Interactive prediction")
    st.caption("Fill in a sample and predict disease category using any trained model.")

    comparison_df: pd.DataFrame = train_state["comparison_df"]  # type: ignore[assignment]
    pipelines: dict[str, Pipeline] = train_state["pipelines"]  # type: ignore[assignment]
    feature_cols: list[str] = train_state["feature_cols"]  # type: ignore[assignment]
    numeric_cols: list[str] = train_state["numeric_cols"]  # type: ignore[assignment]
    defaults: dict[str, object] = train_state["defaults"]  # type: ignore[assignment]

    model_choice = st.selectbox("Model for prediction", comparison_df["model"].tolist(), index=0)
    st.dataframe(comparison_df, use_container_width=True)

    with st.form("predict_form"):
        input_row: dict[str, object] = {}
        for col in feature_cols:
            if col in numeric_cols:
                input_row[col] = st.number_input(col, value=float(defaults[col]), format="%.6f")
            else:
                values = sorted(load_dataset()[col].dropna().astype(str).unique().tolist())
                default_val = str(defaults[col])
                default_idx = values.index(default_val) if default_val in values else 0
                input_row[col] = st.selectbox(col, values, index=default_idx)

        submitted = st.form_submit_button("Predict")

    if submitted:
        model = pipelines[model_choice]
        input_df = pd.DataFrame([input_row], columns=feature_cols)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        class_names = model.classes_
        proba_df = pd.DataFrame({"class": class_names, "probability": proba}).sort_values("probability", ascending=False)
        st.success(f"Predicted disease category: **{pred}**")
        st.dataframe(proba_df, use_container_width=True)

    st.subheader("Test-set confusion matrix (selected model)")
    y_test = train_state["y_test"]
    y_pred = train_state["y_pred_map"][model_choice]
    classes = train_state["classes"]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {model_choice}")
    st.pyplot(fig)

    st.subheader("Classification report (selected model)")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Cell anomaly EDA + modeling", layout="wide")
    st.title("Cell anomaly: interactive EDA and disease prediction")
    st.caption(f"Dataset: `{MAIN_CSV}`")

    model_params = None
    train_now_sidebar = False
    with st.sidebar:
        st.header("Run settings")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
        model_params = (float(test_size), int(random_state))
        train_now_sidebar = st.button("Train models now")
        retrain = st.button("Retrain models")
        if retrain:
            train_models.clear()
            st.session_state["train_state"] = None
            st.session_state["train_params"] = None
            st.success("Model cache cleared.")

    try:
        df = load_dataset()
    except Exception as exc:
        st.error("Failed to load dataset.")
        st.exception(exc)
        return

    if "train_state" not in st.session_state:
        st.session_state["train_state"] = None
    if "train_params" not in st.session_state:
        st.session_state["train_params"] = None

    if train_now_sidebar:
        with st.spinner("Training models..."):
            st.session_state["train_state"] = train_models(
                test_size=model_params[0],
                random_state=model_params[1],
            )
            st.session_state["train_params"] = model_params
        st.success("Models trained.")

    tab_eda, tab_pca, tab_results, tab_predict = st.tabs(
        ["EDA Explorer", "PCA 3D Explorer", "Modeling Results", "Predict"]
    )
    with tab_eda:
        render_eda_tab(df)
    with tab_pca:
        render_pca_tab()
    with tab_results:
        render_results_tab(ROOT / "outputs" / "eda")
    with tab_predict:
        params_changed = st.session_state["train_params"] != model_params
        train_state = st.session_state["train_state"]
        if train_state is None or params_changed:
            st.info("Models are not trained for the current settings yet.")
            if st.button("Train models for Predict tab"):
                with st.spinner("Training models..."):
                    st.session_state["train_state"] = train_models(
                        test_size=model_params[0],
                        random_state=model_params[1],
                    )
                    st.session_state["train_params"] = model_params
                st.rerun()
        else:
            render_predict_tab(train_state)


if __name__ == "__main__":
    main()
