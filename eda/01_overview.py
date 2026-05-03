from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import (
    base_parser,
    categorical_columns,
    load_data,
    merge_reference,
    numeric_columns,
    resolve_output_dir,
    save_df,
    save_text,
)


def build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null_count": [int(df[c].notna().sum()) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "unique_count": [int(df[c].nunique(dropna=False)) for c in df.columns],
        }
    )
    return schema


def build_categorical_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in cols:
        vc = df[col].value_counts(dropna=False)
        rows.append(
            {
                "column": col,
                "n_unique": int(df[col].nunique(dropna=False)),
                "top_value": str(vc.index[0]) if len(vc) else "",
                "top_count": int(vc.iloc[0]) if len(vc) else 0,
                "top_ratio": float(vc.iloc[0] / len(df)) if len(vc) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = base_parser("Generate EDA overview tables.")
    args = parser.parse_args()

    out_dir = resolve_output_dir(args.output_dir)
    table_dir = out_dir / "tables"

    df, ref_df = load_data(args.main_csv, args.reference_csv)
    merged_df = merge_reference(df, ref_df)

    schema_df = build_schema_table(df)
    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)
    numeric_desc = df[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "column"})
    categorical_summary = build_categorical_summary(df, categorical_cols)

    class_balance = (
        df["anomaly_label"]
        .value_counts(dropna=False)
        .rename_axis("anomaly_label")
        .reset_index(name="count")
        .sort_values("anomaly_label")
    )
    class_balance["ratio"] = class_balance["count"] / len(df)

    save_df(schema_df, table_dir / "schema.csv")
    save_df(numeric_desc, table_dir / "numeric_describe.csv")
    save_df(categorical_summary, table_dir / "categorical_summary.csv")
    save_df(class_balance, table_dir / "class_balance.csv")

    for col in categorical_cols:
        freq_df = (
            df[col]
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="count")
        )
        freq_df["ratio"] = freq_df["count"] / len(df)
        save_df(freq_df, table_dir / "categorical_frequencies" / f"{col}_frequency.csv")

    ref_join_status = merged_df["clinical_significance"].isna().map(
        {True: "missing_reference", False: "matched"}
    )
    ref_join_df = ref_join_status.value_counts().rename_axis("join_status").reset_index(name="count")
    ref_join_df["ratio"] = ref_join_df["count"] / len(merged_df)
    save_df(ref_join_df, table_dir / "reference_join_status.csv")

    quality_lines = [
        f"rows={len(df)}",
        f"columns={len(df.columns)}",
        f"total_missing={int(df.isna().sum().sum())}",
        f"duplicate_rows={int(df.duplicated().sum())}",
        f"numeric_columns={len(numeric_cols)}",
        f"categorical_columns={len(categorical_cols)}",
    ]
    save_text("\n".join(quality_lines) + "\n", table_dir / "data_quality.txt")

    print(f"Overview artifacts saved to: {table_dir}")


if __name__ == "__main__":
    main()
