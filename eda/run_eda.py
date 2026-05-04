from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import MAIN_DATA_PATH, REFERENCE_DATA_PATH
from utils import resolve_output_dir


def run_step(
    script_path: Path,
    main_csv: Path,
    reference_csv: Path,
    output_dir: Path,
    style: str,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--main-csv",
        str(main_csv),
        "--reference-csv",
        str(reference_csv),
        "--output-dir",
        str(output_dir),
        "--style",
        style,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EDA pipeline.")
    parser.add_argument("--main-csv", type=Path, default=MAIN_DATA_PATH, help="Path to blood cell CSV.")
    parser.add_argument("--reference-csv", type=Path, default=REFERENCE_DATA_PATH, help="Path to cell type reference CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for this run. If omitted, a timestamped folder is created.",
    )
    parser.add_argument("--style", type=str, default="whitegrid", help="Seaborn style.")
    parser.add_argument(
        "--include-modeling",
        action="store_true",
        help="Also run 04_modeling.py after the EDA stages.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Modeling holdout test size.")
    parser.add_argument("--random-state", type=int, default=42, help="Modeling random state.")
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=400,
        help="Max holdout rows used for SHAP plotting in 04_modeling.py.",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    script_dir = Path(__file__).resolve().parent
    scripts = [
        script_dir / "01_overview.py",
        script_dir / "02_univariate_figures.py",
        script_dir / "03_bivariate_figures.py",
    ]

    for script in scripts:
        run_step(script, args.main_csv, args.reference_csv, output_dir, args.style)

    if args.include_modeling:
        run_step(
            script_dir / "04_modeling.py",
            args.main_csv,
            args.reference_csv,
            output_dir,
            args.style,
            extra_args=[
                "--test-size",
                str(args.test_size),
                "--random-state",
                str(args.random_state),
                "--shap-sample-size",
                str(args.shap_sample_size),
            ],
        )

    print(f"EDA complete. Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
