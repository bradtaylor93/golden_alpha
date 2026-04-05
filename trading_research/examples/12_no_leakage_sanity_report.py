"""Generate strict no-leakage sanity report for a run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trading_research.analysis.sanity_checks import write_sanity_report


def _default_run_id(runs_root: Path) -> str:
    manifests = sorted(runs_root.glob("**/manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        raise ValueError(f"No manifests found under {runs_root}")
    return manifests[-1].parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description="No-leakage sanity checks for workflow run artifacts.")
    parser.add_argument(
        "--runs-root",
        default="trading_research/examples/_output/11_champion_edge/runs",
        help="Runs root directory.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run id to check. If omitted, uses most recent run in --runs-root.",
    )
    parser.add_argument(
        "--output-csv",
        default="trading_research/examples/_output/11_champion_edge/reports/no_leakage_assertions.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_id = args.run_id or _default_run_id(runs_root)
    output_csv = Path(args.output_csv)

    report = write_sanity_report(run_id=run_id, runs_root=runs_root, output_csv=output_csv)
    failures = report[~report["passed"]].copy() if not report.empty else pd.DataFrame()
    critical_failures = failures[failures["severity"] == "critical"] if not failures.empty else pd.DataFrame()

    print("run_id:", run_id)
    print("report:", output_csv.resolve())
    print("total_checks:", len(report))
    print("failed_checks:", len(failures))
    print("critical_failures:", len(critical_failures))

    if not failures.empty:
        print("\nTop failures:")
        print(failures.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

