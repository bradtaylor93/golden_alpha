"""Streamlit dashboard for deep run inspection and quant decision support."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from trading_research.analysis.dashboard_data import (
    build_quant_agent_handoff,
    load_run_snapshot,
    metric_cards,
)


def _render_header(snapshot) -> None:
    st.title("Trading Research Quant Dashboard")
    st.caption(
        "Run-level, fold-level, and asset-level diagnostics for systematic workflow review."
    )

    cards = metric_cards(snapshot)
    cols = st.columns(len(cards))
    for col, (name, value) in zip(cols, cards.items(), strict=False):
        col.metric(name, value)


def _render_flow_view(snapshot) -> None:
    st.subheader("Workflow flow and artifact lineage")
    st.markdown("**Execution graph nodes**")
    st.dataframe(snapshot.graph_nodes, use_container_width=True)
    st.markdown("**Artifact index**")
    st.dataframe(snapshot.artifact_index, use_container_width=True, height=320)


def _render_fold_views(snapshot) -> None:
    st.subheader("Fold-level performance")
    if snapshot.fold_metrics.empty:
        st.info("No fold metrics found.")
        return
    st.dataframe(snapshot.fold_metrics, use_container_width=True)

    st.markdown("**Fold trend over time**")
    st.line_chart(
        snapshot.fold_metrics.set_index("outer_fold_id")[["test_mse", "train_mse"]],
        use_container_width=True,
    )


def _render_asset_views(snapshot) -> None:
    st.subheader("Asset-level diagnostics")
    if snapshot.asset_metrics.empty:
        st.info("No asset-level diagnostics available.")
        return
    st.dataframe(snapshot.asset_metrics, use_container_width=True)

    metric_col = st.selectbox(
        "Asset metric",
        [c for c in snapshot.asset_metrics.columns if c != "asset"],
        index=0,
    )
    chart_data = snapshot.asset_metrics[["asset", metric_col]].set_index("asset")
    st.bar_chart(chart_data, use_container_width=True)


def _render_time_views(snapshot) -> None:
    st.subheader("Over-time behavior")
    if snapshot.predictions.empty:
        st.info("No prediction table found.")
        return
    st.markdown("**Prediction and realized target over time**")
    ts_df = (
        snapshot.predictions.groupby("timestamp", as_index=False)[["prediction", "target"]]
        .mean()
        .set_index("timestamp")
    )
    st.line_chart(ts_df, use_container_width=True)

    st.markdown("**Rolling volatility (std) of prediction errors**")
    err = (snapshot.predictions["target"] - snapshot.predictions["prediction"]).rolling(20).std()
    err_df = snapshot.predictions[["timestamp"]].copy()
    err_df["rolling_error_std_20"] = err
    err_df = err_df.dropna().groupby("timestamp", as_index=False).mean().set_index("timestamp")
    if not err_df.empty:
        st.line_chart(err_df, use_container_width=True)


def _render_handoff(snapshot) -> None:
    st.subheader("Quant agent handoff")
    handoff = build_quant_agent_handoff(snapshot)
    st.code(json.dumps(handoff, indent=2), language="json")
    st.download_button(
        "Download handoff JSON",
        data=json.dumps(handoff, indent=2),
        file_name=f"{snapshot.run_id}_quant_handoff.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="Quant Dashboard", layout="wide")

    runs_root_default = str((Path.cwd() / "runs").resolve())
    runs_root = st.sidebar.text_input("Runs root", value=runs_root_default)
    discovered_runs = []
    root_path = Path(runs_root)
    if root_path.exists():
        discovered_runs = sorted({p.parent.name for p in root_path.glob("**/manifest.json")})
    run_id = st.sidebar.selectbox("Run ID", options=[""] + discovered_runs) if discovered_runs else st.sidebar.text_input("Run ID")

    if not run_id:
        st.info("Enter a run ID in the sidebar to load dashboard data.")
        return

    try:
        snapshot = load_run_snapshot(run_id=run_id, runs_root=Path(runs_root))
    except Exception as exc:  # noqa: BLE001 - dashboard should present errors.
        st.error(f"Failed to load run: {exc}")
        return

    _render_header(snapshot)
    tab_flow, tab_fold, tab_assets, tab_time, tab_handoff = st.tabs(
        [
            "Flow and lineage",
            "Fold analysis",
            "Asset analysis",
            "Over-time analysis",
            "Quant handoff",
        ]
    )

    with tab_flow:
        _render_flow_view(snapshot)
    with tab_fold:
        _render_fold_views(snapshot)
    with tab_assets:
        _render_asset_views(snapshot)
    with tab_time:
        _render_time_views(snapshot)
    with tab_handoff:
        _render_handoff(snapshot)


if __name__ == "__main__":
    main()
