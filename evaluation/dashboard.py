"""
Phantom Tracker - Evaluation Dashboard
======================================
Streamlit web UI for inspecting MOT evaluation results.

Run:
    streamlit run evaluation/dashboard.py -- --results-dir results/

The dashboard reads `*.json` files produced by `evaluation/runner.py` from the
results directory and surfaces them across four sections:

  1. Overview & MOT metrics primer (what each metric means, why it matters)
  2. Single-run inspection (pick one run, see headline + detailed table)
  3. Multi-run comparison (bar charts across runs, side-by-side table)
  4. Industry benchmark comparison (our runs vs published BoT-SORT, ByteTrack,
     DeepSORT, SORT scores)

Each section is self-contained; you can reload any section without re-running
others. The dashboard is read-only - it never modifies results files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make our package imports work when streamlit launches us as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmarks import (  # noqa: E402  (sys.path mutation must precede)
    MOT17_BENCHMARKS, MOT20_BENCHMARKS,
    DEFAULT_BENCHMARK_NAME,
)


# ── helpers ────────────────────────────────────────────────────────────

def _resolve_results_dir() -> Path:
    """Read --results-dir from CLI args (passed after `--` to streamlit)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--results-dir", type=str, default="results")
    args, _ = p.parse_known_args()
    return Path(args.results_dir).expanduser().resolve()


@st.cache_data
def _load_runs(results_dir: str) -> dict[str, dict]:
    """Load every *.json metric file under the results directory."""
    out: dict[str, dict] = {}
    rd = Path(results_dir)
    if not rd.is_dir():
        return out
    for p in sorted(rd.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out[p.stem] = data
        except Exception:
            continue
    return out


def _runs_to_df(runs: dict[str, dict]) -> pd.DataFrame:
    """Flatten runs into a wide DataFrame for tabular display + plotting."""
    rows = []
    for name, m in runs.items():
        meta = m.get("_meta", {})
        config = meta.get("config", {})
        rows.append({
            "Run": name,
            "Sequence": meta.get("sequence", "?"),
            "Frames": meta.get("frames_processed", m.get("Frames", 0)),
            "FPS": meta.get("wall_clock_fps", float("nan")),
            "MOTA": m.get("MOTA", float("nan")) * 100,
            "MOTP": m.get("MOTP", float("nan")) * 100,
            "IDF1": m.get("IDF1", float("nan")) * 100,
            "Precision": m.get("Precision", float("nan")) * 100,
            "Recall": m.get("Recall", float("nan")) * 100,
            "MT": m.get("MT", 0),
            "ML": m.get("ML", 0),
            "IDSwitches": m.get("IDSwitches", 0),
            "Fragmentations": m.get("Fragmentations", 0),
            "FN": m.get("FN", 0),
            "FP": m.get("FP", 0),
            "TP": m.get("TP", 0),
            "GTObjects": m.get("GTObjects", 0),
            **{f"cfg.{k}": v for k, v in config.items()},
        })
    return pd.DataFrame(rows)


# ── page sections ──────────────────────────────────────────────────────

def section_overview():
    st.title("Phantom Tracker - Evaluation Dashboard")
    st.markdown(
        "Inspect MOT-Challenge evaluation results from `evaluation/runner.py`. "
        "Each entry below is a single sequence run scored against ground truth."
    )

    with st.expander("MOT metrics primer (click to expand)"):
        st.markdown("""
**MOTA** (Multi-Object Tracking Accuracy): Combined error rate.
`1 - (FN + FP + IDSw) / GT`. Range `(-inf, 1]`. Single most-cited summary
statistic; a tracker with MOTA = 0.80 means the tracker accumulated errors
(misses, false alarms, identity switches) equal to 20% of ground-truth
detections.

**IDF1** (ID-aware F1): Harmonic mean of ID Precision and ID Recall.
Measures how well the tracker preserves identities over time. Higher than
MOTA when the tracker's bounding boxes are good but it occasionally
swaps IDs; lower when identities flip frequently.

**MOTP** (MO Tracking Precision): Average IoU between matched
predictions and ground truth. Pure localization quality.

**MT / ML** (Mostly Tracked / Mostly Lost): Number of ground-truth tracks
correctly tracked for >=80% of their lifespan (MT) and tracked for <=20% (ML).

**IDSwitches**: Number of times a tracker swapped a track's identity.
Single-most-painful metric for surveillance use cases.

**Fragmentations**: Number of times a tracker lost and re-found a target
(track gaps).

**FN / FP / TP**: False Negatives (missed detections), False Positives
(spurious detections), True Positives (correct detections).
""")


def section_single_run(runs: dict[str, dict]):
    st.header("1. Single-run inspection")
    if not runs:
        st.info("No runs found. Run `evaluation/runner.py` and point this dashboard at the results directory.")
        return
    name = st.selectbox("Pick a run", list(runs.keys()), key="single_run_pick")
    m = runs[name]
    meta = m.get("_meta", {})

    # Headline metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MOTA", f"{m.get('MOTA', 0) * 100:.1f}")
    col2.metric("IDF1", f"{m.get('IDF1', 0) * 100:.1f}")
    col3.metric("ID Switches", m.get("IDSwitches", 0))
    col4.metric("Wall-clock FPS", f"{meta.get('wall_clock_fps', float('nan')):.1f}")

    # Detailed counts
    st.subheader("Counts")
    counts = pd.DataFrame([
        {"Metric": "True positives (TP)", "Value": m.get("TP", 0)},
        {"Metric": "False positives (FP)", "Value": m.get("FP", 0)},
        {"Metric": "False negatives (FN)", "Value": m.get("FN", 0)},
        {"Metric": "Mostly tracked GT (MT)", "Value": m.get("MT", 0)},
        {"Metric": "Mostly lost GT (ML)", "Value": m.get("ML", 0)},
        {"Metric": "Fragmentations", "Value": m.get("Fragmentations", 0)},
        {"Metric": "Total GT detections", "Value": m.get("GTObjects", 0)},
        {"Metric": "Total predictions", "Value": m.get("Predictions", 0)},
    ])
    st.dataframe(counts, hide_index=True, use_container_width=True)

    # Config used
    config = meta.get("config", {})
    if config:
        st.subheader("Configuration used")
        st.json(config, expanded=False)


def section_comparison(runs: dict[str, dict]):
    st.header("2. Multi-run comparison")
    if len(runs) < 2:
        st.info(f"Need at least 2 runs to compare. Have {len(runs)}.")
        return
    df = _runs_to_df(runs)
    selected = st.multiselect("Runs to include", df["Run"].tolist(), default=df["Run"].tolist())
    if not selected:
        st.warning("Select at least one run.")
        return
    sub = df[df["Run"].isin(selected)]

    # Bar chart: MOTA + IDF1 side-by-side per run
    long = sub.melt(
        id_vars=["Run"],
        value_vars=["MOTA", "IDF1", "Precision", "Recall"],
        var_name="Metric", value_name="Score (%)",
    )
    fig = px.bar(long, x="Run", y="Score (%)", color="Metric", barmode="group",
                 title="Headline metrics per run")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Detailed metrics table")
    display_cols = ["Run", "Sequence", "Frames", "FPS", "MOTA", "IDF1",
                    "MT", "ML", "IDSwitches", "Fragmentations", "FN", "FP"]
    st.dataframe(sub[display_cols], hide_index=True, use_container_width=True)


def section_benchmark(runs: dict[str, dict]):
    st.header("3. Comparison vs published benchmarks")

    benchmark_set = st.radio("Benchmark set", ["MOT17 (test)", "MOT20 (test)"], horizontal=True)
    benchmark = MOT17_BENCHMARKS if benchmark_set.startswith("MOT17") else MOT20_BENCHMARKS

    if not runs:
        st.info("No local runs to compare yet. The published baselines below show the target we're aiming for.")
        bench_df = pd.DataFrame([
            {"Tracker": k, **v} for k, v in benchmark.items()
        ])
        st.dataframe(bench_df, hide_index=True, use_container_width=True)
        return

    df = _runs_to_df(runs)
    pick = st.selectbox("Pick one of your runs to compare against published baselines",
                        df["Run"].tolist())
    our = df[df["Run"] == pick].iloc[0]
    our_row = {"Tracker": f"Phantom Tracker [{pick}]",
               "MOTA": float(our["MOTA"]),
               "IDF1": float(our["IDF1"])}
    rows = [our_row] + [
        {"Tracker": k, "MOTA": v.get("MOTA"), "IDF1": v.get("IDF1")}
        for k, v in benchmark.items()
    ]
    bench_df = pd.DataFrame(rows)

    # Bar chart: MOTA + IDF1 of our run vs baselines
    long = bench_df.melt(id_vars=["Tracker"], value_vars=["MOTA", "IDF1"],
                          var_name="Metric", value_name="Score (%)")
    fig = px.bar(long, x="Tracker", y="Score (%)", color="Metric", barmode="group",
                 title=f"Phantom Tracker vs published baselines on {benchmark_set}",
                 color_discrete_map={"MOTA": "#1f77b4", "IDF1": "#ff7f0e"})
    fig.update_layout(yaxis_range=[0, 100])
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(bench_df, hide_index=True, use_container_width=True)
    st.caption(
        "Published numbers are official paper results; ours are computed via "
        "motmetrics on the predictions emitted by `evaluation/runner.py`. "
        "Direct comparison is fair only when the same detector and dataset "
        "split are used."
    )


def section_ablation(runs: dict[str, dict]):
    st.header("4. Ablation: metric vs config hyperparameter")
    if len(runs) < 3:
        st.info(
            "Ablation plots need 3+ runs that vary a single hyperparameter "
            "(e.g. multiple track_buffer values). Run more configurations "
            "to populate this view."
        )
        return
    df = _runs_to_df(runs)
    cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
    cfg_cols_with_variation = [c for c in cfg_cols if df[c].nunique() > 1]
    if not cfg_cols_with_variation:
        st.info("All loaded runs use identical configurations - nothing to ablate.")
        return
    pick_x = st.selectbox("Hyperparameter to vary (X axis)", cfg_cols_with_variation)
    pick_y = st.selectbox("Metric (Y axis)", ["MOTA", "IDF1", "MOTP", "IDSwitches"], index=0)

    fig = px.line(df.sort_values(pick_x), x=pick_x, y=pick_y, markers=True,
                  hover_data=["Run", "Sequence"])
    fig.update_traces(line_width=3, marker_size=10)
    st.plotly_chart(fig, use_container_width=True)


# ── main entry point ───────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Phantom Tracker - Evaluation",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    results_dir = _resolve_results_dir()
    runs = _load_runs(str(results_dir))

    with st.sidebar:
        st.markdown(f"**Results directory**\n\n`{results_dir}`")
        st.markdown(f"**Runs loaded:** {len(runs)}")
        if st.button("Reload"):
            _load_runs.clear()
            st.rerun()
        st.markdown("---")
        st.markdown(f"Default benchmark: **{DEFAULT_BENCHMARK_NAME}**")

    section_overview()
    st.markdown("---")
    section_single_run(runs)
    st.markdown("---")
    section_comparison(runs)
    st.markdown("---")
    section_benchmark(runs)
    st.markdown("---")
    section_ablation(runs)


if __name__ == "__main__":
    main()
