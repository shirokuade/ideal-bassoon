"""Load and preprocess the HPC workload dataset from Prometheus CSVs.

Each CSV contains multi-container, multi-pod data at 2-second intervals.
This loader filters to workload containers only, converts units, and
produces per-pod time series ready for RL training.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.oom_parser import PodOutcome, parse_oom_logs

WORKLOAD_CONTAINERS = {"spark-runner", "lammps-bench", "unet-bench"}

# Only load the columns actually needed for RL training.
# This avoids loading unused columns per CSV.
REQUIRED_COLUMNS = [
    # Metadata (for grouping and filtering)
    "timestamp",
    "experiment",
    "pod",
    "container",
    # Memory metrics (observation space)
    "memory_working_set",
    "memory_rss",
    "memory_usage",
    "memory_cache",
    "memory_limit_kube",
    "memory_request_kube",
    "memory_failures_rate",
    # CPU metrics
    "cpu_limit_kube",
    "cpu_usage_rate",
]

# Memory columns that are in bytes and need conversion to MB
BYTES_TO_MB_COLUMNS = [
    "memory_cache",
    "memory_limit_kube",
    "memory_request_kube",
    "memory_rss",
    "memory_usage",
    "memory_working_set",
]

BYTES_PER_MB = 1_048_576.0

# Minimum number of rows per pod trace to be usable for RL
MIN_TRACE_LENGTH = 10


@dataclass
class PodTrace:
    # 1 pod trace = 1 episode in RL. 

    pod_name: str
    experiment: str
    workload_type: str  # graph, inmem, lammps, mlperf
    data: pd.DataFrame  # Preprocessed time series (sorted by timestamp)
    oom_outcome: PodOutcome | None  # OOM label from job logs (None if not found) -> unused

    @property
    def succeeded(self) -> bool | None:
        return self.oom_outcome.succeeded if self.oom_outcome else None

    def __len__(self) -> int:
        return len(self.data)


def _infer_workload_type(experiment_name: str) -> str:
    name = experiment_name.lower()
    if name.startswith("graph"):
        return "graph"
    elif name.startswith("inmem"):
        return "inmem"
    elif name.startswith("lammps"):
        return "lammps"
    elif name.startswith("mlperf"):
        return "mlperf"
    return "unknown"

# Match a Kubernetes pod name to its OOM outcome.
# Unused
def _match_pod_to_oom(
    pod_name: str, oom_outcomes: dict[str, PodOutcome]
) -> PodOutcome | None:
    
    if pod_name in oom_outcomes:
        return oom_outcomes[pod_name]

    parts = pod_name.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 5:
        base_name = parts[0]
        if base_name in oom_outcomes:
            return oom_outcomes[base_name]

    for oom_name, outcome in oom_outcomes.items():
        if pod_name.startswith(oom_name):
            return outcome

    return None


def load_csv(filepath: str) -> pd.DataFrame:
    # Auto-detect delimiter (semicolon or comma)
    with open(filepath) as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","

    # Only load columns that exist in this file
    all_cols = pd.read_csv(filepath, sep=sep, nrows=0).columns.tolist()
    cols_to_load = [c for c in REQUIRED_COLUMNS if c in all_cols]

    df = pd.read_csv(filepath, sep=sep, usecols=cols_to_load, low_memory=False)

    # Filter to workload container rows only
    if "container" in df.columns:
        df = df[df["container"].isin(WORKLOAD_CONTAINERS)].copy()

    if df.empty:
        return df

    # Convert bytes to MB
    for col in BYTES_TO_MB_COLUMNS:
        if col in df.columns:
            df[col] = df[col] / BYTES_PER_MB

    # Parse timestamp (handles mixed formats)
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(str)
        if ts.iloc[0].count(".") == 1 and "/" in ts.iloc[0]:
            ts = ts.str.replace(r"(\d+)\.(\d+)$", r"\1:\2", regex=True)
        df["timestamp"] = pd.to_datetime(ts, dayfirst=True, format="mixed")
        df = df.sort_values("timestamp")

    return df


def filter_cliff_traces(
    traces: list["PodTrace"],
    spike_threshold: float = 0.05,
    cold_pct: float = 0.10,
    min_spike: float = 0.0,
) -> list["PodTrace"]:
    result = []
    for trace in traces:
        df = trace.data
        if "memory_working_set" not in df.columns:
            continue

        ws = df["memory_working_set"].dropna().values
        if len(ws) < 3:
            continue

        if "memory_limit_kube" in df.columns:
            valid_lim = df["memory_limit_kube"].dropna()
            orig = float(valid_lim.iloc[0]) if not valid_lim.empty else 0.0
        else:
            orig = 0.0
        if orig <= 0:
            peak = float(np.nanmax(ws))
            orig = peak * 2.0 if peak > 0 else 1.0

        found = False
        for i in range(1, len(ws)):
            step_jump = ws[i] - ws[i - 1]
            if ws[i - 1] / orig < cold_pct and step_jump / orig > spike_threshold:
                if step_jump >= min_spike:
                    found = True
                    break

        if found:
            result.append(trace)

    return result


def load_dataset(
    dataset_dir: str = "./dataset",
    min_trace_length: int = MIN_TRACE_LENGTH,
    skip_warmup: bool = True,
    warmup_rows: int = 0,
) -> list[PodTrace]:
    
    oom_outcomes = parse_oom_logs(dataset_dir)

    csv_files = sorted(
        f
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No *.csv files found in {dataset_dir}"
        )

    all_traces: list[PodTrace] = []

    for csv_file in csv_files:
        filepath = os.path.join(dataset_dir, csv_file)
        print(f"Loading {csv_file}...")

        df = load_csv(filepath)
        if df.empty:
            print(f"No workload container rows found, skipping.")
            continue

        for (experiment, pod_name), pod_df in df.groupby(
            ["experiment", "pod"]
        ):
            pod_df = pod_df.reset_index(drop=True)

            if skip_warmup:
                if len(pod_df) > warmup_rows * 2:
                    pod_df = pod_df.iloc[warmup_rows:].reset_index(drop=True)
                else:
                    warmup_col = "memory_working_set"
                    if warmup_col in pod_df.columns:
                        valid_mask = pod_df[warmup_col].notna()
                        if valid_mask.any():
                            first_valid = valid_mask.idxmax()
                            pod_df = pod_df.iloc[first_valid:].reset_index(drop=True)

            if len(pod_df) < min_trace_length:
                continue

            workload_type = _infer_workload_type(experiment)
            oom_outcome = _match_pod_to_oom(pod_name, oom_outcomes)

            trace = PodTrace(
                pod_name=pod_name,
                experiment=experiment,
                workload_type=workload_type,
                data=pod_df,
                oom_outcome=oom_outcome,
            )
            all_traces.append(trace)

    print(
        f"\nLoaded {len(all_traces)} pod traces "
        f"({sum(1 for t in all_traces if t.succeeded is True)} succeeded, "
        f"{sum(1 for t in all_traces if t.succeeded is False)} failed, "
        f"{sum(1 for t in all_traces if t.succeeded is None)} unknown)"
    )
    return all_traces
