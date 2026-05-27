# Since we don't use labeled data for RL training, we drop the OOM label parsing.
# This code is ignored/unused.

import csv
import os
import re
from dataclasses import dataclass


@dataclass
class PodOutcome:
    pod_name: str
    workload_type: str  # graph, inmem, lammps, mlperf
    problem_size: str  
    memory_size: str  
    succeeded: bool  # True if SuccessCriteriaMet, False if FailureTarget


# CSV files (new format): {workload}_oom_log.csv
_CSV_FILES = {
    "graph_oom_log.csv":  "graph",
    "inmem_oom_log.csv":  "inmem",
    "lammps_oom_log.csv": "lammps",
    "mlperf_oom_log.csv": "mlperf",
}

# Legacy .log files (fallback if CSV not present)
_LOG_FILES = {
    "GRAPH-oom.log":  "graph",
    "INMEM-oom.log":  "inmem",
    "LAMMPS-oom.log": "lammps",
    "MLPERF-oom.log": "mlperf",
}

# All known memory/resource tier names, ordered smallest to largest.
MEMORY_SIZES = [
    "verysmall",     # 1Gi
    "small",         # 2Gi
    "mediumsmall",   # 4Gi
    "mediumsmall2",  # 6Gi
    "medium",        # 8Gi
    "medium2",       # 10Gi
    "mediumlarge",   # 12Gi
    "mediumlarge2",  # 14Gi
    "large",         # 16Gi
    "large2",        # 18Gi
    "large3",        # 20Gi
    "large4",        # 22Gi
    "xlarge",        # 24Gi
    "xlarge2",       # 26Gi
    "xlarge3",       # 30Gi
    "2xlarge",       # 34Gi
    "2xlarge2",      # 38Gi
    "3xlarge",       # 44Gi
    "huge",          # 50Gi
    "extreme",       # 56Gi
    "massive",       # additional tier
    "xlarge-plus",   # inmem variant
    "2xlarge-v2",
    "3xlarge-v2",
    "4xlarge",
    "4xlarge-v2",
    "5xlarge",
    "5xlarge-v2",
    "huge-v2",
    "extreme-v2",
]

# Pre-sorted by length descending for greedy matching
_MEMORY_SIZES_SORTED = sorted(MEMORY_SIZES, key=len, reverse=True)

# Legacy log patterns
_ARROW_PATTERN = re.compile(
    r"→\s+(\S+):\s+(SuccessCriteriaMet|FailureTarget)"
)
_JOB_PATTERN = re.compile(
    r"Job\s+(\S+)\s+ended with status:\s+(SuccessCriteriaMet|FailureTarget)"
)


def _parse_pod_name(pod_name: str, workload_type: str) -> tuple[str, str]:
    suffix = pod_name[len(workload_type) + 1:]  # strip "workload-" prefix

    for mem_size in _MEMORY_SIZES_SORTED:
        if suffix == mem_size:
            return "", mem_size
        if suffix.endswith("-" + mem_size):
            problem_size = suffix[:-(len(mem_size) + 1)]
            return problem_size, mem_size

    # Fallback: split on last hyphen
    parts = suffix.rsplit("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return suffix, "unknown"


def _parse_oom_csv(filepath: str, workload_type: str) -> list[PodOutcome]:
    """Parse a CSV OOM file: columns job-name, status."""
    outcomes = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_name = row["job-name"].strip()
            status = row["status"].strip()
            if not job_name or not status:
                continue
            succeeded = status == "SuccessCriteriaMet"
            problem_size, memory_size = _parse_pod_name(job_name, workload_type)
            outcomes.append(PodOutcome(
                pod_name=job_name,
                workload_type=workload_type,
                problem_size=problem_size,
                memory_size=memory_size,
                succeeded=succeeded,
            ))
    return outcomes


def _parse_combined_oom_csv(filepath: str) -> list[PodOutcome]:
    outcomes = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_name = row.get("job_name", "").strip()
            status = row.get("status", "").strip()
            workload_type = row.get("workload", "").strip().lower()
            if not job_name or not status:
                continue
            succeeded = status == "SuccessCriteriaMet"
            problem_size, memory_size = _parse_pod_name(job_name, workload_type)
            outcomes.append(PodOutcome(
                pod_name=job_name,
                workload_type=workload_type,
                problem_size=problem_size,
                memory_size=memory_size,
                succeeded=succeeded,
            ))
    return outcomes


def _infer_workload_from_filename(filename: str) -> str:
    """Guess workload type from a filename that doesn't follow naming conventions."""
    lower = filename.lower()
    for wt in ("graph", "inmem", "lammps", "mlperf"):
        if wt in lower:
            return wt
    return "unknown"


def _detect_csv_format(filepath: str) -> str:
    """Return 'combined', 'per_workload', or 'unknown' based on CSV headers."""
    with open(filepath, newline="") as f:
        headers = next(csv.reader(f), [])
    if "job_name" in headers and "workload" in headers:
        return "combined"
    if "job-name" in headers:
        return "per_workload"
    return "unknown"


def _parse_oom_log(filepath: str, workload_type: str) -> list[PodOutcome]:
    """Parse a legacy text OOM log file using regex patterns."""
    outcomes = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            match = _ARROW_PATTERN.search(line) or _JOB_PATTERN.search(line)
            if not match:
                continue
            pod_name = match.group(1)
            succeeded = match.group(2) == "SuccessCriteriaMet"
            problem_size, memory_size = _parse_pod_name(pod_name, workload_type)
            outcomes.append(PodOutcome(
                pod_name=pod_name,
                workload_type=workload_type,
                problem_size=problem_size,
                memory_size=memory_size,
                succeeded=succeeded,
            ))
    return outcomes


def parse_oom_logs(dataset_dir: str = "./dataset") -> dict[str, PodOutcome]:
    all_outcomes: dict[str, PodOutcome] = {}
    handled_files: set[str] = set()

    # --- Step 1: per-workload named files ---
    for workload_type in ("graph", "inmem", "lammps", "mlperf"):
        csv_name = f"{workload_type}_oom_log.csv"
        log_name = f"{workload_type.upper()}-oom.log"
        csv_path = os.path.join(dataset_dir, csv_name)
        log_path = os.path.join(dataset_dir, log_name)

        if os.path.exists(csv_path):
            outcomes = _parse_oom_csv(csv_path, workload_type)
            handled_files.add(csv_name)
            source = csv_name
        elif os.path.exists(log_path):
            outcomes = _parse_oom_log(log_path, workload_type)
            handled_files.add(log_name)
            source = log_name
        else:
            continue

        for outcome in outcomes:
            all_outcomes[outcome.pod_name] = outcome
        print(f"  OOM labels [{workload_type}]: {len(outcomes)} entries from {source}")

    # --- Step 2: auto-discover any remaining *oom* files ---
    try:
        dir_files = sorted(os.listdir(dataset_dir))
    except FileNotFoundError:
        return all_outcomes

    for fname in dir_files:
        if "oom" not in fname.lower() or fname in handled_files:
            continue
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.isfile(fpath):
            continue

        outcomes: list[PodOutcome] = []
        fmt = ""

        if fname.endswith(".csv"):
            detected = _detect_csv_format(fpath)
            if detected == "combined":
                outcomes = _parse_combined_oom_csv(fpath)
                fmt = "combined CSV"
            elif detected == "per_workload":
                wt = _infer_workload_from_filename(fname)
                outcomes = _parse_oom_csv(fpath, wt)
                fmt = f"per-workload CSV ({wt})"
            else:
                print(f"  Warning: unrecognized CSV format in {fname}, skipping")
                continue
        elif fname.endswith(".log"):
            wt = _infer_workload_from_filename(fname)
            outcomes = _parse_oom_log(fpath, wt)
            fmt = f"log ({wt})"
        else:
            continue

        new_count = sum(1 for o in outcomes if o.pod_name not in all_outcomes)
        for outcome in outcomes:
            if outcome.pod_name not in all_outcomes:
                all_outcomes[outcome.pod_name] = outcome
        print(f"  OOM labels [auto]: {len(outcomes)} entries from {fname} "
              f"({new_count} new, format: {fmt})")

    return all_outcomes


def validate_dataset_oom(dataset_dir: str) -> dict:
    """Check every pod name in a dataset directory against its OOM log.

    For each workload, loads the OOM log and all data CSVs, then reports
    which pod names could be matched and which could not.

    Args:
        dataset_dir: Path to the dataset directory (training or evaluation).

    Returns:
        Dict with per-workload match statistics and unmatched pod samples.
    """
    import csv as _csv

    oom_outcomes = parse_oom_logs(dataset_dir)

    # Collect all data CSVs (exclude OOM log CSVs)
    try:
        csv_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.endswith(".csv") and "oom" not in f.lower()
        )
    except FileNotFoundError:
        print(f"Directory not found: {dataset_dir}")
        return {}

    if not csv_files:
        print(f"No data CSV files found in {dataset_dir}")
        return {}

    # Group pod names by workload
    workload_pods: dict[str, set[str]] = {}
    for csv_file in csv_files:
        filepath = os.path.join(dataset_dir, csv_file)
        with open(filepath, newline="") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        with open(filepath, newline="") as f:
            reader = _csv.DictReader(f, delimiter=sep)
            for row in reader:
                pod = row.get("pod", "").strip()
                exp = row.get("experiment", "").strip().lower()
                if not pod:
                    continue
                # Infer workload from experiment name
                for wt in ("graph", "inmem", "lammps", "mlperf"):
                    if exp.startswith(wt):
                        workload_pods.setdefault(wt, set()).add(pod)
                        break

    results = {}
    print(f"\n=== OOM Label Validation: {dataset_dir} ===\n")

    for wt in sorted(workload_pods.keys()):
        pods = workload_pods[wt]
        matched, unmatched = [], []

        for pod in sorted(pods):
            # Mirror _match_pod_to_oom logic from loader.py
            if pod in oom_outcomes:
                matched.append(pod)
                continue
            parts = pod.rsplit("-", 1)
            if len(parts) == 2 and len(parts[1]) == 5:
                base = parts[0]
                if base in oom_outcomes:
                    matched.append(pod)
                    continue
            if any(pod.startswith(name) for name in oom_outcomes):
                matched.append(pod)
                continue
            unmatched.append(pod)

        total = len(pods)
        match_pct = len(matched) / total * 100 if total else 0
        status = "✅" if not unmatched else "❌"

        print(f"  {status} {wt.upper()}: {len(matched)}/{total} matched ({match_pct:.0f}%)")
        if unmatched:
            print(f"     Unmatched samples ({len(unmatched)} pods):")
            for p in unmatched[:5]:
                print(f"       - {p}")
            if len(unmatched) > 5:
                print(f"       ... and {len(unmatched) - 5} more")

        # OOM log summary for this workload
        wt_outcomes = [v for v in oom_outcomes.values() if v.workload_type == wt]
        succeeded = sum(1 for o in wt_outcomes if o.succeeded)
        failed = sum(1 for o in wt_outcomes if not o.succeeded)
        print(f"     OOM log: {len(wt_outcomes)} entries "
              f"({succeeded} succeeded, {failed} failed)")

        results[wt] = {
            "total_pods": total,
            "matched": len(matched),
            "unmatched": len(unmatched),
            "match_pct": match_pct,
            "unmatched_samples": unmatched[:10],
            "oom_log_entries": len(wt_outcomes),
            "oom_log_succeeded": succeeded,
            "oom_log_failed": failed,
        }

    print("\n" + "=" * 50)
    return results


def get_oom_summary(outcomes: dict[str, PodOutcome]) -> dict:
    """Generate a summary of OOM outcomes by workload type."""
    summary: dict[str, dict] = {}

    for outcome in outcomes.values():
        wt = outcome.workload_type
        if wt not in summary:
            summary[wt] = {"total": 0, "succeeded": 0, "failed": 0}
        summary[wt]["total"] += 1
        if outcome.succeeded:
            summary[wt]["succeeded"] += 1
        else:
            summary[wt]["failed"] += 1

    for wt, s in summary.items():
        s["failure_rate"] = s["failed"] / s["total"] if s["total"] > 0 else 0.0

    return summary
