"""Evaluation script for the trained PPO memory management agent.
Evaluates on real HPC workload data with per-workload metrics.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.agent.config import Config
from src.data.loader import load_dataset
from src.env.real_data_env import RealDataMemoryEnv
from src.env.wrappers import FrameStackWrapper


def evaluate(
    model_path: str,
    config: Config | None = None,
    workload_filter: str | None = None,
    n_episodes: int | None = None,
    save_plots: bool = True,
    plot_dir: str = "./plots/real",
) -> dict:
    """Evaluate a trained model on real HPC workload data.

    Args:
        model_path: Path to the saved PPO model.
        config: Training/evaluation configuration.
        n_episodes: Max episodes. None = all available traces.
        save_plots: Whether to save episode plots.
        plot_dir: Directory for plots.

    Returns:
        Dictionary with overall and per-workload metrics.
    """
    config = config or Config()
    model = PPO.load(model_path)

    all_traces = load_dataset(config.EVAL_DATASET_DIR)
    if workload_filter:
        all_traces = [t for t in all_traces if t.workload_type == workload_filter]
    if n_episodes is not None:
        all_traces = all_traces[:n_episodes]

    workload_metrics: dict[str, dict] = defaultdict(lambda: {
        "rewards": [], "waste_ratios": [], "oom_counts": [],
        "util_ratios": [], "waste_reductions": [],
        "oom_episodes": 0, "baseline_oom_episodes": 0, "baseline_oom_rates": [],
    })

    plot_count = 0

    for i, trace in enumerate(all_traces):
        env = RealDataMemoryEnv(config=config, pod_traces=[trace])
        if config.FRAME_STACK_N > 0:
            env = FrameStackWrapper(env, n_frames=config.FRAME_STACK_N)
        obs, info = env.reset(seed=config.SEED + i)

        episode_rewards = []
        episode_waste = []
        episode_oom = 0
        episode_util = []
        total_original_waste_safe = 0.0
        total_agent_waste_safe = 0.0
        total_steps = 0
        usages = [info["memory_usage"]]
        agent_limits = [info["memory_upper_limit"]]
        original_limit = info.get("original_limit", info["memory_upper_limit"])
        original_limits = [original_limit]
        failure_rates = [info.get("memory_failures_rate", 0.0)]

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            episode_rewards.append(reward)
            episode_waste.append(info["waste"] / original_limit if original_limit > 0 else 0.0)
            is_oom = info["oom"]
            episode_oom += int(is_oom)
            if info["memory_upper_limit"] > 0:
                episode_util.append(info["memory_usage"] / info["memory_upper_limit"])

            if not is_oom:
                total_original_waste_safe += info.get("original_waste", 0.0)
                total_agent_waste_safe += info["waste"]

            usages.append(info["memory_usage"])
            agent_limits.append(info["memory_upper_limit"])
            original_limits.append(original_limit)
            failure_rates.append(info.get("memory_failures_rate", 0.0))

        waste_reduction = 0.0
        if total_original_waste_safe > 0:
            waste_reduction = (total_original_waste_safe - total_agent_waste_safe) / total_original_waste_safe
        
        # succeeded=False -> OOM/failure
        # baseline_oom_rate = (
        #     1.0 if trace.succeeded is False
        #     else (0.0 if trace.succeeded is True else float("nan"))
        # )

        wt = trace.workload_type
        m = workload_metrics[wt]
        m["rewards"].append(np.sum(episode_rewards))
        m["waste_ratios"].append(np.mean(episode_waste))
        m["oom_counts"].append(episode_oom)
        m["util_ratios"].append(np.mean(episode_util) if episode_util else 0.0)
        m["waste_reductions"].append(waste_reduction)
        if episode_oom > 0:
            m["oom_episodes"] += 1
        if trace.succeeded is False:
            m["baseline_oom_episodes"] += 1
        # m["baseline_oom_rates"].append(baseline_oom_rate)

        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            _plot_real_episode(
                usages, agent_limits, original_limits,
                trace.pod_name, trace.workload_type,
                trace.succeeded, plot_count, plot_dir,
                failure_rates=failure_rates,
            )
            plot_count += 1

    results = {"per_workload": {}, "overall": {}}
    all_r, all_w, all_o, all_u, all_wr, all_bl = [], [], [], [], [], []
    total_oom_episodes = 0
    total_baseline_oom_episodes = 0

    print("\n=== Real Data Evaluation Results ===")
    for wt in sorted(workload_metrics.keys()):
        m = workload_metrics[wt]
        n = len(m["rewards"])
        known_baselines = [v for v in m["baseline_oom_rates"] if not np.isnan(v)]
        wt_result = {
            "n_episodes": n,
            "avg_reward": float(np.mean(m["rewards"])),
            "avg_waste_ratio": float(np.mean(m["waste_ratios"])),
            "avg_oom_per_episode": float(np.mean(m["oom_counts"])),
            "oom_episodes": m["oom_episodes"],
            "baseline_oom_episodes": m["baseline_oom_episodes"],
            "baseline_oom_rate": float(np.mean(known_baselines)) if known_baselines else float("nan"),
            "avg_utilization": float(np.mean(m["util_ratios"])),
            "avg_waste_reduction": float(np.mean(m["waste_reductions"])),
        }
        results["per_workload"][wt] = wt_result
        print(f"\n--- {wt.upper()} ({n} episodes) ---")
        print(f"  Avg Reward:              {wt_result['avg_reward']:.2f}")
        print(f"  Avg Waste Ratio:         {wt_result['avg_waste_ratio']:.4f}")
        print(f"  Avg OOM/Episode:         {wt_result['avg_oom_per_episode']:.2f}")
        print(f"  OOM Episodes (agent):    {wt_result['oom_episodes']} / {n}")
        print(f"  OOM Episodes (baseline): {wt_result['baseline_oom_episodes']} / {n}")
        bl = wt_result['baseline_oom_rate']
        print(f"  Baseline OOM Rate:       {bl:.1%}  (production ground-truth)" if not np.isnan(bl) else "  Baseline OOM Rate:       N/A")
        print(f"  Avg Utilization:         {wt_result['avg_utilization']:.4f}")
        print(f"  Avg Waste Reduction:     {wt_result['avg_waste_reduction']:.1%}")
        all_r.extend(m["rewards"])
        all_w.extend(m["waste_ratios"])
        all_o.extend(m["oom_counts"])
        all_u.extend(m["util_ratios"])
        all_wr.extend(m["waste_reductions"])
        all_bl.extend([v for v in m["baseline_oom_rates"] if not np.isnan(v)])
        total_oom_episodes += m["oom_episodes"]
        total_baseline_oom_episodes += m["baseline_oom_episodes"]

    results["overall"] = {
        "n_episodes": len(all_r),
        "avg_reward": float(np.mean(all_r)) if all_r else 0.0,
        "avg_waste_ratio": float(np.mean(all_w)) if all_w else 0.0,
        "avg_oom_per_episode": float(np.mean(all_o)) if all_o else 0.0,
        "oom_episodes": total_oom_episodes,
        "baseline_oom_episodes": total_baseline_oom_episodes,
        "avg_utilization": float(np.mean(all_u)) if all_u else 0.0,
        "avg_waste_reduction": float(np.mean(all_wr)) if all_wr else 0.0,
        "baseline_oom_rate": float(np.mean(all_bl)) if all_bl else 0.0,
    }
    ov = results["overall"]
    print(f"\n--- OVERALL ({ov['n_episodes']} episodes) ---")
    print(f"  Avg Reward:                  {ov['avg_reward']:.2f}")
    print(f"  Avg Waste Ratio:             {ov['avg_waste_ratio']:.4f}")
    print(f"  Avg OOM/Episode:             {ov['avg_oom_per_episode']:.2f}")
    print(f"  OOM Episodes from agent:     {ov['oom_episodes']} / {ov['n_episodes']}")
    print(f"  OOM Episodes from data trace:{ov['baseline_oom_episodes']} / {ov['n_episodes']}")
    print(f"  Avg Utilization:             {ov['avg_utilization']:.4f}")
    print(f"  Avg Waste Reduction:         {ov['avg_waste_reduction']:.1%}")
    print(f"  Avg Baseline OOM Rate:       {ov['baseline_oom_rate']:.1%}")

    print("====================================\n")
    return results

def _plot_real_episode(
    usages: list[float], agent_limits: list[float], original_limits: list[float],
    pod_name: str, workload_type: str, succeeded: bool | None,
    plot_idx: int, plot_dir: str,
    failure_rates: list[float] | None = None,
) -> None:
    """Plot real data episode: usage vs. agent limit vs. original K8s limit."""
    steps = np.arange(len(usages))
    time_seconds = steps * 2  # 2-second intervals

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_seconds, usages, label="Memory Usage (Working Set)", color="steelblue", linewidth=1.5)
    ax.plot(time_seconds, agent_limits, label="Agent Recommended Limit", color="tomato", linewidth=1.5, linestyle="--")
    ax.plot(time_seconds, original_limits, label="Original K8s Limit", color="gray", linewidth=1.0, linestyle=":")
    ax.fill_between(time_seconds, usages, agent_limits, alpha=0.1, color="tomato", label="Agent Waste")

    for i in range(len(usages)):
        if agent_limits[i] < usages[i]:
            ax.axvspan(time_seconds[i] - 1, time_seconds[i] + 1, color="red", alpha=0.3)

    if failure_rates is not None and any(r > 0 for r in failure_rates):
        ax2 = ax.twinx()
        ax2.bar(
            time_seconds, failure_rates,
            width=1.6, color="orange", alpha=0.5, label="Memory Failure Rate",
        )
        ax2.set_ylabel("Failure Rate (events/s)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_ylim(bottom=0)
        
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        all_handles, all_labels = lines + bars, labels + bar_labels
    else:
        all_handles, all_labels = ax.get_legend_handles_labels()

    ax.legend(
        all_handles, all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(all_handles),
        frameon=True,
        fontsize=9,
    )

    status = "OK" if succeeded else ("FAILED (OOM)" if succeeded is False else "Unknown")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"[{workload_type.upper()}] {pod_name} - Job: {status}")
    ax.grid(True, alpha=0.3)

    path = os.path.join(plot_dir, f"real_{plot_idx + 1}_{workload_type}_{pod_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")   # bbox_inches keeps the legend in frame
    plt.close(fig)
    print(f"Plot saved: {path}")
