#!/usr/bin/env python3
"""Entry point for evaluating a trained PPO memory management agent on real HPC data."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.config import Config
from src.agent.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO memory management agent"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to saved model (default: ./models/real/ppo_memory_final)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of evaluation episodes (default: all available traces)"
    )
    parser.add_argument(
        "--workload", type=str, default="all",
        help="Workload filter: graph/inmem/lammps/mlperf/all"
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Directory to save plots"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--frame-stack", type=int, default=0,
        help="Frame stack N (0=disabled, 6=stack last 6 obs). Must match how the model was trained."
    )
    args = parser.parse_args()

    config = Config(FRAME_STACK_N=args.frame_stack)
    model_path = args.model or "./models/real/ppo_memory_final"
    best_model_path = "./models/real/best/best_model"
    plot_dir = args.plot_dir or "./plots/real"
    workload_filter = args.workload if args.workload != "all" else None

    print("=== PPO Memory Management Evaluation (Real HPC Data) ===")
    print(f"Model:           {best_model_path}")
    print(f"Workload filter: {workload_filter or 'all'}")
    print("=========================================================\n")

    evaluate(
        model_path=model_path,
        config=config,
        workload_filter=workload_filter,
        n_episodes=args.episodes,
        save_plots=not args.no_plots,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
