#!/usr/bin/env python3
"""Entry point for training the PPO memory management agent on real HPC data."""

import argparse
import multiprocessing
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.config import Config
from src.agent.train import train

CPU_COUNT = multiprocessing.cpu_count()


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for memory management"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--workload", type=str, default="all",
        help="Workload type: graph/inmem/lammps/mlperf or 'all'"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-envs", type=str, default="auto",
        help=(
            "Parallel environments.  'auto' = all CPUs − 1  "
            f"(currently {CPU_COUNT - 1} on this machine), "
            "or pass an explicit integer (default: auto)"
        ),
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 3e-4)"
    )
    args = parser.parse_args()

    # Resolve n_envs
    if args.n_envs == "auto":
        n_envs = max(1, CPU_COUNT - 1)   # leave 1 core for the main process
    else:
        n_envs = int(args.n_envs)

    config = Config(
        TOTAL_TIMESTEPS=args.timesteps or Config().TOTAL_TIMESTEPS,
        SEED=args.seed,
        N_ENVS=n_envs,
        LEARNING_RATE=args.lr or 3e-4,
    )
    workload_filter = args.workload if args.workload != "all" else None

    print("=== PPO Memory Management Training (Real HPC Data) ===")
    print(f"Timesteps:       {config.TOTAL_TIMESTEPS:,}")
    print(f"Workload filter: {workload_filter or 'all'}")
    print(f"Seed:            {config.SEED}")
    print(f"Parallel envs:   {config.N_ENVS}  (CPUs available: {CPU_COUNT})")
    print(f"Learning rate:   {config.LEARNING_RATE}")
    print(f"Network arch:    pi={config.NET_ARCH_PI}  vf={config.NET_ARCH_VF}")
    print(f"Steps/rollout:   {config.N_STEPS * config.N_ENVS:,}  "
          f"(n_steps={config.N_STEPS} × n_envs={config.N_ENVS})")
    print("=======================================================\n")

    train(config=config, workload_filter=workload_filter)


if __name__ == "__main__":
    main()
