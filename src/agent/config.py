
from dataclasses import dataclass


@dataclass
class Config:

    # Environment constants (scaled for real HPC data: up to 56 Gi limits)
    MIN_MEMORY_LIMIT: float = 64.0  # MB
    MAX_ADJUSTMENT_PCT: float = 0.10  # Max adjustment = 15% of current agent limit per step
    MAX_STEPS_PER_EPISODE: int = 0  # Determined by pod trace length
    TARGET_MARGIN_PCT: float = 0.15  # 15% default margin
    INITIAL_LIMIT_MULTIPLIER: float = 1.0  # Start from real K8s limit
    PROGRESS_BAR = False  # Whether to show progress bars during training/evaluation
    
    # PPO hyperparameters T1
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 1024
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5

    # Training settings
    TOTAL_TIMESTEPS: int = 1_000_000
    N_ENVS: int = 4
    EVAL_FREQ: int = 25_000
    N_EVAL_EPISODES: int = 5
    CHECKPOINT_FREQ: int = 0
    SEED: int = 42

    # Network architecture (128,128 for 12D observation)
    NET_ARCH_PI: tuple = (32, 32)
    NET_ARCH_VF: tuple = (256, 256)

    # Frame stacking: 0 = disabled, N > 0 = stack last N observations (obs becomes N*base_dimD)
    FRAME_STACK_N: int = 0

    # action[1] surge multiplier.
    # failure_scale = 1 + |action_1| × failures_norm × SURGE_BOOST_PCT
    # e.g. SURGE_BOOST_PCT=2.0 → max scale = 3.0 when failures_norm=1 and |action_1|=1
    SURGE_BOOST_PCT: float = 2.0

    # Init-phase detection
    COLD_USAGE_PCT: float = 0.15   # usage below this fraction of original_limit = cold/init
    IDLE_NORM: float = 20.0        # idle_steps_norm saturates at 1.0 after this many cold steps

    # Failure rate above this threshold discourages trimming
    FAILURE_RATE_THRESHOLD: float = 0.5

    # Dataset
    DATASET_DIR: str = "./dataset"
    EVAL_DATASET_DIR: str = "./dataset/evaluation"

    # Paths
    MODEL_SAVE_DIR: str = "./models/real"
    BEST_MODEL_DIR: str = "./models/real/best"
    CHECKPOINT_DIR: str = "./models/real/checkpoints"
    LOG_DIR: str = "./logs/ppo_memory_real"
    EVAL_LOG_DIR: str = "./logs/eval_real"
