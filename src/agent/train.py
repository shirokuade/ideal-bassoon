
import multiprocessing
import os
import platform

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.agent.config import Config
from src.data.loader import load_dataset
from src.env.real_data_env import RealDataMemoryEnv
from src.env.wrappers import FrameStackWrapper


def _make_env_fn(config: Config, pod_traces: list, workload_filter: str | None):
    def _inner():
        env = RealDataMemoryEnv(
            config=config,
            pod_traces=pod_traces,
            workload_filter=workload_filter,
        )
        if config.FRAME_STACK_N > 0:
            env = FrameStackWrapper(env, n_frames=config.FRAME_STACK_N)
        return env
    return _inner


def create_env(config, pod_traces, workload_filter=None, n_envs=1):
    factories = [_make_env_fn(config, pod_traces, workload_filter)
                 for _ in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(factories)

    # Linux VM -> forkserver (safe with PyTorch)
    # start_method = {"Linux": "forkserver", "Darwin": "fork"}.get(
    #     platform.system(), "spawn"
    # )
    # start_method = {"Darwin": "fork"}.get(
    # platform.system(), "spawn"   # Linux gets "spawn" — reliable on all VMs
    # )

    start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    return SubprocVecEnv(factories, start_method=start_method)


def create_model(env, config: Config) -> PPO:
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=list(config.NET_ARCH_PI),
            vf=list(config.NET_ARCH_VF),
        ),
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=config.LOG_DIR,
        verbose=0,
        seed=config.SEED,
    )
    return model


def train(config=None, workload_filter=None) -> PPO:
    config = config or Config()

    # torch.set_num_threads(1)

    for d in [config.BEST_MODEL_DIR, config.CHECKPOINT_DIR,
              config.LOG_DIR, config.EVAL_LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    print("Loading HPC dataset...")
    pod_traces = load_dataset(config.DATASET_DIR)

    if workload_filter:
        filtered = [t for t in pod_traces if t.workload_type == workload_filter]
        print(f"Filtered to {len(filtered)} '{workload_filter}' traces "
              f"(from {len(pod_traces)} total)")
        pod_traces = filtered

    if not pod_traces:
        raise ValueError("No pod traces available for training.")

    train_env = create_env(config, pod_traces, n_envs=config.N_ENVS)
    eval_env  = create_env(config, pod_traces, n_envs=1)

    model = create_model(train_env, config)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=config.EVAL_LOG_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
    )
    # Remove comment if you want to have checkpoints during training, but it can slow down training significantly
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(config.CHECKPOINT_FREQ // config.N_ENVS, 1),
    #     save_path=config.CHECKPOINT_DIR,
    # )

    callbacks = [eval_callback]
    if config.CHECKPOINT_FREQ > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(config.CHECKPOINT_FREQ // config.N_ENVS, 1),
            save_path=config.CHECKPOINT_DIR,
        ))
    
    # Train
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=config.PROGRESS_BAR,
    )

    # model.learn(
    #     total_timesteps=config.TOTAL_TIMESTEPS,
    #     callback=[eval_callback, checkpoint_callback],
    #     progress_bar=config.PROGRESS_BAR,
    # )

    final_path = os.path.join(config.MODEL_SAVE_DIR, "ppo_memory_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    train_env.close()
    eval_env.close()
    return model
    
