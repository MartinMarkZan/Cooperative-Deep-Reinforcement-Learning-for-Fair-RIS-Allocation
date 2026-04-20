from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
import supersuit as ss

from src import Config
from src.envs import RISAuctionEnv
from src.utils import RewardLogger


def linear_schedule(initial_value: float):
    """Linear learning-rate schedule: progress 1 to 0."""
    return lambda progress_remaining: progress_remaining * initial_value


def make_vec_env(cfg: Config, render_mode: str | None = None):
    """Create a vectorized PettingZoo env."""
    env = RISAuctionEnv(cfg, render_mode)
    # Supersuit convert to vector environment.
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Supersuit concatenate multiple environments (CPUs > 1 don't work in Win).
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=cfg.num_vec_envs, num_cpus=1, base_class="stable_baselines3")
    return env, vec_env


if __name__ == "__main__":
    start = time.time()

    cfg = Config()
    save_folder = Path("results") / f"{cfg.model_name}_{cfg.f_name}"
    log_dir = save_folder / "log" # Store logs for Tensorboard.
    best_model_path = save_folder / "best_model.zip"

    # Create train and eval envs (same config, separate instances).
    train_env, train_vec_env = make_vec_env(cfg)
    eval_env, eval_vec_env = make_vec_env(cfg)

    # Callbacks.
    reward_logger = RewardLogger(eval_env)
    early_stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=40, verbose=1)
    eval_callback = EvalCallback(eval_vec_env, 
        callback_after_eval=CallbackList([reward_logger, early_stop_callback]), 
        n_eval_episodes=20, eval_freq=cfg.n_steps, log_path=log_dir, 
        best_model_save_path=save_folder, deterministic=True, warn=False)

    # Load or create model.
    if best_model_path.exists():
        print("Loading existing model...")
        model = PPO.load(best_model_path, env=train_vec_env)
    else:
        print("Creating new model...")
        # Proximal policy optimization model.
        model = PPO(
                "MlpPolicy",
                train_vec_env,
                learning_rate=linear_schedule(0.0003),
                n_steps=cfg.n_steps,
                gamma=1.0,
                # TODO keep or remove: ent_coef=0.001,
                tensorboard_log=log_dir,
                verbose=1,
                device="auto",
            )

    model.learn(total_timesteps=cfg.timesteps, callback=eval_callback, reset_num_timesteps=False)

    end = time.time()
    print(f"End of execution. Elapsed time: {(end - start) / 60:.2f} minutes.", )
