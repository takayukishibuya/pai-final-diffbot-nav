"""PPO で差動二輪ロボットのナビゲーション方策を学習する

例:  python train.py --steps 1500000 --envs 8
出力: models/ppo_diffbot.zip (最終), models/best/best_model.zip (評価成功率が最良),
      logs/ (学習曲線用の Monitor CSV, 評価ログ)
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from diffbot_nav.env import DiffBotNavEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true", help="サブプロセスを使わない (デバッグ用)")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    vec_cls = DummyVecEnv if args.dummy else SubprocVecEnv
    venv = make_vec_env(DiffBotNavEnv, n_envs=args.envs, seed=args.seed,
                        monitor_dir="logs/train_monitor", vec_env_cls=vec_cls,
                        monitor_kwargs=dict(info_keywords=("is_success", "collision")))
    eval_env = Monitor(DiffBotNavEnv())
    eval_cb = EvalCallback(eval_env, n_eval_episodes=40, eval_freq=max(50_000 // args.envs, 1),
                           best_model_save_path="models/best", log_path="logs/eval",
                           deterministic=True, verbose=1)

    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=512, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        seed=args.seed, verbose=1, device="cpu",
    )
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=eval_cb, progress_bar=False)
    model.save("models/ppo_diffbot")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> models/ppo_diffbot.zip")
    venv.close()


if __name__ == "__main__":
    main()
