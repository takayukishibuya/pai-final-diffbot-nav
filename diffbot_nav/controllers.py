"""比較用の制御器

- RuleBasedController : 人が設計したポテンシャル法 (ゴールへの引力 + 障害物からの斥力)
- PPOController       : 強化学習 (PPO) で学習したニューラルネットワーク方策による推論
"""
from __future__ import annotations

import math

import numpy as np

from .env import N_RAYS, RAY_MAX, W_MAX


class RuleBasedController:
    """ポテンシャル法による反射的な障害物回避 (AI を使わないベースライン)"""

    name = "ルールベース (ポテンシャル法)"

    def __init__(self, influence=0.7, gain=1.5):
        self.influence = influence
        self.gain = gain
        self.angles = np.linspace(-math.pi, math.pi, N_RAYS, endpoint=False)

    def reset(self):
        pass

    def __call__(self, obs, env=None):
        ranges = obs[:N_RAYS] * RAY_MAX
        sb, cb = obs[N_RAYS + 1], obs[N_RAYS + 2]
        fx, fy = cb, sb                                    # ゴール方向への引力 (単位ベクトル)
        for a, r in zip(self.angles, ranges):
            if r < self.influence:                         # 近い障害物から斥力
                k = (self.influence - r) / self.influence
                fx -= self.gain * k * math.cos(a)
                fy -= self.gain * k * math.sin(a)
        heading = math.atan2(fy, fx)
        w = float(np.clip(2.5 * heading / W_MAX, -1, 1))
        front = min(r for a, r in zip(self.angles, ranges) if abs(a) <= math.pi / 4)
        v = float(np.clip(math.cos(heading), 0, 1) * np.clip((front - 0.12) / 0.5, 0, 1))
        return np.array([v, w], dtype=np.float32)


class PPOController:
    """学習済み PPO 方策 (ニューラルネット) で行動を推論する"""

    name = "AI (強化学習 PPO)"

    def __init__(self, model_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")

    def reset(self):
        pass

    def __call__(self, obs, env=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
