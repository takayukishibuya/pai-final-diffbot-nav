"""動作確認 (学習前に 30 秒ほどで終わる)。 python check_env.py"""
import math
import time

import numpy as np

from diffbot_nav.controllers import RuleBasedController
from diffbot_nav.env import DiffBotNavEnv

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


env = DiffBotNavEnv(n_obstacles=(0, 0))
obs, _ = env.reset(seed=0)
check("観測の次元", obs.shape == (21,), str(obs.shape))
check("観測が有限値", np.all(np.isfinite(obs)))

# 前進テスト: v=最大で 2 秒 -> 約 1 m 進むはず
x0, y0, yaw0 = env.robot_pose()
for _ in range(20):
    env.step([1.0, 0.0])
x1, y1, yaw1 = env.robot_pose()
d = math.hypot(x1 - x0, y1 - y0)
check("前進 (2秒で約1m)", 0.6 < d < 1.15, f"moved={d:.2f} m, yaw change={abs(yaw1 - yaw0):.2f} rad")

# 旋回テスト: ω=最大で 1 秒 -> 約 2 rad 回るはず
env.reset(seed=1)
_, _, a0 = env.robot_pose()
total = 0.0
prev = a0
for _ in range(10):
    env.step([0.0, 1.0])
    _, _, a = env.robot_pose()
    total += math.atan2(math.sin(a - prev), math.cos(a - prev))
    prev = a
check("旋回 (1秒で約2rad)", 1.2 < total < 2.4, f"turned={total:.2f} rad")

# 距離センサ: 壁が見えているか
env = DiffBotNavEnv()
obs, _ = env.reset(seed=3)
r = env.ranges
check("距離センサ", r.min() < 2.0 and r.max() <= 2.0, f"min={r.min():.2f} max={r.max():.2f}")

# 速度
t0 = time.time()
n = 0
for s in range(5):
    obs, _ = env.reset(seed=s)
    done = False
    while not done:
        obs, _, te, tr, info = env.step(env.action_space.sample())
        done = te or tr
        n += 1
sps = n / (time.time() - t0)
check("シミュレーション速度", sps > 300, f"{sps:.0f} steps/s")

# ルールベースで数エピソード
rule = RuleBasedController()
res = []
for s in range(10):
    obs, _ = env.reset(seed=100 + s)
    done = False
    while not done:
        obs, _, te, tr, info = env.step(rule(obs, env))
        done = te or tr
    res.append("success" if info["is_success"] else ("collision" if te else "timeout"))
print("    ルールベース 10 エピソード:", res)
check("ルールベースが一部到達できる", res.count("success") >= 2)

# 3D 描画
try:
    import mujoco
    rd = mujoco.Renderer(env.model, 240, 320)
    rd.update_scene(env.data)
    img = rd.render()
    check("3D 描画", img.std() > 1, f"{img.shape}")
except Exception as e:
    print("[WARN] 3D 描画不可 (動画は 2D 表示のみになります):", e)

try:
    import stable_baselines3, torch  # noqa: F401
    check("stable-baselines3 / torch", True, f"sb3 {stable_baselines3.__version__}, torch {torch.__version__}")
except Exception as e:
    check("stable-baselines3 / torch", False, str(e))

print("\n==> ALL OK" if ok else "\n==> どこかが FAIL です。このログを Claude に見せてください")
import sys
sys.exit(0 if ok else 1)
