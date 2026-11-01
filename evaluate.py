"""学習済み AI とルールベースを同じランダム環境で比較評価し, 学習曲線も描く

例:  python evaluate.py --model models/best/best_model.zip --episodes 300
出力: results/eval.json, results/eval_summary.md, results/comparison.png, results/learning_curve.png
"""
import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from diffbot_nav.controllers import PPOController, RuleBasedController  # noqa: E402
from diffbot_nav.env import CTRL_DT, DiffBotNavEnv  # noqa: E402

for f in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Noto Sans CJK JP", "IPAexGothic", "Yu Gothic", "Meiryo"]:
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break


def run(controller, episodes, seed0):
    env = DiffBotNavEnv()
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        controller.reset()
        n_obs = len(env.active_obstacles)
        x0, y0, _ = env.robot_pose()
        path, prev = 0.0, (x0, y0)
        done, info = False, {}
        while not done:
            obs, r, term, trunc, info = env.step(controller(obs, env))
            x, y, _ = env.robot_pose()
            path += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            done = term or trunc
        outcome = "success" if info["is_success"] else ("collision" if term else "timeout")
        rows.append(dict(seed=seed0 + ep, n_obstacles=n_obs, outcome=outcome,
                         time=env.steps * CTRL_DT, path=path))
    return rows


def summarize(rows):
    n = len(rows)
    succ = [r for r in rows if r["outcome"] == "success"]
    s = dict(
        episodes=n,
        success_rate=len(succ) / n,
        collision_rate=sum(r["outcome"] == "collision" for r in rows) / n,
        timeout_rate=sum(r["outcome"] == "timeout" for r in rows) / n,
        mean_time_success=float(np.mean([r["time"] for r in succ])) if succ else None,
        mean_path_success=float(np.mean([r["path"] for r in succ])) if succ else None,
    )
    bins = {"4-6個": (4, 6), "7-8個": (7, 8), "9-10個": (9, 10)}
    s["success_by_obstacles"] = {}
    for k, (lo, hi) in bins.items():
        sub = [r for r in rows if lo <= r["n_obstacles"] <= hi]
        if sub:
            s["success_by_obstacles"][k] = sum(r["outcome"] == "success" for r in sub) / len(sub)
    return s


def plot_comparison(summ, path):
    names = list(summ.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cats = ["success_rate", "collision_rate", "timeout_rate"]
    labels = ["ゴール到達", "衝突", "時間切れ"]
    colors = ["#2a9d8f", "#e76f51", "#8d99ae"]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for c, lab, col in zip(cats, labels, colors):
        vals = np.array([summ[nm][c] for nm in names]) * 100
        axes[0].bar(x, vals, bottom=bottom, color=col, label=lab, width=0.55)
        for i, v in enumerate(vals):
            if v >= 6:
                axes[0].text(x[i], bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white",
                             fontsize=11, fontweight="bold")
        bottom += vals
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("割合 [%]")
    axes[0].set_title("結果の内訳")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    axes[0].set_ylim(0, 100)

    bins = list(next(iter(summ.values()))["success_by_obstacles"].keys())
    w = 0.38
    for i, nm in enumerate(names):
        vals = [summ[nm]["success_by_obstacles"].get(b, 0) * 100 for b in bins]
        axes[1].bar(np.arange(len(bins)) + (i - 0.5) * w, vals, width=w, label=nm,
                    color=["#8d99ae", "#2a9d8f"][i % 2])
    axes[1].set_xticks(np.arange(len(bins)), bins)
    axes[1].set_xlabel("障害物の数")
    axes[1].set_ylabel("ゴール到達率 [%]")
    axes[1].set_title("障害物の数ごとの到達率")
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curve(path, window=300):
    files = glob.glob("logs/train_monitor/*.monitor.csv")
    if not files:
        return False
    import csv
    eps = []
    for f in files:
        with open(f) as fh:
            fh.readline()
            for row in csv.DictReader(fh):
                eps.append((float(row["t"]), float(row["r"]), int(row["l"]),
                            row.get("is_success", "False") == "True"))
    eps.sort()
    steps = np.cumsum([e[2] for e in eps])
    rew = np.array([e[1] for e in eps])
    suc = np.array([e[3] for e in eps], dtype=float)
    k = min(window, max(len(eps) // 10, 1))
    ma = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")  # noqa: E731
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps[k - 1:] / 1e6, ma(suc) * 100, color="#2a9d8f", label="ゴール到達率 (移動平均)")
    ax1.set_xlabel("学習ステップ数 [百万]")
    ax1.set_ylabel("ゴール到達率 [%]", color="#2a9d8f")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(steps[k - 1:] / 1e6, ma(rew), color="#264653", alpha=0.6, label="エピソード報酬")
    ax2.set_ylabel("エピソード報酬 (移動平均)", color="#264653")
    ax1.set_title("PPO の学習曲線")
    ax1.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best/best_model.zip")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=10_000)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    ctrls = [RuleBasedController(), PPOController(args.model)]
    rows, summ = {}, {}
    for c in ctrls:
        rows[c.name] = run(c, args.episodes, args.seed)
        summ[c.name] = summarize(rows[c.name])
        print(c.name, json.dumps(summ[c.name], ensure_ascii=False, indent=1))

    with open("results/eval.json", "w") as f:
        json.dump(dict(summary=summ, episodes=rows), f, ensure_ascii=False, indent=1)
    with open("results/eval_summary.md", "w") as f:
        f.write(f"| 制御器 | 到達率 | 衝突率 | 時間切れ | 到達時の平均所要時間 | 到達時の平均経路長 |\n|---|---|---|---|---|---|\n")
        for nm, s in summ.items():
            t = f"{s['mean_time_success']:.1f} s" if s["mean_time_success"] else "-"
            p = f"{s['mean_path_success']:.2f} m" if s["mean_path_success"] else "-"
            f.write(f"| {nm} | {s['success_rate']*100:.1f}% | {s['collision_rate']*100:.1f}% | "
                    f"{s['timeout_rate']*100:.1f}% | {t} | {p} |\n")
    plot_comparison(summ, "results/comparison.png")
    plot_learning_curve("results/learning_curve.png")
    print("saved results/")


if __name__ == "__main__":
    main()
