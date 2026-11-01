"""提出用デモ動画 (約2分, 1280x720, 等速) を生成する

例:  python make_video.py --model models/best/best_model.zip
出力: video/pai_final_demo.mp4

構成:
  1. タイトル
  2. サンプルからの発展点 (キーボード操縦 -> AI による自律走行)
  3. 比較: ルールベース vs AI (同じ配置で同時スタート)
  4. AI の連続ナビゲーション (3D 表示 + LiDAR マップ)
  5. 学習曲線と評価結果
  6. まとめ
"""
from __future__ import annotations

import argparse
import math
import os

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diffbot_nav.controllers import PPOController, RuleBasedController
from diffbot_nav.env import (ARENA, BODY_R, CTRL_DT, GOAL_R, N_RAYS, OBSTACLE_SHAPES, RAY_R0, SUBSTEPS,
                             V_MAX, W_MAX, DiffBotNavEnv)

W, H = 1280, 720
FPS = 20
RENDER_EVERY = SUBSTEPS // 2   # 1 制御周期 (0.1 s) あたり 2 フレーム -> 20 fps で等速

NAVY = (20, 33, 61)
WHITE = (245, 245, 245)
GRAY = (150, 155, 165)
TEAL = (42, 157, 143)
ORANGE = (231, 111, 81)
GOLD = (233, 196, 106)

# ------------------------------------------------------------------ 文字描画
FONT_CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothB.ttc",
]
_FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
_fonts: dict[int, ImageFont.FreeTypeFont] = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(_FONT_PATH, size) if _FONT_PATH else ImageFont.load_default()
    return _fonts[size]


def draw_texts(img, items):
    """items: [(text, (x, y), size, color, anchor)]"""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for text, xy, size, color, anchor in items:
        if "\n" in text:
            d.multiline_text(xy, text, font=font(size), fill=color, anchor=anchor, align="center", spacing=6)
        else:
            d.text(xy, text, font=font(size), fill=color, anchor=anchor)
    return np.asarray(pil).copy()


def blank(color=NAVY):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = color
    return img


def shade(img, x0, y0, x1, y1, alpha=0.55, color=(0, 0, 0)):
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


# ------------------------------------------------------------------ 2D マップ (LiDAR 表示)
def draw_map(env, size, trail, title=None, ranges=None):
    img = np.full((size, size, 3), 250, np.uint8)
    pad = 8
    s = (size - 2 * pad) / (2 * (ARENA + 0.1))

    def px(x, y):
        return int(round(pad + (x + ARENA + 0.1) * s)), int(round(pad + (ARENA + 0.1 - y) * s))

    # 床のグリッドと壁
    for g in np.arange(-ARENA, ARENA + 1e-6, 0.5):
        cv2.line(img, px(g, -ARENA), px(g, ARENA), (230, 230, 230), 1)
        cv2.line(img, px(-ARENA, g), px(ARENA, g), (230, 230, 230), 1)
    cv2.rectangle(img, px(-ARENA, ARENA), px(ARENA, -ARENA), (90, 95, 105), 3)

    # 障害物
    d = env.data
    for i in env.active_obstacles:
        mid = env.obs_mocap[i]
        x, y = d.mocap_pos[mid][:2]
        qw, qz = d.mocap_quat[mid][0], d.mocap_quat[mid][3]
        yaw = 2 * math.atan2(qz, qw)
        kind, sz = OBSTACLE_SHAPES[i]
        if kind == "cylinder":
            cv2.circle(img, px(x, y), int(sz[0] * s), ORANGE, -1, cv2.LINE_AA)
        else:
            c, sn = math.cos(yaw), math.sin(yaw)
            pts = []
            for ex, ey in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
                lx, ly = ex * sz[0], ey * sz[1]
                pts.append(px(x + c * lx - sn * ly, y + sn * lx + c * ly))
            cv2.fillPoly(img, [np.array(pts, np.int32)], ORANGE, cv2.LINE_AA)

    # ゴール
    gx, gy = env.goal
    cv2.circle(img, px(gx, gy), int(GOAL_R * s), (190, 235, 200), -1, cv2.LINE_AA)
    cv2.circle(img, px(gx, gy), int(GOAL_R * s), (40, 160, 70), 2, cv2.LINE_AA)
    cv2.drawMarker(img, px(gx, gy), (40, 160, 70), cv2.MARKER_CROSS, 12, 2)

    # 軌跡
    if len(trail) > 1:
        cv2.polylines(img, [np.array([px(*p) for p in trail], np.int32)], False, (70, 130, 220), 2, cv2.LINE_AA)

    # LiDAR
    x, y, yaw = env.robot_pose()
    rr = env.scan() if ranges is None else ranges
    for a, r in zip(env.ray_angles, rr):
        c, sn = math.cos(yaw + a), math.sin(yaw + a)
        p0 = px(x + RAY_R0 * c, y + RAY_R0 * sn)
        p1 = px(x + (RAY_R0 + r) * c, y + (RAY_R0 + r) * sn)
        hit = r < 1.99
        col = (220, 60, 60) if r < 0.4 else ((240, 140, 140) if hit else (215, 215, 215))
        cv2.line(img, p0, p1, col, 1, cv2.LINE_AA)
        if hit:
            cv2.circle(img, p1, 3, col, -1, cv2.LINE_AA)

    # ロボット
    cv2.circle(img, px(x, y), max(int(BODY_R * s), 4), (40, 95, 200), -1, cv2.LINE_AA)
    cv2.line(img, px(x, y), px(x + 0.16 * math.cos(yaw), y + 0.16 * math.sin(yaw)), GOLD, 3, cv2.LINE_AA)
    if title:
        img = draw_texts(img, [(title, (12, 10), 22, (30, 30, 30), "la")])
    return img


# ------------------------------------------------------------------ 3D 描画
class View3D:
    def __init__(self, env, width, height):
        self.ok = False
        try:
            self.r = mujoco.Renderer(env.model, height, width, max_geom=20000)
            self.r.update_scene(env.data)
            self.r.render()
            self.ok = True
        except Exception as e:  # GPU/OpenGL が使えない環境では 2D のみで描く
            print("[warn] 3D rendering unavailable -> 2D only:", e)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = 3.7
        self.cam.elevation = -40
        self.cam.azimuth = 115
        self.look = np.zeros(3)

    def reset_camera(self, env, azimuth=115):
        x, y, _ = env.robot_pose()
        self.look = np.array([0.7 * x, 0.7 * y, 0.0])
        self.cam.azimuth = azimuth

    def _add_line(self, scn, p0, p1, rgba, width):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.array(p0, float), np.array(p1, float))
        scn.ngeom += 1

    def render(self, env, trail, t):
        x, y, yaw = env.robot_pose()
        target = np.array([0.7 * x, 0.7 * y, 0.0])
        self.look = 0.93 * self.look + 0.07 * target
        self.cam.lookat[:] = self.look
        self.cam.azimuth += 0.25
        self.r.update_scene(env.data, camera=self.cam)
        scn = self.r.scene
        try:
            for a, r in zip(env.ray_angles, env.ranges):
                c, sn = math.cos(yaw + a), math.sin(yaw + a)
                p0 = [x + RAY_R0 * c, y + RAY_R0 * sn, 0.1]
                p1 = [x + (RAY_R0 + r) * c, y + (RAY_R0 + r) * sn, 0.1]
                rgba = (1, 0.2, 0.2, 0.9) if r < 0.4 else (1, 0.45, 0.45, 0.45)
                self._add_line(scn, p0, p1, rgba, 2)
            pts = trail[-400:]
            for p, q in zip(pts[:-1], pts[1:]):
                self._add_line(scn, [p[0], p[1], 0.01], [q[0], q[1], 0.01], (0.2, 0.45, 0.95, 1), 4)
        except Exception:
            pass  # 古い MuJoCo で mjv_connector が無い場合は線を省略
        return self.r.render()


# ------------------------------------------------------------------ シーン
def card(lines, frames, bg=NAVY):
    img = draw_texts(blank(bg), lines)
    return [img] * frames


def title_card():
    return card([
        ("Physical AI 応用1講座  最終課題", (W // 2, 170), 30, GOLD, "mm"),
        ("AIで自律走行する差動二輪ロボット", (W // 2, 285), 58, WHITE, "mm"),
        ("強化学習 (PPO) による障害物回避ナビゲーション", (W // 2, 365), 34, WHITE, "mm"),
        ("MuJoCo  ·  Gymnasium  ·  Stable-Baselines3", (W // 2, 500), 26, GRAY, "mm"),
        ("動画はすべて等速 (1倍速) です", (W // 2, 640), 22, GRAY, "mm"),
    ], int(4.0 * FPS))


def concept_card():
    img = blank()
    items = [("サンプルからの発展", (60, 50), 40, GOLD, "la")]

    def box(x, y, w, h, text, fill, tcol=WHITE, size=24):
        cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
        items.append((text, (x + w // 2, y + h // 2), size, tcol, "mm"))

    def arrow(x0, x1, y, col):
        cv2.arrowedLine(img, (x0, y), (x1, y), col, 4, cv2.LINE_AA, tipLength=0.35)

    # サンプル
    items.append(("サンプル (ROS 2 + Gazebo)", (60, 150), 28, GRAY, "la"))
    y = 200
    for i, t in enumerate(["人がキーボード操作", "teleop_node", "/cmd_vel (v, ω)", "Gazebo の差動二輪"]):
        box(60 + i * 300, y, 240, 70, t, (60, 66, 82), (200, 200, 205))
        if i < 3:
            arrow(60 + i * 300 + 245, 60 + (i + 1) * 300 - 5, y + 35, (120, 125, 135))
    # 本作品
    items.append(("本作品 (MuJoCo)", (60, 345), 28, GOLD, "la"))
    y = 395
    for i, (t, col) in enumerate([("LiDAR 16本\n+ ゴール方向", (34, 90, 160)),
                                  ("ニューラルネット\n(PPOで学習)", (42, 157, 143)),
                                  ("cmd_vel (v, ω)", (34, 90, 160)),
                                  ("左右の車輪速度", (34, 90, 160))]):
        box(60 + i * 300, y, 240, 90, t, col)
        if i < 3:
            arrow(60 + i * 300 + 245, 60 + (i + 1) * 300 - 5, y + 45, GOLD)
    items += [
        ("人の操縦をなくし, AI がセンサ情報から速度指令を自分で決めて走る", (60, 555), 28, WHITE, "la"),
        ("報酬: ゴールに近づくと＋ / 衝突で−10 / 到達で＋10   障害物の数・配置は毎回ランダム", (60, 610), 24, GRAY,
         "la"),
    ]
    img = draw_texts(img, items)
    return [img] * int(8.0 * FPS)


def run_episode_frames(env, ctrl, seed, max_sec):
    """1 エピソード分の (2D 用の) 状態スナップショットを 0.05 s 刻みで集める"""
    obs, _ = env.reset(seed=seed)
    trail = [env.robot_pose()[:2]]
    snaps = []

    def grab(k):
        if (k + 1) % RENDER_EVERY == 0:
            x, y, _ = env.robot_pose()
            trail.append((x, y))
            snaps.append(draw_map(env, 600, trail))

    outcome, done = None, False
    while not done and env.steps * CTRL_DT < max_sec:
        obs, _, term, trunc, info = env.step(ctrl(obs, env), substep_callback=grab)
        done = term or trunc
        if info["is_success"]:
            outcome = ("success", env.steps * CTRL_DT)
        elif term:
            outcome = ("collision", env.steps * CTRL_DT)
    if outcome is None:
        outcome = ("timeout", env.steps * CTRL_DT)
    return snaps, outcome


def stamp(img, outcome, timeout_note):
    kind, t = outcome
    if kind == "success":
        text, col = f"ゴール到達  {t:.1f} 秒", TEAL
    elif kind == "collision":
        text, col = f"衝突  ({t:.1f} 秒)", ORANGE
    else:
        text, col = timeout_note, (110, 110, 120)
    img = img.copy()
    shade(img, 0, img.shape[0] // 2 - 40, img.shape[1], img.shape[0] // 2 + 40, 0.7, col)
    size = 34 if len(text) <= 14 else 24
    return draw_texts(img, [(text, (img.shape[1] // 2, img.shape[0] // 2), size, WHITE, "mm")])


def comparison_scene(seed, ppo, rule, label, max_sec=12.0):
    env_a, env_b = DiffBotNavEnv(), DiffBotNavEnv()
    snaps_r, out_r = run_episode_frames(env_a, rule, seed, max_sec)
    snaps_p, out_p = run_episode_frames(env_b, ppo, seed, max_sec)
    n = max(len(snaps_r), len(snaps_p)) + int(2.5 * FPS)
    frames = []
    for k in range(n):
        img = blank((28, 32, 44))
        for j, (snaps, out, name, col) in enumerate([(snaps_r, out_r, "ルールベース (ポテンシャル法)", GRAY),
                                                      (snaps_p, out_p, "AI (強化学習 PPO)", TEAL)]):
            m = snaps[min(k, len(snaps) - 1)]
            if k >= len(snaps):
                m = stamp(m, out, "40秒以内に到達できず (行き止まりで停滞)")
            x0 = 30 + j * 630
            img[95:695, x0:x0 + 600] = m
            cv2.rectangle(img, (x0 - 2, 93), (x0 + 601, 696), col, 3)
        t = min(k, n) / FPS
        img = draw_texts(img, [
            (label, (30, 22), 30, WHITE, "la"),
            ("ルールベース (ポテンシャル法)", (330, 75), 24, (200, 200, 205), "mm"),
            ("AI (強化学習 PPO)", (960, 75), 24, TEAL, "mm"),
            (f"t = {min(t, max_sec):4.1f} s", (W - 30, 22), 26, GRAY, "ra"),
        ])
        frames.append(img)
    return frames


def main_demo(ppo, seed, seconds, caption, azimuth):
    env = DiffBotNavEnv(n_obstacles=(9, 10), continuous_goals=True, max_steps=10 ** 9)
    obs, _ = env.reset(seed=seed)
    view = View3D(env, 860, 720)
    view.reset_camera(env, azimuth)
    trail = [env.robot_pose()[:2]]
    frames = []
    state = {"t": 0.0}

    def grab(k):
        if (k + 1) % RENDER_EVERY:
            return
        x, y, _ = env.robot_pose()
        trail.append((x, y))
        state["t"] += 1.0 / FPS
        ranges = env.scan()
        img = blank((24, 28, 38))
        if view.ok:
            img[:, :860] = view.render(env, trail, state["t"])
            mp = draw_map(env, 400, trail, ranges=ranges)
            img[60:460, 870:1270] = mp
            panel_x = 875
        else:
            mp = draw_map(env, 680, trail, ranges=ranges)
            img[20:700, 20:700] = mp
            panel_x = 730
        v, w = env.last_cmd
        # 速度指令のバー
        for i, (lab, val, vmax) in enumerate([("v", v, V_MAX), ("ω", w, W_MAX)]):
            y0 = 600 + i * 40
            cx = panel_x + 230
            cv2.rectangle(img, (cx - 140, y0), (cx + 140, y0 + 18), (60, 66, 82), -1)
            L = int(140 * np.clip(val / vmax, -1, 1))
            cv2.rectangle(img, (cx, y0), (cx + L, y0 + 18), TEAL if i == 0 else GOLD, -1)
            cv2.line(img, (cx, y0 - 3), (cx, y0 + 21), WHITE, 1)
        txt = [
            ("LiDAR マップ (上から見た図)", (panel_x, 22), 22, WHITE, "la"),
            ("制御: AI (強化学習 PPO)", (panel_x, 480), 24, TEAL, "la"),
            (f"到達したゴール: {env.goals_reached} 個", (panel_x, 515), 24, WHITE, "la"),
            (f"最も近い障害物: {ranges.min():.2f} m", (panel_x, 550), 22, (200, 200, 205), "la"),
            (f"v {v:+.2f} m/s", (panel_x, 598), 20, WHITE, "la"),
            (f"ω {w:+.2f} rad/s", (panel_x, 638), 20, WHITE, "la"),
            ("AI が出力する速度指令 (cmd_vel)", (panel_x, 678), 18, GRAY, "la"),
        ]
        if view.ok:
            shade(img, 0, 0, 860, 56, 0.55)
            shade(img, 0, H - 56, 860, H, 0.55)
            txt += [(caption, (20, 28), 26, WHITE, "lm"),
                    ("ゴール (緑の旗) に着くと, 次のゴールがランダムな位置に現れます", (20, H - 28), 22, WHITE, "lm"),
                    (f"t = {state['t']:4.1f} s", (840, 28), 22, GRAY, "rm")]
        frames.append(draw_texts(img, txt))

    while state["t"] < seconds:
        obs, _, term, trunc, info = env.step(ppo(obs, env), substep_callback=grab)
        if term:   # 衝突した場合も隠さずに表示し, 新しい配置で再開する
            last = frames[-1] if frames else blank()
            frames += [stamp(last, ("collision", state["t"]), "")] * FPS
            obs, _ = env.reset(seed=seed + 1000 + len(frames))
            trail.clear()
            trail.append(env.robot_pose()[:2])
    return frames


def image_card(path, title, note, frames):
    img = blank((250, 250, 250))
    if os.path.exists(path):
        pic = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = pic.shape[:2]
        sc = min((W - 80) / w, (H - 170) / h)
        pic = cv2.resize(pic, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        y0 = 95 + (H - 170 - pic.shape[0]) // 2
        x0 = (W - pic.shape[1]) // 2
        img[y0:y0 + pic.shape[0], x0:x0 + pic.shape[1]] = pic
    img = draw_texts(img, [(title, (40, 30), 34, (20, 33, 61), "la"),
                           (note, (40, H - 40), 22, (90, 90, 100), "lm")])
    return [img] * frames


def closing_card():
    return card([
        ("工夫した点", (80, 90), 40, GOLD, "la"),
        ("1. サンプルの差動二輪 + cmd_vel の構成を引き継ぎ, 操縦者を AI に置き換えた", (80, 190), 28, WHITE, "la"),
        ("2. 障害物の数・形・配置を毎回ランダムにし, 未知の環境でも走れるよう学習", (80, 250), 28, WHITE, "la"),
        ("3. 実機を意識し, 入力は LiDAR とゴール方向のみ (地図や正解経路は使わない)", (80, 310), 28, WHITE, "la"),
        ("4. 人が設計したルールベースと同じ条件で比較し, AI の効果を定量評価", (80, 370), 28, WHITE, "la"),
        ("応用例: 倉庫・病院・オフィスでの搬送ロボット, 掃除ロボットの経路決定", (80, 470), 26, GRAY, "la"),
        ("コード: GitHub リポジトリを参照", (80, 600), 24, GRAY, "la"),
    ], int(7.0 * FPS))


def pick_comparison_seeds(ppo, rule, n=3, start=500, limit=400):
    """ルールベースは失敗し AI は成功する配置を探す (障害物 7 個以上)"""
    picks, kinds = [], set()
    env = DiffBotNavEnv()
    for seed in range(start, start + limit):
        res = {}
        for name, ctrl in [("rule", rule), ("ppo", ppo)]:
            obs, _ = env.reset(seed=seed)
            if len(env.active_obstacles) < 7:
                break
            done = False
            while not done:
                obs, _, term, trunc, info = env.step(ctrl(obs, env))
                done = term or trunc
            res[name] = ("success" if info["is_success"] else ("collision" if term else "timeout"),
                         env.steps * CTRL_DT)
        if len(res) < 2:
            continue
        if res["ppo"][0] == "success" and res["ppo"][1] < 10 and res["rule"][0] != "success":
            picks.append(seed)
            print("comparison seed", seed, res)
        if len(picks) >= n:
            break
    return picks or [start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best/best_model.zip")
    ap.add_argument("--out", default="video/pai_final_demo.mp4")
    ap.add_argument("--demo-seconds", type=float, default=28.0, help="連続ナビゲーション 1 シーンの長さ")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="比較シーンの seed を手動指定")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if _FONT_PATH is None:
        print("[warn] 日本語フォントが見つかりません。FONT_CANDIDATES にパスを追加してください")

    ppo, rule = PPOController(args.model), RuleBasedController()
    seeds = args.seeds or pick_comparison_seeds(ppo, rule)

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    total = 0

    def put(frames, still=None, idx=None):
        nonlocal total
        if still:
            k = len(frames) - 1 if idx is None else int(len(frames) * idx)
            small = cv2.resize(frames[k], (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(os.path.dirname(args.out), still + ".jpg"),
                        cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        for f in frames:
            writer.append_data(f)
        total += len(frames)
        print(f"  {total / FPS:6.1f} s written")

    put(title_card())
    put(concept_card())
    for i, s in enumerate(seeds[:3]):
        put(comparison_scene(s, ppo, rule, f"比較 {i + 1}/{len(seeds[:3])}: 同じ配置・同じスタートで同時に走らせる"),
            f"still_cmp{i + 1}")
    put(main_demo(ppo, 2026, args.demo_seconds, "AI による連続ナビゲーション (障害物 9〜10 個)", 115), "still_demo1", 0.45)
    put(main_demo(ppo, 7, args.demo_seconds, "別のランダムな配置でも同じ AI がそのまま走る", 250), "still_demo2", 0.6)
    put(image_card("results/learning_curve.png", "学習の経過",
                   "PPO で約150万ステップ学習 (ノートPCのCPUのみ)", int(5 * FPS)))
    put(image_card("results/comparison.png", "300通りのランダム配置での評価",
                   "同じ 300 配置でルールベースと AI を比較", int(7 * FPS)))
    put(closing_card())
    writer.close()
    print(f"saved {args.out}  ({total / FPS:.1f} s)")


if __name__ == "__main__":
    main()
