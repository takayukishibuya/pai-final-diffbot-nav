#!/usr/bin/env bash
# 2) 学習 -> 評価 -> 動画生成 を一気に行う  (MacBook Air で 15 分程度)
#    学習ステップ数を変えたい場合:  STEPS=1000000 bash run_all.sh
set -eo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs
python check_env.py 2>&1 | tee logs/check_env.txt
python train.py --steps "${STEPS:-1500000}" --envs "${ENVS:-8}" 2>&1 | tee logs/train.txt
python evaluate.py --episodes 300 2>&1 | tee logs/evaluate.txt
python make_video.py 2>&1 | tee logs/video.txt
echo "完了: video/pai_final_demo.mp4"
