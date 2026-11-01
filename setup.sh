#!/usr/bin/env bash
# 1) 仮想環境を作ってライブラリを入れ, 動作確認する  (5分程度)
set -eo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
mkdir -p logs
PY=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
echo "using $PY ($($PY --version))"
[ -d .venv ] || "$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python check_env.py 2>&1 | tee logs/check_env.txt
