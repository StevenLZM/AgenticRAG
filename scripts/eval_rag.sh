#!/usr/bin/env bash
set -euo pipefail
eval_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$eval_repo_root"
eval_python="${RAG_EVAL_PYTHON:-$eval_repo_root/var/eval-venv-ragas042/bin/python}"
if [[ ! -x "$eval_python" ]]; then
  echo "Missing Ragas Python runtime: set RAG_EVAL_PYTHON to an existing configured interpreter." >&2
  exit 2
fi
export PYTHONPATH="$eval_repo_root/src:$eval_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
exec "$eval_python" -m evals.command "$@"
