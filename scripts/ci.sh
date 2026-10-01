#!/usr/bin/env bash
# 场景：CI 或本地验收；预期：依赖锁定、静态检查、100%覆盖及离线演示全部通过。
set -euo pipefail
cd "$(dirname "$0")/.."
# 单进程取 cgroup 内存上限四分之一且最多 4 GiB，pytest 不启用 xdist。
ceiling=17179869184
if [ -r /sys/fs/cgroup/memory.max ]; then
    read -r ceiling < /sys/fs/cgroup/memory.max
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    read -r ceiling < /sys/fs/cgroup/memory/memory.limit_in_bytes
fi
budget_kb=4194304
if [ "$ceiling" != max ] && [ "$ceiling" -lt 17179869184 ]; then
    budget_kb=$((ceiling / 4096))
fi
ulimit -v "$budget_kb"
UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 uv sync --locked --all-extras --dev
uv run ruff check .
uv run mypy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pytest --cov=tushare_entitlement_hub --cov-report=term-missing --cov-fail-under=100
uv run tushare-hub --help
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/demo.sh
