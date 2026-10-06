#!/usr/bin/env bash
# 场景：CI 或本地验收；预期：依赖锁定、静态检查、100%覆盖及离线演示全部通过。
set -euo pipefail
cd "$(dirname "$0")/.."
# Flux 优先使用现有临时 uv；其他环境使用预装 uv，脚本不安装工具。
TUSHARE_UV_BIN=uv
if [ -x /tmp/tushare-tools/bin/uv ]; then
    TUSHARE_UV_BIN=/tmp/tushare-tools/bin/uv
fi
# 场景：预装镜像或本地解释器变化；预期：打印实际版本，低于 3.11 时立即失败。
# --no-sync 在安装依赖前检查后续项目命令使用的解释器，所有 Python 调用仍经 uv。
"$TUSHARE_UV_BIN" run --no-sync python -c 'import sys; print("Python", sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
# 实际内存由容器 cgroup 与 DuckDB 自身限制；不限制分配器预留的虚拟地址空间。
# 各阶段顺序执行，依赖安装、Ruff、数值库和 pytest 均保持单任务或单线程。
UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 "$TUSHARE_UV_BIN" sync --locked --all-extras --dev
RAYON_NUM_THREADS=1 "$TUSHARE_UV_BIN" run ruff check .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TUSHARE_UV_BIN" run mypy
# 数百万投影和数百 MiB 宽结果压力用例已独立验收，避免超过平台两小时作业时限。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TUSHARE_UV_BIN" run pytest -m "not resource_intensive" --cov=tushare_entitlement_hub --cov-report=term-missing --cov-fail-under=100
"$TUSHARE_UV_BIN" run tushare-hub --help
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$TUSHARE_UV_BIN" run --no-project bash scripts/demo.sh
