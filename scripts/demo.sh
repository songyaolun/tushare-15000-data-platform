#!/usr/bin/env bash
# 场景：确定性离线端到端演示；预期：十类产品、Parquet 和覆盖债务可复核。
set -euo pipefail
cd "$(dirname "$0")/.."
TUSHARE_DEMO_DIR="${TUSHARE_DEMO_DIR:-$(mktemp -d /tmp/tushare-hub-demo.XXXXXX)}"
mkdir -p "$TUSHARE_DEMO_DIR"
uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" catalog sync --fixture fixtures/demo/catalog --observed-at 2024-01-03T00:00:00Z
for api in daily stock_basic moneyflow fund_daily fut_daily cb_daily sw_daily ths_hot cn_cpi dividend fund_adj; do
    uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" capability probe "$api" --fixture fixtures/demo/responses.json --raw-directory "$TUSHARE_DEMO_DIR/raw" --observed-at 2024-01-03T00:00:00Z
    start_day=2024-01-02
    if [ "$api" = cn_cpi ]; then start_day=2023-12-01; fi
    uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" ingest plan "$api" "$start_day" 2024-01-02 --output "$TUSHARE_DEMO_DIR/$api.plan.json" --observed-at 2024-01-03T00:00:00Z
    uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" ingest run "$TUSHARE_DEMO_DIR/$api.plan.json" --fixture fixtures/demo/responses.json --references fixtures/demo/references.json --raw-directory "$TUSHARE_DEMO_DIR/raw" --observed-at 2024-01-03T00:00:00Z
done
uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" quality audit daily --references fixtures/demo/references.json
uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" products build --as-of 2024-01-03 --output "$TUSHARE_DEMO_DIR/products" --parquet "$TUSHARE_DEMO_DIR/standard.parquet"
uv run tushare-hub --db "$TUSHARE_DEMO_DIR/hub.duckdb" coverage report --as-of 2024-01-03 --output "$TUSHARE_DEMO_DIR/coverage.json"
