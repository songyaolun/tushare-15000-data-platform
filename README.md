# Tushare 权益利用平台

面向 15,000 积分账户的可审计数据平台。持续登记接口与权益，按政策采集、保留历史版本，并生成信息产品与覆盖债务。独立购买接口和不确定文档不会被误计为已开通。

```bash
uv sync --all-extras --dev
uv run tushare-hub --help
uv run --no-project bash scripts/demo.sh
```

默认演示和测试完全离线，使用确定性合成数据。没有使用真实 Tushare Token，没有调用真实数据接口；演示产物不代表真实市场结论或账户权限。

- [架构说明](docs/架构说明.md)
- [运行手册](docs/运行手册.md)
- [安全说明](docs/安全说明.md)
- [贡献指南](docs/贡献指南.md)
- [接口目录与权益维护](docs/接口目录与权益维护.md)

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov=tushare_entitlement_hub --cov-report=term-missing --cov-fail-under=100
```

官方目录来自 [Tushare 接口文档](https://tushare.pro/document/2) 和 [积分与独立权限总表](https://tushare.pro/document/1?doc_id=290)。固定快照与抓取清单位于 fixtures/catalog/official。线上文档同步需显式执行 `catalog sync --online`；真实数据请求另需显式 live 选项。
