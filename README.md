# Tushare 权益利用平台

面向 15,000 积分账户的可审计数据平台。持续登记接口与权益，按政策采集、保留历史版本，并生成信息产品与覆盖债务。独立购买接口和不确定文档不会被误计为已开通。

```bash
uv sync --all-extras --dev
uv run tushare-hub --help
uv run --no-project bash scripts/demo.sh
```

默认演示和测试完全离线，使用确定性合成数据。没有使用真实 Tushare Token，没有调用真实数据接口；演示产物不代表真实市场结论或账户权限。

真实接入支持按账户批量验证全部公共只读接口，目录中的独立权限、未知权限、停更或文档待复核标签不会阻止探测。成功证据用于放行后续采集，文档事实保留。参见[真实接口接入](docs/真实接口接入.md)，其中说明批量扫描、已有凭据复用、原始样本和 `pro_bar` 的 HTTPS 组合适配。

- [架构说明](docs/架构说明.md)
- [运行手册](docs/运行手册.md)
- [安全说明](docs/安全说明.md)
- [贡献指南](docs/贡献指南.md)
- [接口目录与权益维护](docs/接口目录与权益维护.md)
- [产品加工基础层](docs/产品加工基础层.md)
- [产品说明书](docs/产品说明书.md)
- [股票研究闭环](docs/股票研究闭环.md)
- [全产品离线研究](docs/全产品离线研究.md)
- [主题预期与基金研究](docs/主题预期与基金研究.md)
- [跨市场与融资研究](docs/跨市场与融资研究.md)

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov=tushare_entitlement_hub --cov-report=term-missing --cov-fail-under=100
```

官方目录来自 [Tushare 接口文档](https://tushare.pro/document/2) 和 [积分与独立权限总表](https://tushare.pro/document/1?doc_id=290)。固定快照与抓取清单位于 fixtures/catalog/official。线上文档同步需显式执行 `catalog sync --online`；真实数据请求另需显式 live 选项。

通用本地数据层支持正式采集、标准化、逐行验收以及 JSONL、Parquet、DuckDB 导出。后续产品可直接复用数据，不受既有产品分类限制。运行方式和质量边界见[本地数据参考](docs/本地数据参考.md)。

共享产品基础层通过 `product-foundation initialize/validate` 离线读取参考库，提供实体、时间、单位和覆盖审计。`research initialize/run/validate` 保留原六产品的 99 个真实消费者，并可显式运行 P03S 短期交易、P03M 中期基本面、P03L 长期配置三套独立排名。旧 P03 的公式、标识和默认行为继续兼容。

`research-all build-config/run/validate` 增加 P02、P05、P08、P09、P10、P11、P12 的 107 个消费者，默认统一运行原十三产品；P03S、P03M、P03L 需显式选择，不会悄悄改变既有默认批次。结果保留业务指标、证据、门禁和血缘，逐接口区分注册、分配、执行、可读取、已配置、已观测与未解析。缺失历史或证据会明确降级，历史 PIT 不自动认证。正式数据的 observed 数量须以本地全量运行和缺口审计为准；命令与容量边界见[全产品离线研究](docs/全产品离线研究.md)。
