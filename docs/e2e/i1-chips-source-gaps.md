# I1筹码源缺失证据与显式兼容

适用revision：本文件所在提交，分支feat/i1-chips-source-gaps，基于
`732d824793266f9ad1bd1095ec280270a0efef08`。本次只修改原始证据及采集器，
不修改模型、股票池、交易或生产调度。下游Research/App精确冻结清单见App
`docs/e2e/i1-portable-inference.md`，按Hub→Research→App顺序推送。

环境：Linux/macOS、uv、Python3.11+；测试无Token、无网络；rebuild extra用于
现有CLI边界测试的Domain/Pandas依赖，其SHA以本提交pyproject.toml/uv.lock为准。

```bash
env -u PYTHONPATH uv run --frozen --extra rebuild pytest tests/test_return80_research_collection.py tests/test_client_raw.py tests/test_return80_research_boundaries.py tests/test_return80_research_sources.py -q
uv run --frozen ruff check src/tushare_entitlement_hub/client.py src/tushare_entitlement_hub/raw.py src/tushare_entitlement_hub/return80_research/collection.py tests/test_return80_research_collection.py
git diff --check
codegraph sync .
codegraph status .
```

离线链：MockTransport返回cyq_chips首屏50101及已知指定数据不存在消息→独立errors
原始envelope→provider_data_unavailable分片及SHA→显式allow_chips_unavailable
读取其余成功分片→重复运行验证证据且不重新请求。成功邻股数据必须完整保留，原错误
状态码不得变为0，不能生成成功空页。通用采集/读取默认严格，I1由下游显式启用。

失败关闭：非筹码接口、权限/未知错误、超时、后续页50101、无效股票/日期/参数、
证据损坏、请求不匹配及未经显式启用的恢复均停止。成功数据不能写入write_error；
凭据不得写入任一原始层。真实数据只读重放可选，另存仓外证据，不自动执行。

桌面及390x844浏览器not applicable：无UI。跨端用同一请求/fixture验证状态、内容
和SHA；本机测试不代表Mac实测。通过标准为全部上述门禁通过、证据独立且可恢复、
CodeGraph无pendingChanges。保存revision、退出码和日志到仓外。

2026-10-27公共采集迁移适用本文件所在提交：`return80_research/collection.py`迁至
`market_collection/collection.py`、`EXTRA_FIELDS`迁至`market_collection/fields.py`，
旧`return80_research.collection`保留薄re-export（对象同一）。请求字段、分页键、
`return80-collection-plan/v1` schema、原始页SHA/digest与缓存键均不变。
新增回归`tests/test_market_collection_contract.py`锁定re-export同一性、
`request`冻结契约与plan schema/digest稳定；Research I1已改指
`tushare_entitlement_hub.market_collection.*`。
不发送通知、不部署、不交易、不注册调度；失败保留证据并回滚上一个冻结SHA。
