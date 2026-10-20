# Return80 独立研究机验收

## Revision 与范围

实现基于研究分支 `42cb57f0706a010337952d23cca7be691d3db1be`，通过
`feat/return80-research-portability` 集成到研究主线。最终实现 revision：
`faaff711ec5866c2b1bcfd2e71dbafa854e22857`；后续本计划与使用说明提交不改变实现。
GitHub PR：<https://github.com/songyaolun/tushare-15000-data-platform/pull/4>。

本次发布采集计划、来源适配、47 列/标签重建、冻结数据导入、新特征追加、原生 HGB
训练与独立回放入口。保留全样本研究、126 交易日半衰期和原经济规则；不改变
StockOps App、生产模型、账户或推送配置。新机步骤见
[独立研究机工作流](../stockops-research-workspace.md)。

依赖按 contracts → data → domain → 研究库安装，根 `uv.lock` 与来源配置固定：

| 依赖 | 精确 commit SHA |
|---|---|
| stockops-contracts | `292fdebc3f22fe48e2b3e637c4491dfe58d1e81b` |
| stockops-data | `3a122705f57de8d024241aa2bae9abd0a12feec9` |
| stockops-domain | `6d6313634734fa7d06cd3414a169fa61e0adb4d9` |

训练 extra 为 sklearn 1.9.1，TA-Lib 为 0.7.1。域层不反向依赖研究库。

## 环境与离线必跑链路

Python 3.11 与 uv；安装使用 `uv sync --frozen --all-extras --group dev --no-editable`。
清除 `PYTHONPATH`，核对模块来自安装目录，不能从另一个 checkout 导入未发布源码。
首次安装需要 GitHub 和包源访问，重建依赖使用 GitHub SSH。测试不需要 Tushare Token
或网络；采集使用真实 Client/RawStore 配合 FakeTransport。

必跑链路为：分页原文 → materialize-sources → history/dataset/panel → 47 列/标签/权重
→ 按日期证券追加特征 → 原生小样本训练 → 预测映射 → 正常/双成本回放 → 账本核验。
另一条从冻结 ZIP 导入，与原始重建使用同一 `return80-training-inputs/v1` 契约。
真实 51200 轮大模型不在软件验收中重新训练。

```bash
env -u PYTHONPATH uv sync --frozen --all-extras --group dev --no-editable
env -u PYTHONPATH uv run --frozen --all-extras --no-editable pytest tests/test_return80_research_*.py tests/test_replay_inputs.py tests/test_replay_candidate_count.py -q
uv run --frozen --all-extras ruff check .
uv run --frozen --all-extras mypy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen --all-extras pytest -m 'not resource_intensive' --cov=src/tushare_entitlement_hub --cov-report=term-missing --cov-fail-under=100
```

全仓门禁沿用 CI 的 `not resource_intensive` 范围，数百万行压力用例单列。
离线演示通过 `scripts/demo.sh`，产物仍标注 fixture，不冒充真实市场数据。
本轮为研究 CLI，桌面/390×844、网页跨端和浏览器验收 not applicable。

## 失败关闭与真实数据边界

必须覆盖：重复或错位候选轴、未来公告可见日期、未成熟标签、NaN/Inf、float32 溢出、
错误单位、分页重复/截断、缺字段/错日期、原始页及派生源篡改、来源中途修订、gzip
历史文件、输入/输出目录复用、回放越界选股、CPU/cgroup/内存边界和训练真实进度。
空 ST 名单只有显式成功日期分区才可转为零；缺日保持未知。

每次训练实时检查 affinity、各级 cgroup、内存、负载及 GPU 信息；用同一实际输入短测
线程数。HGB 使用 CPU。先保存原生模型和完整分数，再由操作者决定后续实验，不强制
每组执行全量便携校验。成功以阶段清单为准，心跳不冒充完成百分比。

可选真实链路由操作者使用自己的私有凭据执行 `collect --live`。采集保留原始响应和
观察时间，不把供应商当前提供的历史版本认证为历史 PIT。原文或关系修订必须形成新
冻结数据版本。比较原研究时必须从其 sources/protocol 清单核对全部分区、字节 SHA
及行数，不能仅凭缓存目录名挑选文件。计划测试明确覆盖最后一个可见财务报告期。

本轮无真实行情请求、飞书通知、发布、券商连接、交易或调度注册。研究 CLI 不提供
这些副作用入口。GPU 框架及驱动尚未验证；没有新增或宣称任何收益改进。

## 验收记录与证据

证据目录：`~/stockops-exports/research-portability-validation/`，不提交市场数据。

- 首节点移入计算器，146 项原始测试通过。代码来源为筹码
  `f578550eab95e28d493eb7b0c58c4afcb368aefb`、预告
  `662ccd39b3c397467dc0245daed9819acaaf39e7`、财务
  `2f8fa58f2c42b1228eab363259ae21ba3775611e`。
- GitHub 独立 checkout `072dc9ceb6e7510be3e9a3775a8c1f0d15d7f83b`：100 项研究
  用例通过。后续 gzip 实现 `faaff711ec5866c2b1bcfd2e71dbafa854e22857` 重新冻结安装，
  13 项受影响用例通过；安装源码与提交逐字节一致。见 `clean-final-verification.json`。
- 冻结训练包独立导入：520690×47、490655 个成熟训练行，13 个数组 SHA 全部一致。
  见 `clean-import-parity.json`。原 103MB ZIP 不含完整回放行情。
- 重建的 20 个信号、9 个辅助矩阵、候选轴、旧评分和基础 36 列与原归档逐项一致；
  520690×4 独立交易标签也全部一致。见 `rebuild-parity.json`、`labels-parity.json`。
- gzip 失败后保留已验证中间产物，后续调用公开库函数继续，避免重算前段。首次附加
  特征比对发现 9 个差值：验收源清单漏了原训练 20260930 的财务指标 131 行和利润表
  144 行补充分区。按原 sources 清单及 SHA 补齐后单独重算财务三列；不手改特征数值。
  见 `financial-source-correction.json`。完整最新入口由合成端到端用例验证。
- 原生旧分数的独立回放，正常/双成本共18份CSV与原归档逐字节一致。比较配置为
  100万元/原范围/Top15/2%/开盘已有现金/1%历史成交额限额，非生产12万元分档配置。
  见 `replay-parity.json`。
- 全仓 Ruff、mypy（225 个源码文件）和独立环境离线演示通过。新入口与回放输入
  1172 条语句的定向覆盖率100%；全仓覆盖另行验收，不以定向覆盖代替。

最终归档核对通过：完整520690×47矩阵、490655成熟行、全部标签/权重/行轴共13个
数组逐项一致，最大数值差为0；41份回放输入SHA全部一致。见
`training-rebuild-parity.json`、`replay-input-rebuild-parity.json`。既有18份两成本
回放CSV已逐字节核对，不对相同输入与实现重复运行回放。

全仓门禁通过：5948 passed、3 deselected，耗时4826.17秒；3项为既有CI规则
排除的资源压力测试。全仓运行后，补充订单过期、开盘不可成交、入场过滤和行业缺失
四种取消/等待原因测试，确认不伪造成交、临时开盘阻断仅允许后续可成交日买入。
相关证据/核账18项、换仓及训练边界45项全部通过。

覆盖率汇总只使用最终源码对应的行号：移除运行期间变更模块的旧记录，分别使用
最终gzip重建测试、核账/执行证据/换仓/训练定向测试刷新。合并后225个源码文件、
20804条语句全部覆盖，`coverage report --fail-under=100`退出0；缺失语句为0。
原有8条排除语句未增加。证据为`pytest-all.log`、`coverage-final.json`、
`merge-coverage.py`和`final-gates.json`。

已同步GitHub主分支`0d0781b3766ca79b09a5b4e0d52b6ae662f23ac4`新增的凭据说明，
保留统一的`data/secrets/tushare.env`路径。训练包103398538字节、回放包895200806
字节，完整SHA与跨机器使用步骤见独立研究机指南。PR4通过门禁后以保留提交的方式
集成到GitHub研究主线，Codebase及StockOps App不属于本次同步范围。
纯文档与计划断言更新不改变已验证实现；相关节点的浏览器验收 not applicable。
