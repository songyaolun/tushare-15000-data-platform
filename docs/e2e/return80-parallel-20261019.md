# 12万元主板并行实验验收

本轮是离线研究，不改变StockOps生产模型、页面或通知。研究基础提交
`73d478078d744b6662798d36933d954cc3afa6ca`；产品App主分支
`b28df2be1aa99ae28f9454be9816aad66664e5c3`，领域锁定提交
`31f0cad07d1c9b21872908abc0bcd4d473132cd5`用于核对主板筛选口径。
本分支仅扩展显式本金的独立CSV审计及主板证券轴掩码；无产品依赖更新。

## 统一实验契约

- 原36列、原成熟标签、原模型超参数及全五年拟合范围保持；不划金融验证/测试集。
- 账户从2021-10-18以120000元空仓启动，截至2026-10-16。只做沪深主板；
  先限制完整候选池，再按模型分及一级行业每日一只取Top10，共同资格失败不补位。
- 市场广度及特征保留全市场口径，单股5%、行业40%、弱市新买上限15%，正常市场100%。
- 止盈4.5%、止损8%、最长30交易日、次日开盘及5%追价上限；原配对替换规则不改。
- 正常成本沿用最低5元/万三佣金、历史过户费与印花税、单边5bp滑点；压力情景
  把完整手续费与滑点翻倍，整手和预算重新计算。该费率是模型假设，不声称用户券商费率。
- 各组输出相同1643日×5451证券轴评分；缺失特征保留NaN，但可选候选必须有有效预测。
  历史不足另做同日期、同可用候选掩码的基准对照，不能与完整五年年化直接比较。

## 离线与真实数据边界

Python3.11、uv。软件测试不联网、不用Token：

```bash
uv run --frozen pytest tests/test_discovery_experiment_scope.py tests/test_discovery_audit.py -q
uv run --frozen ruff check src/tushare_entitlement_hub/strategy_discovery/audit.py src/tushare_entitlement_hub/strategy_discovery/experiment_scope.py tests/test_discovery_experiment_scope.py tests/test_discovery_audit.py
```

资金复用原始历史缓存。筹码、融资与业绩仅在各自研究目录通过已有HTTPS适配器有界取数，
原始包络先归档，凭据不写入命令、异常或manifest。标记供应商重建历史与不可变时点归档
的区别；只有公告日不足以证明19:30可用，资金、价格、融资资格、修订时间不得猜测或补零。
不发飞书、不部署、不注册调度、不连接账户。无UI变更，桌面/手机浏览器验收为
`not applicable`；跨端一致性由不修改任何产品快照及资产保证。

## 证据与通过标准

证据根为`~/.cache/tushare-hub/stockops-parallel-20261019/`。统一`replay.py`记录
评分、原上下文、行业选择脚本及自身SHA256，输出逐日权益、逐笔交易、候选与费用。
该脚本复用原研究缓存`stockops-candidate-allocation-v2-20261019/run.py`与
`stockops-sector-cap-20261019/run.py`，路径和指纹随protocol记录，不依赖PYTHONPATH。
所有12万元账户调用`audit_result(..., initial_capital=120000)`独立复算CSV，
首日本金错用100万元必须失败；无穷、NaN或非正本金拒绝。
股票代码检查应与产品scope-v1-c0-s0一致，证券选择不得漏出主板范围。
报告正常/双倍成本年化、回撤、胜率、成交数、逐年收益与覆盖限制；保留失败方向。
没有历史时点证据的方向可以报告不可识别，不得制造收益结果或修改生产规则。

可复用入口已保存为`scripts/replay_return80_ablation.py`，显式传入缓存根：

```bash
OPENBLAS_NUM_THREADS=1 uv run --frozen python scripts/replay_return80_ablation.py \
  --cache-root "$HOME/.cache/tushare-hub" --output /new/empty/output
```

每个新增评分使用`--scores /path/to/scores.npy --name moneyflow4`等参数。
源码入口与首次缓存脚本只在缓存根传参方式不同，首次产物保留原脚本SHA，不覆写证据。

本节点实测：Python3.11下83项离线测试通过，两个源码模块Ruff/Mypy通过，脚本Ruff通过。
统一基准两成本各1212日通过独立CSV审计：正常成本年化77.9019747632%、最大回撤9.1648070221%、
胜率71.7333333333%、1875笔平仓；双倍成本年化70.0902074832%、最大回撤10.2864483485%、
胜率71.3516424340%、1857笔平仓。全部候选符合主板范围，且仍是全五年样本内结果。

## 最终比较节点

适用执行入口提交：`f557e85a840111bbfbb969b203cfbe10b2f03b8b`。最终节点只增加研究报告、
数据共享索引说明和本验收记录，不改变可执行行为；软件测试不重复运行，UI验收仍为
`not applicable`。各单组源码已分别完成离线门禁：资金11项、筹码14项、业绩31项、
融资18项；组合拼接13项。各新增特征模块达到100%行覆盖，不等同于全仓100%覆盖。

原基准、四单组、一次组合各两成本，再加两模型的同业绩已知候选对照，共16个账户，
每个1212日，均通过显式12万元独立CSV审计。全部五个新模型的520690条候选便携预测
与训练器输出逐值一致，误差0。未额外训练组合或搜索窗口。

独立Agent进一步复算资金与组合、业绩及可用性对照、筹码与融资账户，并重选1212日
主板及行业cap候选。证据分别为`moneyflow/review.json`、`moneyflow-earnings/review.json`、
`earnings/independent-csv-audit.json`、`earnings/peer-account-review.json`；最后四账户
逐笔收益误差0，年化/回撤/逐年指标最大差8.88e-16。
筹码/融资时点、单位、上市日期和固定原36列的独立限定复核见
`earnings/peer-feature-review.json`，无阻塞问题，不据此认证供应商严格历史PIT。

`shared-inputs.json`把原参考库、原资金缓存、本轮补充历史及各组来源审计连起来；
2513项索引证据/分区路径已检查存在，所引用审计文件SHA已复核。新增raw未导入正式
accepted库。完整模型和评分SHA、研究分支及MR见`docs/stockops-parallel-20261019.md`。
生产App保持main干净，CodeGraph索引最新；没有更新App锁文件或生产调度。
