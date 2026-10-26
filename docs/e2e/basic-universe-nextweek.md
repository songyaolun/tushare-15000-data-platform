# 基础资格周度样本准备验收

适用revision为本文件所在提交，开发前hub基线为
`072dc9ceb6e7510be3e9a3775a8c1f0d15d7f83b`。独立资格模块只依赖numpy/pandas及本仓
原始记录解析；使用锁文件声明的运行环境。跨仓整体E2E由stockops-app的
`docs/e2e/basic-universe-nextweek.md`管理，最终必须固定已推送hub/research SHA，
不可依赖本地PYTHONPATH；真实运行和独立安装的来源SHA写产物。

## 输入与离线链路

显式提供panel、raw历史根及可选身份公告证据；无默认机器路径、无网络、无训练。
raw包含完整L/D/P证券主表、官方交易日历和全市场逐日ST回执。验证每页SHA、API、
公开参数、日期、主键、行数与分页完备性。证券主表中轴外证券逐条分类；目标期
可能合格的证券不在panel时失败关闭。

资格仅看当时上市超过250自然日、未退市、非ST、排除科创/北交、原始价>1元及
自然行情20根均成交额>3000万元。正面信号、watch、ATR、行业、技术风险不参与筛选。
保留每日轴和逐板块分母，另投影每个交易周末样本；未知基础资料不伪装合格，
真实缺口、自然窗口不足、重复身份分开记录。

```bash
uv run --frozen pytest tests/test_return80_basic_universe.py -q
uv run --frozen ruff check src/tushare_entitlement_hub/return80_research/basic_universe.py tests/test_return80_basic_universe.py
uv run --frozen mypy src/tushare_entitlement_hub/return80_research/basic_universe.py --follow-imports=skip
git diff --check
codegraph sync .
codegraph status .
```

fixture覆盖阈值、ST缺页、源SHA/分页/日期冲突、缺证券、完整周末投影、历史退市、
无信号资格保留和别名双计拒绝。wheel环境从direct_url取来源commit，禁止借cwd的
Git冒充生产者revision；无Git的独立fixture必须通过。

## 身份与真实数据验收

证券更码仅在官方同一上市主体证据与重叠六字段行情完全相等时消除旧代码重复，
同步映射历史ST。保留原始文件和别名声明，不覆盖raw。
本次300114→302132有深交所2025-028公告及1228根完全一致OHLC/成交额/复权证据，
排除旧别名不减少唯一证券。002260退市整理30根的前19根自然窗口不足保留说明。

通过要求：全量证券/ST来源验证、每板块完整分母、每日/周末数组有序唯一、未知主表
缺口解决、所有真实行情缺口有代码与日期、不将构建complete误当全行情覆盖认证。
实际新候选4567522行、周末962118行/255周；该数值仅作为本机当前输入的验收事实。

## 外部边界与证据

默认只读缓存与离线fixture。真实补采、模型训练、生产更新另走根计划授权阶段；
本入口无飞书/Cloudflare/交易/调度副作用。桌面与390x844浏览器、跨端UI一致性
not applicable，本次无UI改动；跨机器通过文件SHA和独立冻结安装验证，未作Mac实测。

证据：coverage.json、sources.json、daily-board-coverage.csv、gap_indices.npz、
master-axis-exclusions.json、身份公告及全部数组SHA。最终从已推送SHA安装到独立
环境后输出新目录重跑，并与原候选轴逐文件哈希核对，不覆盖首次运行证据。

## 原47列与风险字段重建

本节适用于同提交的basic_features模块，依赖stockops-domain精确revision
`6d6313634734fa7d06cd3414a169fa61e0adb4d9`；先安装本仓uv.lock固定环境，
再使用冻结panel、完整周末候选轴、经身份校验的000300.SH信号和财务manifest。
默认无凭据、离线运行：

```bash
uv run --frozen --extra rebuild pytest tests/test_return80_basic_features.py -q
uv run --frozen ruff check src/tushare_entitlement_hub/return80_research/basic_features.py tests/test_return80_basic_features.py
uv run --frozen mypy src/tushare_entitlement_hub/return80_research/basic_features.py --follow-imports=skip
```

离线fixture覆盖49列顺序、无信号候选保留、公告严格下一交易日生效、NaN保留、
风险OR布尔与unknown、错误API/字段/日期/SHA/行数/offset/has_more失败关闭。
signals_proof必须来自实际执行complete.json；核验原始benchmark的index_daily身份、
000300.SH逐行代码、日期覆盖、实际读取SHA、全部输出数组SHA及冻结源码快照。
仅手写指数标签不能通过，旧数组比较不一致不等于新执行失败。
额外覆盖真实cap静默截断、跨页部分重叠全字段重复拒绝，以及值或update_flag不同
版本原样保留。forecast保守cap3500，income实测9000，fina实测12000；请求limit
大于cap且返回触cap时不可认作末页。分页闭合不冒充全市场完整或历史PIT认证。

真实链路只读现有缓存；需要补采时须显式授权并保留原文。forecast全27报告期采用
低于保守cap的单页证据与10股跨年交叉验证；income/fina触cap期采用2000/6000
双布局经济多重集/顺序/重复审计，未触cap期保留原源。所有期、原始文件SHA和
覆盖分母写回执，不因无预告事件删证券。新版写新baseline-v2目录，旧版保留。

验收须逐列输出全轴新旧差及旧训练共同轴差：分别归因000985误作000300的基准
错误、forecast分页漏源补回和无法重建旧panel时未确证来源的残留小数值差，不能
宣称新旧47列完全同义或完全一致。保存manifest/coverage/complete、源码快照、
依赖SHA、源差逐条记录、差异阈值分布；不得训练、发布、发消息或修改暂停任务。
本次无UI行为，桌面/390x844浏览器与跨端UI一致性not applicable；文件跨机器
一致性须根计划的无PYTHONPATH冻结安装门禁，未在Mac运行不得声明Mac实测。
