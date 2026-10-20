# 在独立机器上重建 Return80 的 47 列研究数据

本文面向只能访问 GitHub、使用自己 Tushare 凭据的机器。核验基于开发分支
`feat/return80-annual250-astra` 的 `3dcb683b336dae12fab7d3c3f8f801db0feab18a`。
本轮只读取既有离线数据，没有执行下文的联网采集，也没有在另一台机器完成全链路验收。

**当前分支具备通用原始数据采集、研究特征库和冻结矩阵训练入口，但尚不具备从空目录
生成本轮全部 47 列、标签和收益账户的一键入口。** 部分生产者仍在私有实验脚本或其他
研究分支中。下文给出实际命令、逐列公式、数据范围及缺口；不能把本机缓存路径当成
远程机器能够下载的文件。GPU 不改变数据口径；本轮使用 CPU 的 sklearn HGB，未实现
或验证 GPU 训练替代方案。

## 获取代码与准备环境

```bash
git clone --branch feat/return80-annual250-astra \
  https://github.com/songyaolun/tushare-15000-data-platform.git
cd tushare-15000-data-platform
git rev-parse HEAD
env -u PYTHONPATH uv sync --frozen --python 3.11
env -u PYTHONPATH uv run --frozen tushare-hub --help
env -u PYTHONPATH uv run --frozen tushare-hub reference build --help
env -u PYTHONPATH uv run --frozen tushare-hub ingest plan --help
```

交付时须核对 GitHub 与 Codebase 同一分支的精确 SHA；不以分支名代替版本固定。
`uv.lock` 固定项目依赖，但当前模型还要求 `scikit-learn==1.9.1`，它不在项目基础依赖中。
模型命令须显式加 `--with scikit-learn==1.9.1`；离线执行再加 `--offline`，前提是该机器
已经下载对应包。新机器不能假定本机 uv 缓存存在。清除 `PYTHONPATH`，不要用另一份
源码覆盖已安装模块来掩盖缺依赖。

Tushare 凭据放在仓库外、权限为 `0600` 的 dotenv 文件，内容键为 `TUSHARE_TOKEN`。
命令只传文件路径；读取实现为 `src/tushare_entitlement_hub/credentials.py`。
使用自己的实际接口权限和频率限制；15000 积分或目录中存在接口不等于所有接口均获授权。

## 原始下载入口及其边界

每次新训练先实时检查CPU affinity、cgroup各级配额、可用内存和已有负载，再确认GPU
型号/显存以及所用算法是否支持GPU；GPU存在不等于sklearn HGB能够使用它。记录
OMP/BLAS/MKL设置，用同一输入做有界短测并核对数值一致性，再选择模型并行数和每模型
线程数。本次3×16只适用于当前64核机器的实测，不是跨机器固定配置。Linux可用
`lscpu`、`taskset -pc $$`、`cat /proc/self/cgroup`、`free -h`、`ps`查看资源；沿cgroup
所属路径向上核对`cpu.max`，不能只看宿主机核心数。GPU驱动可用时以`nvidia-smi`核验。
已运行任务改变设置前先评估重启成本，不为了补日志重新训练已完成拟合的模型。

新训练默认输出框架迭代和fit/native_predict/portable_verify阶段，长阶段30秒心跳；
`--training-verbose 0`只关框架逐轮信息，`--quiet-progress`关闭新增细节。当前已加载的
旧任务不能热加轮数日志，只能报告真实进程观测和上一次经证据确认的阶段。研究观察入口：

```bash
env -u PYTHONPATH uv run --frozen --python 3.11 python \
  scripts/research/watch_return80_progress.py --root /你的研究产物目录 --pid 你的进程号
```

该观察器适用本次四模型32账户目录协议；只读转发日志及已保存的完成数，不下载数据，
不启动任务，不以CPU活跃或elapsed冒充完成百分比。实际运行请把目录和PID替换为本机值。

研究默认`--portable-validation bounded`：回放使用完整native scores，便携检查最多
512行，包含首尾、各列有限值最小/最大值、首个/最后NaN及均匀代表行。预算不足以容纳
边界时拒绝，不丢边界；检查范围与行索引SHA写入审计。原生模型/全预测/评分矩阵先落盘，
检查点哈希写完后才进行便携导出。`--portable-validation full`仅在选定交付模型或
导出器变更时显式使用；有界检查通过不等于完整portable认证。本轮已启动的旧进程没有
安全中途导出接口，无法靠修改磁盘代码跳过已进入的循环，不能直接kill后重新训练。

先导入仓库提供的公开接口目录，再查看实际登记结果：

```bash
mkdir -p "$HOME/return80-rebuild"
env -u PYTHONPATH uv run --frozen tushare-hub \
  --db "$HOME/return80-rebuild/hub.duckdb" \
  catalog sync --fixture fixtures/catalog/official
env -u PYTHONPATH uv run --frozen tushare-hub \
  --db "$HOME/return80-rebuild/hub.duckdb" catalog list
```

目录 fixture 不等于市场数据。若目标接口未登记，须先核验目录政策；公开文档更新入口是
`catalog sync --online`，它会联网。下面是**明确范围的真实采集命令示例**，仅下载两个
交易日的 `daily`，不声称已覆盖五年全市场：

```bash
env -u PYTHONPATH uv run --frozen tushare-hub \
  --db "$HOME/return80-rebuild/hub.duckdb" \
  ingest plan daily 2026-10-15 2026-10-16 --live \
  --output "$HOME/return80-rebuild/daily-plan.json"
env -u PYTHONPATH uv run --frozen tushare-hub \
  --db "$HOME/return80-rebuild/hub.duckdb" \
  ingest run "$HOME/return80-rebuild/daily-plan.json" --live \
  --credential-file "$HOME/.config/tushare/credentials.env" \
  --raw-directory "$HOME/return80-rebuild/raw"
```

`reference build` 也可按接口声明全部分区，生成通用原始页、Parquet 和 DuckDB。
参数文件为 `{ "接口名": {"参数": "值"} }`，同一接口多个分区可用对象数组。
例如 `{"stock_basic":[{"list_status":"L"},{"list_status":"D"},{"list_status":"P"}]}`
声明三个证券状态；完整性仍须按返回分页、状态和字段核验。已有目录与参数文件后执行：

```bash
env -u PYTHONPATH uv run --frozen tushare-hub \
  --db "$HOME/return80-rebuild/hub.duckdb" \
  reference build --live --apis stock_basic \
  --credential-file "$HOME/.config/tushare/credentials.env" \
  --parameters "$HOME/return80-rebuild/stock-basic-parameters.json" \
  --output "$HOME/return80-rebuild/reference"
env -u PYTHONPATH uv run --frozen tushare-hub \
  reference rebuild --output "$HOME/return80-rebuild/reference"
```

最后一条为离线回建。通用采集器不会自动把这些目录转换成本轮 `history.duckdb`、
`panel.json` 或候选矩阵。`strategy-optimize collect-history` 要求现成交易日历，
`regime-research collect` 要求现成 `dataset.json`；`signal-research run` 读取已认证的
`history_root`，不是空目录下载器。`strategy-discovery prepare` 生成通用研究特征，
也不直接生成本轮 36 列或 47 列。上述入口的事实来源为对应包的 `cli.py`。

## 原始数据范围与完整性

共同面板覆盖 `20200102–20261016` 的 1643 个交易日。候选日期为
`20211018–20261016`，共 520690 行；训练为其中 490655 行，最后信号日 `20260826`。
至少保留 250 个交易日预热，不把 2021 年候选起点当作全部行情下载起点。

| 来源接口 | 最小消费字段及单位 | 原研究范围、分区与用途 |
|---|---|---|
| `trade_cal` | exchange、cal_date、is_open、pretrade_date | 完整市场交易日历；不能按某一股票有报价的日期造日历 |
| `stock_basic` | ts_code、list_status、list_date、delist_date | 完整 L/D/P 状态与证券轴；筹码审计现有 L/D 共 5911 代码，P 状态完整性未认证 |
| `daily` | ts_code、trade_date、OHLC、pre_close、vol（手）、amount（千元）、pct_chg | 20200102–20261016，按交易日取全市场并分页；补充矩阵转成交量为股、成交额为元 |
| `adj_factor`、`stk_limit` | 日期/代码、adj_factor、up_limit、down_limit | 同期逐日；成交用原始价与当日边界，技术价另算复权 |
| `suspend_d` | 日期/代码、停牌时间与类型等停牌证据 | 全市场全天停牌与部分停牌须区分；缺价不能直接当停牌 |
| `daily_basic` | 日期/代码、close、volume_ratio、total_share（万股）、total_mv/circ_mv（万元） | 财务来源清单有 1643 个日分区、8081761 行、页限额 6000；本轮估值使用同日总股本 |
| `moneyflow` | buy_lg_amount、sell_lg_amount、buy_elg_amount、sell_elg_amount（万元） | 20210901–20261016 按日全市场，原补充采集页限额 9000；主力净额为四项相减后乘 10000 |
| `stock_st` | ts_code、trade_date | 20211018–20261016 全市场名单；只有完整日期响应才能把未出现证券标为非 ST |
| `index_daily` | 日期、OHLC、vol | 000300.SH 为技术相对强度基准；面板另需 000985.CSI/H00985.CSI 市场与收益基准 |
| `index_classify`、`index_member_all`、`sw_daily` | 分类代码/层级、ts_code、l1/l2_code、in_date、out_date、指数日线 | SW2021，成员当前/历史 is_new=Y/N 均取；面板行业与信号的 L2 归属不可混用 |
| `dividend` | 实施状态、登记/除权/派息日期、现金及送转比例 | 覆盖证券与除权日；账户、标签及公司行动需完整分配与冲突处理，不只查候选当日 |
| `cyq_perf` | 日期/代码、cost_5pct、cost_50pct、cost_95pct、weight_avg | 20210901–20261016，逐日全市场；四个成本使用同源单位，不混原始收盘价 |
| `forecast_vip` | ts_code、end_date、ann_date、first_ann_date、p_change_min、p_change_max | 20210630–20260930 共 22 报告期，原缓存 30450 行；按报告期保留版本、重复与冲突 |
| `fina_indicator_vip` | ts_code、ann_date、end_date、dt_netprofit_yoy、netprofit_yoy | 20200331–20260930 共 27 期，95 页、245413 行、页限额 3000 |
| `income_vip` | ts_code、ann_date、f_ann_date、end_date、report_type、comp_type、n_income_attr_p（元） | 同 27 期，80 页、200408 行、页限额 3000；按报告身份生成归母 TTM |

财务来源清单还包含 `cashflow_vip` 的 86 页、218900 行。原 3 列质量构建器读取了它，
但当前 47 列已删除 `cash_profit_backing`，只保留质量前两列；新的最小构建器可以不依赖
现金流，现有私有构建路径仍会读取。不能因历史清单里有现金流就把现金列加回模型。

每次采集保存接口、完整参数、offset/limit、返回行数、观察时间、原始响应 SHA256，
分页至真实结束并检查截断、重复主键及冲突。逐证券日核对上市、退市、停牌、应有行情、
涨跌停和资金字段，再按全部行业分别报告覆盖；缺口保留代码和原因，不删分母凑满。
表中行数是旧快照事实，不是新下载必须硬凑到的目标。

## 47 列公式与顺序

所有列在同一 `(days, securities)` 候选行轴上拼接，最终为 float32；保留 NaN，拒绝 Inf。
二元信号命中为 1，未命中为 0；历史不足或缺必需输入不能冒充有效命中。
模型不包含代码、日期、行业 ID 或未来标签作为特征。

有两套 ATR，重建时必须区分。通用技术特征使用有效行情序列上的 TA-Lib ATR14
（Wilder 递推），价格为 `raw_price × adj_factor / 首个有效因子`。旧排序风险输入
使用 `raw_price × adj_factor`，`ATR_simple14=mean(TR,14)`，
`TR=max(H−L,abs(H−C_prev),abs(L−C_prev))`；结构止损为 `min(L,20)−0.2×ATR_simple14`。
缩放不影响比例，但简单平均与 Wilder ATR 不等价。

| 序号 | 列名 | 公式及窗口 |
|---|---|---|
| 1 | signal_group_score | `(旧总分−0.2×liquidity_score−0.1×risk_efficiency_score)/0.7`；旧总分已舍入到 0.01，反解误差上界 `0.005/0.7` |
| 2 | liquidity_score | `100×clip(log(max(ADV20,1)/20000000)/log(25),0,1)`；ADV20 是含当日的完整 20 市场日成交额均值，单位元 |
| 3 | risk_efficiency_score | `100×min(1,ATR_simple14/(reference−stop))`；ATR、stop 为正且 reference>stop，否则为 0 |
| 4–6 | return1、return5、return20 | `C/C_lagN−1`；有效行情观测上的 1/5/20 根历史 |
| 7 | extension_atr | `(C−SMA20)/ATR_Wilder14` |
| 8 | close_location | `(C−L)/(H−L)` |
| 9 | boll_position | `(C−SMA20+2×std20)/(4×std20)`；总体标准差 ddof=0 |
| 10 | kdj_j | 9 日 RSV，K/D 初值 50；`K=2K_prev/3+RSV/3`，`D=2D_prev/3+K/3`，`J=3K−2D`；不裁剪 J，未知 RSV 重置递推 |
| 11 | atr_fraction | 旧风险输入 `ATR_simple14/reference`，不是通用技术 ATR |
| 12 | above_ma20_age | 当前市场日索引减最近一次 `C<=SMA20` 的市场日索引；无锚点为 NaN |
| 13 | sector_breadth | 当日行业有效成员中 `C>SMA20` 的比例 |
| 14 | sector_diffusion | 行业广度减 5 个市场日前同一行业广度 |
| 15 | sector_momentum20 | `expm1(sum(log1p(行业日均收益),20))`，须完整 20 市场日 |
| 16 | relative_return20 | 个股 return20 减当日所属行业 sector_momentum20 |

安全除法只接纳正有限分母；技术缺失保持 NaN。停牌不参与技术指标递推，映射到市场日
时不填价格。行业广度与收益各自要求至少 3 个有效成员且覆盖当日已知成员的 60%；收益
只接纳连续两个市场日可交易的证券。行业成员变化按当日归属计算，不用最终成员回填。
实现见 `strategy_optimization/features.py` 与 `strategy_discovery/sector.py`。

下表序号 17–36 的列名均带 `signal:` 前缀。记 C/O/H/L 为旧信号复权价格，V 为股数，
A 为元成交额，M 为上述主力净额，VR 为 `daily_basic.volume_ratio`，B 为沪深300收盘。
未另写时滚动窗口含当日，lag/shift 先移位再滚动；窗口以自然有效行情行计算。
“完整 N”另要求市场日历上连续 N 个槽位都有收盘与成交量。

| 序号 | 信号名 | 命中条件（全部同时满足） |
|---|---|---|
| 17 | volatility_contraction_breakout | C>前20根H最大值；V/前20根V均值≥1.5；(近20根H最大−L最小)/C≤0.2；完整60 |
| 18 | platform_breakout | C>前20根H最大值；V/前20根V均值≥1.2；完整60 |
| 19 | strong_first_pullback | C/C_prev−1≥0.01；C>SMA20、SMA60；3根收益在[−0.06,0.03]；完整120 |
| 20 | panic_gap_recovery | O≤0.97×C_prev 且 O/C_prev−1≤−0.03；C≥C_prev；完整21 |
| 21 | liquidity_transition | A近5均值≥1.5×前20均值；q=A近5均值/shift5后的20均值≥1.5，q_prev<1.5；V近5均值/shift5后的20均值≥1.3；abs(5根收益)≤0.08 |
| 22 | relative_strength_continuation | C≥0.95×近60根H最大及C最大；C/C_lag20−B/B_lag20≥0.05；完整60 |
| 23 | long_lower_shadow_support | min(O,C)−L>0.04×C；VR>1.5；C<0.95×C_lag5 |
| 24 | oversold_rebound | RSI14<30；C<0.9×C_lag5；C>1.01×C_prev |
| 25 | capital_oversold_accumulation | C<0.9×C_lag5；sum(M,5)>0.06×sum(A,5)，sum(A,5)>0；5根中至少4根M>0 |
| 26 | limit_up_pullback | 前根原始H触及涨停；前根连板段有已知非涨停锚点、连板数≤5；原始pre_close>0；原始C<0.93×pre_close |
| 27 | limit_up_first_board | 当根收于涨停、连板数1且有已知非涨停锚点；近20根涨停≤5；非一字板 |
| 28 | limit_up_relay | 当根收于涨停、连板数2且有已知非涨停锚点；非一字板 |
| 29 | low_volume_oversold | C<0.9×C_lag5；V<0.7×近20根V均值；RSI14<30 |
| 30 | capital_price_divergence | C<0.95×C_lag3；sum(M,3)>0.05×sum(A,3)，sum(A,3)>0 |
| 31 | price_volume_acceleration | C>1.08×C_lag5；VR>1.2；C>SMA20；C<1.25×C_lag20 |
| 32 | spring_sos | C≥前9根C最大、C>SMA20、V>1.2×前20根V均值；前10根存在L<0.99×前40根L最小且C收回该低点的spring；当根C>前10根H最大 |
| 33 | sector_heat_leader | 所属L2行业为热行业；个股5根收益降序min秩≤ceil(行业人数×0.19) |
| 34 | sector_limit_up_wave | 所属L2行业为涨停潮行业；原始C>0；个股1根收益降序min秩≤ceil(行业人数×0.20) |
| 35 | sector_heat_laggard | 所属L2行业为热行业；个股5根收益升序min秩≤ceil(行业人数×0.20) |
| 36 | relative_resilience_3pp | 个股5根收益<0，且比同日期对齐的沪深300高出严格超过0.03 |

信号 RSI14 使用涨/跌幅各自 `ewm(alpha=1/14,adjust=False,min_periods=14)` 后的
`100×gain/(gain+loss)`。收于涨停定义为原始 C≥up_limit−0.005 且 C、up_limit>0；
触及定义为原始 H≥up_limit；一字板是原始 O=H=L=C。行业信号只比较成员数>50的 L2
行业；热行业为成员5根收益均值降序 min 秩≤ceil(行业数×0.09)，涨停潮为涨停家数同样
前9%且至少3家。成员区间为 `in_date<=T<out_date`，空 out_date 仍在；并列不强拆。
它们与第13–16列的行业聚合口径分开保存，不能相互替换。

上述信号的实际生产者为私有 `stockops-pit-validation-20261018/batch.py` 的
`stock_signals` 及 `stockops-pit-redesign-20261018/export_candidates.py` 的
`sector_hits`；Spring、抗跌还调用 StockOps 的 `local_strategies.py`。当前开发分支
没有这些完整生产者，表格不应被误读为已发布实现。

| 序号 | 列名 | 公式、可见时间与缺失 |
|---|---|---|
| 37 | chip_median_weighted_offset | `(cost_50pct−weight_avg)/weight_avg` |
| 38 | chip_width90 | `(cost_95pct−cost_5pct)/cost_50pct` |
| 39 | chip_width90_change5 | width90(t)−width90(t−5市场日)；含端点6日均有效才输出 |
| 40 | forecast_growth_midpoint | `(p_change_min+p_change_max)/200` |
| 41 | forecast_growth_interval_width | `(p_change_max−p_change_min)/100` |
| 42 | forecast_growth_revision | 同报告期当前版本 midpoint 减前版本 midpoint；无前版本 NaN |
| 43 | forecast_age_sessions | `log1p(t−E)`；E为可见公告日严格之后的首交易日，并非未变换的日数 |
| 44 | forecast_post_event_excess | 从E至t累计 `log1p(个股日收益)−log1p(剔除自身的同行简单平均日收益)` |
| 45 | deducted_profit_growth | `dt_netprofit_yoy/100` |
| 46 | deducted_growth_gap | `(dt_netprofit_yoy−netprofit_yoy)/100` |
| 47 | earnings_to_price | 可见归母TTM利润元 / (当日未复权close元×total_share万股×10000) |

筹码四成本必须有限、为正，且 cost5≤cost50≤cost95；上市前来源行拒绝。
公告以 `max(ann_date,first_ann_date)` 为准，严格晚于公告日的交易日19:30才可见；
最新可见报告期优先，再取该报告期可见版本，超过180自然日过期。重复合并，冲突不能
任意择一。公告反应使用前一日行业与资格、当日和前日正有限价格，至少5个合格同行且
不含自身；任意中间路径缺失使反应为NaN但保留年龄。公告早于日历起点时两列均未知。

财务取 `max(ann_date,f_ann_date)` 之后首交易日19:30可见；指标源没有 f_ann_date 时
依现有有效 ann_date。按报告身份、版本及公司类型处理，TTM为“本期累计+上年全年−
上年同期”，全年直接取年度值。最新报告缺组件时不退回旧报告；不兼容报表或公司类型
不能拼接。利润可负或为零；市值未知、非正、单位校验或重复值冲突时为NaN。

## 候选行、标签和权重

本轮候选轴来自原 `eligible & (technical_risk==0)` 的 `np.where`，按日期优先、证券列
其次，不在拼接特征时重新排序。旧资格包含上市超过250**自然日**、未退市、非科创板与
北交所、原始价>1元、自然行情20根均额>3000万元、非ST、至少一项正面信号、单因子
观察规则及有限正风险输入。这是旧实验的精确口径，与250交易日预热不是同一条件。
旧评分还依赖冻结 `strategies.yaml` 的七证据组、同类簇最大值、组内[0.75,0.25]及
跨组[0.65,0.25,0.10]权重、各信号权重/可靠性，以及最后总分的舍入。

技术风险剔除包括最近5根出现单日≤−9.5%、5根≤−8%、较20/60根最高收盘回撤
≤−15%/−25%、20根收益样本标准差×sqrt(252)≥0.60、20根成交额中位数<3000万元、
5/20根均额≤0.4、近5根存在零成交量。候选轴的 `common` 另外记录 valid、beta、
capitalization、已知行业、当日原始amount≥20000千元及涨跌停带宽>0.07；不直接从轴
删除 common 不合格行，而由标签状态记录。完整 panel 的 beta/市值与市场状态也属于
标签和账户复现依赖，不能仅凭本页47列跳过。

`long-labels.npy` 为四列：净收益、持有交易日数、退出日索引、状态。状态0不合格、
1未成交、2右删失、3已平仓。标签用独立50000元、4%止盈/8%止损/30日、追价≤5%，
含费用与分红。最终共享现金账户止盈为4.5%，两者不能混称同一收益。

训练选择状态3、信号日≥20211018、索引<日历长度−31，并要求实际退出日≤20261016。
每行原权重为该信号日训练行数的倒数；半衰期H时再乘
`2**(-(截止日索引−信号日索引)/H)`，最后除以全体均值。等权不乘衰减，H只取126/252/504。
全五年样本内拟合；2026空仓重置不使模型变成2026年初可知。

## 当前九份训练输入与尚未发布的生产链

下列均是本机冻结中间产物的相对缓存路径，**不是下载地址**：

| 文件 | 作用 |
|---|---|
| `stockops-return80-model-20261019/features.npy` | 520690×36基础矩阵 |
| `stockops-return80-model-20261019/models/hgb_leaves31_rounds1600.json` | 基础列顺序和模型参数；不是本轮最终1600轮模型 |
| `stockops-utility-score-20261019/candidate-axis.npz` | days/securities/common固定行轴 |
| `stockops-joint-score-exit-20261019/long-labels.npy` | 长持有独立成交标签 |
| `p03-regime-alpha-20261018/panel/panel.json` | 日历与证券轴等元数据；纯训练无需全套panel二进制 |
| `stockops-chips-earnings-20261019/candidate-chips-earnings.npy` | 3筹码+3预告 |
| `stockops-next-experiments-20261019/event/candidate-features.npy` | 2公告事件 |
| `stockops-cash-quality-audit-20261020/quality-without-cash.npy` | 2质量列 |
| `stockops-next-experiments-20261019/valuation/candidate-features.npy` | 1估值列 |

训练入口为 `scripts/research/train_feature_case.py`，要求已生成且SHA匹配的spec及输入。
spec中的 `candidate_sectors.npy` 只供 `event_placebo` 使用，本轮半衰期案例不读取。
完整回放还需要panel全部 signals/values/market数组、旧eligible/scores/technical_risk/
frozen-inference、分红与distribution-resolutions，以及私有context/sector-cap脚本。
`scripts/replay_return80_ablation.py` 当前仍通过 `runpy` 读取这些私有脚本；九文件不能
替代回放数据。独立核源另外读取认证的 `history.duckdb`。

| 缺口 | 当前事实及补齐验收 |
|---|---|
| 原始数据→认证history→dataset/panel | 通用采集目录与研究消费者之间缺少本轮完整、无本机路径的编排；须发布完整参数/分页/规范化/覆盖审计入口，再从空目录验证 |
| 20信号、旧评分与候选轴 | 上述batch/export/labels脚本及冻结strategies配置未进入本分支；还依赖StockOps模块及其精确revision，不能只复制20个名字 |
| 筹码与预告6列 | `chip_features.py` 在筹码研究checkout，`earnings_features.py`及`research_history/earnings_audit.py`在预告checkout；合并6列的可调用生产者未在当前分支找到 |
| 事件2列 | `forecast_event_reaction.py`与私有event/build.py未进入本分支；原source revision为 `662ccd39b3c397467dc0245daed9819acaaf39e7` |
| 财务与估值3列 | `financial_quality_features.py`、`financial_quality_artifacts.py`及构建包装不在本分支；原source revision为 `2f8fa58f2c42b1228eab363259ae21ba3775611e` |
| 标签与收益回放 | 私有long-label生产者和context/sector-cap编排仍未移植；须发布参数化入口及离线测试，不能把本机绝对路径当安装步骤 |

这些源revision只说明当时材料来源；本文未验证它们均已推送GitHub、依赖可安装或与当前
开发分支相容。正式补齐时先提交依赖实现及测试，再固定已推送SHA，最后在无本机缓存、
无PYTHONPATH环境验收。当前研究输入不因补入口而重抓或改写。

## 重建结果应如何验收

重新下载拿到的是供应商当前提供的历史版本。筹码、公告、财报与成员数据属于
`supplier_reconstructed`，未认证历史PIT；其中预告多数在2026-10-19观察，20260630
报告期复用10月2日快照，20260930期复用10月16日快照。按公告日控制可见性不会自动
恢复当时未修订版本。单股单日probe一致也不能证明全历史不可变。

新机器先保存自己的全量原始快照与来源清单，按全部证券/行业核验覆盖；再核对日历、
证券轴、候选顺序、47列名称、float32/NaN、成熟标签及权重。若与原快照SHA不同，应
列出数据与候选差异并建立新冻结版本，不能通过补零、删行或套用旧SHA冒充逐位复现。
需要精确重现旧收益时，必须有合法取得的原版本原始快照及全部生产代码；本机缓存
并未作为远程可下载包交付。

本页属于命令与数据血缘说明，不新增可执行行为；浏览器/跨端验收不适用。完整实验
门禁见 [近期加权E2E计划](e2e/annual250-recent-weighting-20261020.md)。新机器的联网
下载属于独立执行阶段；本轮没有运行行情API、部署、通知、真实交易或调度注册。
