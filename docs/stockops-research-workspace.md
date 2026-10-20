# StockOps 独立研究机工作流

本指南对应 `feat/return80-research-portability` 开发阶段。当前已实现冻结数据导入、
显式原始采集计划、日频新特征、按日期证券拼接、预测矩阵映射和无私有脚本的收益回放。
原始47列从零重建和训练便捷入口仍在本任务后续节点验收；不能将此阶段称为全部完成。

## 安装与数据

使用GitHub研究仓代码及冻结uv环境：`uv sync --frozen --python 3.11 --extra training`。
训练额外依赖固定 `scikit-learn==1.9.1`；数据命令本身不需要sklearn。
`uv run --frozen return80-research --help` 列出已安装入口，不需要执行仓库外脚本。

已交付103MB ZIP解压后：

```bash
uv run --frozen return80-research import-training --bundle /你的路径/half126-training-20261020 --output /你的路径/research/training
```

导入验证SHA、47列顺序、完整候选行轴、成熟标签和126日权重。输出独立NPY与相对路径
`dataset.json`，支持内存映射；不依赖旧机器HOME、原spec的绝对路径或PYTHONPATH。
已有输出不覆盖。数据集中的X含全部候选，训练用`X[fit_indices]`；y和sample_weight
已经是训练子集，不能再次用fit_indices索引。预测要覆盖全部X并保持行序。

## 新增Tushare数据和特征

采集复用现有HTTPS Client、RawStore、限流、重试和完整分页。传公开JSON计划，不传Token：

```json
{"schema":"return80-collection-plan/v1","requests":[{"api":"daily_basic","parameters":{"trade_date":"20261016"},"fields":["ts_code","trade_date","turnover_rate"],"primary_key":["ts_code","trade_date"],"page_size":3000,"maximum_pages":10}]}
```

这是单日示例；训练用新增列必须列全所需历史分区与预热窗口。财报接口用period并带完整
报告身份主键，不能假设ts_code/end_date就足以区分所有版本。接口是否授权以实际返回为准。

```bash
uv run --frozen return80-research collect --plan plan.json --output raw-study --credential-file /你的私有目录/tushare.env --live
```

默认每分钟60次，可用`--rate-per-minute`调整。无`--live`失败关闭。每分区保留原始响应、
观察时间和哈希，已完成分区只在哈希匹配时恢复；分页重复/错日期/少字段不写成功标记。

日频特征recipe示例：

```json
{"schema":"return80-daily-features/v1","features":[{"name":"turnover_mean5","api":"daily_basic","field":"turnover_rate","scale":0.01,"operation":"mean","window":5,"lag":0,"unit":"fraction","formula_version":"turnover-mean5/v1"}]}
```

```bash
uv run --frozen return80-research build-daily-features --dataset research/training --collection raw-study --recipe recipe.json --output blocks/turnover
uv run --frozen return80-research append-features --dataset research/training --block blocks/turnover/block.json --output research/with-turnover
```

已支持value、mean、std、change、return及市场交易日lag。均值/标准差窗口任一天缺数则未知；
change/return比较端点，return要求正分母。只消费该交易日可用值；若提供available_field，
字段未知或晚于该行情日则关闭该行。公告/财报不能套用日频填充；使用移入的公告/财务事件
计算器控制可见时间，或通过自定义feature block提供明确首次可用日期。

自定义新特征输出CSV列为`trade_date,ts_code,available_date,<列名...>`，旁边block.json
声明`return80-feature-block/v1`、文件SHA、feature_names、source、units、formula_version。
严格按股票与信号日连接；重复、榜外轴、列名冲突及available_date晚于信号日拒绝。缺数
留NaN并报告覆盖，不补零。一个新数据集保留全部原始列、标签及权重。

## 回放新模型

原收益回放数据现在可导出成独立目录：

```bash
uv run --frozen return80-research export-replay --cache-root /原机器缓存 --output replay-data
```

这一步在已有原数据的机器运行一次；将replay-data整体复制到新机器。约1.8GB行情面板
加候选/旧评分/公司行动，清单中固定每个文件SHA。新机器不需要原缓存里的run.py。
当前冻结训练ZIP未包含这个回放目录，需要另行搬运。

训练后保存全部候选预测到predictions.npy，再映射并按原配置回放：

```bash
uv run --frozen return80-research map-predictions --dataset research/with-turnover --predictions predictions.npy --output scores.npy
uv run --frozen return80-research replay --dataset replay-data --scores scores.npy --output accounts/new-model --capital 1000000 --scope original --position-weight .02 --candidate-count 15 --opening-cash-only --maximum-daily-participation .01 --capture-execution-evidence --name comparison
```

这个配置是原half126研究比较基线，不是当前12万元均衡分档生产配置。比较时原模型和新模型
必须使用同一资本、范围和执行规则。命令同时运行正常/双成本，保留交易、费用、逐日净值、
实际成交/取消与核账。全五年样本内拟合/选参，没有金融holdout；训练误差不等于策略收益。

## 当前迁移证据

真实520690×47矩阵导入通过；490655成熟训练标签和原126日权重保持一致。
在新导出目录运行原half126分数、100万元/原范围/Top15/2%/1%历史成交限额，正常及
双成本的9类CSV各自与原归档逐字节一致，共18份。包含daily、trades、entries、fills、
order_attempts、positions、distributions、allocation_states、liquidity_orders。

证据在本机`~/stockops-exports/research-portability-validation/replay-parity.json`。
这验证数据迁移及移入源码没有改变这两个账户，不等于验证GPU训练或新数据提高收益。
