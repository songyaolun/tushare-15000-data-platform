# StockOps 独立研究机工作流

另一台机器只需 GitHub 代码、uv 和自己的数据目录。研究入口已提供原始数据采集、
47 列与标签重建、训练、新特征追加和收益回放；不需要复制开发机缓存中的 Python 脚本。
最终验收记录见 [E2E 计划](e2e/return80-research-portability.md)。

已有冻结训练包时，先走下文的导入流程，最快开始训练。要增加数据、重算特征或更新日期，
走“从原始数据重建”流程。重新下载得到供应商当前提供的历史版本，应单独冻结，
不能因为日期相同就假定与旧归档、旧收益完全一致。

## 安装与数据

```bash
git clone https://github.com/songyaolun/tushare-15000-data-platform.git
cd tushare-15000-data-platform
git switch main
git pull --ff-only
git pull --ff-only
git rev-parse HEAD
env -u PYTHONPATH uv sync --frozen --python 3.11 --extra rebuild --extra training
env -u PYTHONPATH uv run --frozen --extra rebuild --extra training return80-research --help
```

按最终交付记录的完整 SHA 固定代码，后续更新也应记录新 SHA。安装需要访问 GitHub
及包源；`rebuild` 的三个 StockOps 依赖已固定提交，GitHub SSH 访问需在新机器配置好。
只导入数据训练时可以省去 `--extra rebuild`。每次 `uv run` 仍须带需要的 extra，
避免 uv 将刚安装的可选依赖同步移除。训练依赖固定 `scikit-learn==1.9.1`。

研究代码与 StockOps App 分开安装。App 是每日运行与推送入口，本仓负责数据研究和训练；
新机器无需安装 App，也无需设置跨仓 `PYTHONPATH`。

已有冻结数据可通过两个本地导出包搬运到另一台机器，不要求两机在同一内网：

| 文件 | 用途 | 字节数与 SHA256 |
|---|---|---|
| `half126-training-20261020.zip` | 原47列、候选轴、标签和权重，约103MB | `103398538`；`588734d35352918a58d733665f45bb52431c73766f7c19b5a47c3b59dc0f2e0a` |
| `half126-replay-20261020.zip` | 完整回放输入与旧模型原生分数，约895MB | `895200806`；`0de7b1a981602930284b0afff2d4b1a811979aa97de955f0ea395b34c638257e` |

本机导出位于 `~/stockops-exports/`，它是本地路径，不能当作新机器的下载URL。
传输后用 `shasum -a 256 文件名.zip` 核对，再解压。ZIP不含原始接口全历史；需要
新增数据时仍按后面的采集计划取数。数据不进入Git。

训练包解压后：

```bash
uv run --frozen return80-research import-training --bundle /你的路径/half126-training-20261020 --output /你的路径/research/training
```

导入验证SHA、47列顺序、完整候选行轴、成熟标签和126日权重。输出独立NPY与相对路径
`dataset.json`，支持内存映射；不依赖旧机器HOME、原spec的绝对路径或PYTHONPATH。
已有输出不覆盖。数据集中的X含全部候选，训练用`X[fit_indices]`；y和sample_weight
已经是训练子集，不能再次用fit_indices索引。预测要覆盖全部X并保持行序。

## 新增Tushare数据和特征

采集复用现有HTTPS Client、RawStore、限流、重试和完整分页。凭据统一使用
`data/secrets/tushare.env`，按[README 的凭据约定](../README.md#tushare-凭据约定)
在本机配置；命令显式传入路径，程序不自动加载。传公开JSON计划，不传Token：

```json
{"schema":"return80-collection-plan/v1","requests":[{"api":"daily_basic","parameters":{"trade_date":"20261016"},"fields":["ts_code","trade_date","turnover_rate"],"primary_key":["ts_code","trade_date"],"page_size":3000,"maximum_pages":10}]}
```

这是单日示例；训练用新增列必须列全所需历史分区与预热窗口。财报接口用period并带完整
报告身份主键，不能假设ts_code/end_date就足以区分所有版本。接口是否授权以实际返回为准。

```bash
uv run --frozen return80-research collect --plan plan.json --output raw-study --credential-file data/secrets/tushare.env --live
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

这一步在已有原数据的机器运行一次，再通过操作者选择的文件传输方式搬运整个目录。
它包含约 1.8GB 行情面板及候选、旧评分、公司行动，每个文件有 SHA。
回放包已将这个目录打包；解压得到 `half126-replay-20261020/` 和
`half126-baseline-20261020/`。前者给 `replay --dataset`，后者的 `scores.npy` 给
`--scores` 即可回放旧模型；其 `protocol.json` 保存分数SHA和原配置。训练 ZIP
不包含回放行情。若不搬运行情，可以在新机执行后文重建流程，
使用它生成的 `rebuilt/replay`；这属于新数据版本，须与同次生成的训练数据配套。

训练后保存全部候选预测到predictions.npy，再映射并按原配置回放：

```bash
uv run --frozen return80-research map-predictions --dataset research/with-turnover --predictions predictions.npy --output scores.npy
uv run --frozen return80-research replay --dataset replay-data --scores scores.npy --output accounts/new-model --capital 1000000 --scope original --position-weight .02 --candidate-count 15 --opening-cash-only --maximum-daily-participation .01 --capture-execution-evidence --name comparison
```

这个配置是原half126研究比较基线，不是当前12万元均衡分档生产配置。比较时原模型和新模型
必须使用同一资本、范围和执行规则。命令同时运行正常/双成本，保留交易、费用、逐日净值、
实际成交/取消与核账。全五年样本内拟合/选参，没有金融holdout；训练误差不等于策略收益。

## 当前迁移证据

真实520690×47矩阵导入通过。从原始归档重建后，全部47列、490655成熟训练行、
标签和126日权重逐项一致；41份回放输入的SHA也一致。重建验收发现并补齐了原训练
已用的20260930财务补充分区，按原清单核对来源，未手改任何特征值。
在新导出目录运行原half126分数、100万元/原范围/Top15/2%/1%历史成交限额，正常及
双成本的9类CSV各自与原归档逐字节一致，共18份。包含daily、trades、entries、fills、
order_attempts、positions、distributions、allocation_states、liquidity_orders。

证据在本机`~/stockops-exports/research-portability-validation/replay-parity.json`。
这验证数据迁移及移入源码没有改变这两个账户，不等于验证GPU训练或新数据提高收益。

## 从原始数据重建

安装：`uv sync --frozen --extra rebuild --extra training`。重建复用已经推送的固定
StockOps domain源码，而不是依赖旁边恰好存在的成员checkout。精确依赖见E2E计划。

```bash
# 先生成目录/日历请求；以下写的目录均由操作者指定。
uv run --frozen --extra rebuild return80-research make-plan --start 20200101 --end 20261016 --output bootstrap-plan.json
# 真采集要显式--live，密钥文件为本机私有文件；不要把Token写进命令。
uv run --frozen --extra rebuild return80-research collect --plan bootstrap-plan.json --output bootstrap --credential-file data/secrets/tushare.env --live
uv run --frozen --extra rebuild return80-research make-plan --start 20200101 --end 20261016 --bootstrap bootstrap --output history-plan.json
uv run --frozen --extra rebuild return80-research collect --plan history-plan.json --output history-collection --credential-file data/secrets/tushare.env --live
uv run --frozen --extra rebuild return80-research materialize-sources --bootstrap bootstrap --collection history-collection --output sources
uv run --frozen --extra rebuild return80-research prepare-sources --sources sources --output prepared
uv run --frozen --extra rebuild return80-research rebuild-matrices --spec prepared/rebuild.json --output rebuilt
# 显式选择训练轮数；下面只是四轮运行示例，不等于正式模型的容量。
uv run --frozen --extra training return80-research train --dataset rebuilt/training --output model-smoke --rounds 4 --thread-ceiling 16
```

训练会检查当前机器资源，使用实际输入做有界线程对比并核对分数一致，再运行
指定轮数。默认HGB走CPU；GPU信息只作为资源记录，未宣称HGB支持GPU。
XGBoost/LightGBM/CatBoost可直接消费dataset中的X、fit_indices、y和sample_weight，
预测完整候选轴后使用`map-predictions`接入相同回放。当前未验证任何GPU框架。

重建同时产出`training/`和完整`replay/`目录。采集按日、证券、报告期
和历史成员分页，原文留存，未知值不填零。供应商今天返回的历史修订可能与原
归档不同；新下载结果要形成新冻结版本，不能自动宣称重现旧收益。

## 给 GPU 训练程序的数据约定

`training/dataset.json` 列出每个 NPY 文件的 SHA 与列名。先用
`return80_research.dataset.load_dataset()` 检查，再交给训练框架：

```python
from pathlib import Path
import numpy as np
from tushare_entitlement_hub.return80_research.dataset import load_dataset

arrays, manifest = load_dataset(Path("rebuilt/training"))
fit_features = arrays["X"][arrays["fit_indices"]]
fit_targets = arrays["y"]
fit_weights = arrays["sample_weight"]
# 用目标框架训练，再对 arrays["X"] 的全部行预测，保持原顺序。
# np.save("predictions.npy", predictions, allow_pickle=False)
```

`X` 为全部候选的 float32 矩阵，NaN 表示缺失；`y` 和 `sample_weight` 已是成熟训练
子集。标签的独立交易止盈为 4%，共享组合回放止盈为 4.5%，两者不可混用。
目标 GPU 框架及驱动由该机器单独验证，本项目未把 XGBoost、LightGBM 或 CatBoost
加入冻结环境，也未声称已验证它们的 GPU 训练。另建实验环境时应固定框架版本，保存
模型参数、数据清单 SHA 和资源记录，预测后仍接同一回放入口比较。

当前重建固定原研究口径：47 列、2021-10-18 起候选、成熟标签、126 交易日半衰期、
全样本拟合。新增数据源本身不会改变这些规则。训练入口只保存原生模型和预测，
不自动重跑全量独立预测校验或账户回放；成功以 `complete.json` 为准。失败保留中间
产物供定位，重新执行请选择新输出目录。采集分区支持校验后续跑。

历史归档支持普通JSON和gzip压缩文件，哈希固定实际文件字节。复现原实验前应依据
原sources/protocol清单核对全部报告期和补充分区；通用历史计划会覆盖截止日前最新
季度，不能凭目录名称挑选旧缓存。完整来源、原始行数和SHA随验收证据保存。
