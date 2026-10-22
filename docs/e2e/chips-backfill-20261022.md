# 新基础资格池筹码定向补采验收

适用 revision：包含 `return80_research/chips_backfill.py` 的本次提交；最终完整 SHA
由根任务交付记录固定。本模块只依赖本仓 Client、RawStore、DuckDB、NumPy，精确
版本由本仓 `uv.lock` 固定。规范化复用研究仓现有 `chip_distribution_summary`，研究仓
最终依赖 SHA 由根任务冻结，不允许把本地 PYTHONPATH 当跨机器安装方式。

环境：Python 3.11+、uv 冻结环境和可写 VA 原始根。离线测试不需要凭据。
真实采集只在已获授权且指定 `--live` 时运行，凭据通过既有私有配置读取；不得打印、
持久化或哈希凭据。官方 HTTPS、连接复用；不改旧计划、旧原始层或旧 summary。

离线必跑：`uv run --frozen pytest tests/test_return80_chips_backfill.py -q`，加上 Ruff
及 `git diff --check`。覆盖精确日期集合分组、重复/截断分页、证券错配、原始 SHA
篡改、缺失日期保留、50101 非成功证据、连续异常预算及空计划。

已授权真实阶段：新轴周末候选 t 与精确日历 t-5 减旧 summary 已有键，形成缺口
parquet；先保存其 SHA 和分片计划，进行有界分页/范围/吞吐实测，再按新日期优先执行。
每个成功原始响应先进入 RawStore，完整短页证明结束后才形成收据；明确 50101 写独立
失败原文证据且保留全部缺口，禁止伪装成功空页。正常接口空页也单列缺失日期。
限额与线程按当前实测调整，保留前一配置和已完成收据；不无限重试未知错误。
短于完整滚动窗口的成功探测不能证明持续限额。后续若需确认未知限额，至少连续覆盖
两个完整窗口，并同时观察主采集与其他接口的实际成功量。2026-10-22 的 cyq_chips
持续超限返回 40203，明确对象为 cyq_chips 接口、每分钟 1000 次；主采集回退为总
950 次/分钟，保留余量，不再上探已知硬限。限频分片是可重试失败，绝非空源或50101；
原失败原文保留，按原计划重试，最终不得有未重试分片。测试覆盖先限频后成功仍保留
原错误证据，真实限频正文只保存既有安全原始层，不写聊天或凭据。

多进程验收：先用相同已保存页做 1×12 与 3×4 worker 的有限离线重放，核对页数、
行数、需日期数一致及真实耗时，再决定是否启用 `--processes 3 --workers 12`。
`--api-rate` 和 `--account-rate` 均是所有进程总预算，均分至分片进程；请求按索引
取模互斥，原始分片标识不变。每进程写独立 workers 进度和 PID/starttime，主进程汇总。
新增与复用数量分别统计，只用新增工作计算 ETA，新增阶段不足一分钟时 ETA 为 null。
切换前在收据完整边界停止旧主，不并行两个主任务重复同一队列；执行源码、计划 SHA、
配置和 PID/starttime 随每次执行归档。测试必须验证分片预算均分、总计数、独立进度与
共享原始路径，未知失败不得因子进程正常退出而变成整体成功。
已运行任务若父汇总停止但子进程仍健康，使用 `monitor_workers` 从各子进程的独立
进度恢复汇总，不为补日志重启采集。短暂文件读取失败保留上次观察；只有全部子进程
最终 manifest 存在才允许总终态，任何残余失败或未完成分片都不能报告完整成功。

配额恢复：VA 出现 Errno122 后所有采集进程必须停止请求，保存已有收据。已授权本地
持久暂存 `/home/tiger/stockops-staging/basic-universe-nextweek-20261022-v1/chips`，
先检查实际可写/fsync、字节与 inode 空间，再通过指向原有效 partition 的 symlink 复用，
恢复时仍逐页 SHA 验证，禁止修改其原始数据。此临时混合根依赖 VA，不可宣称可搬迁完整包。
根目录存在 `archive-raw-pages` 时，新分片先保存全部 RawStore 原页、完成分页校验，再
归档 `raw.tar.gz`，receipt 增加 `raw_archive: {path, sha256}`；原 `files` 中每页路径和
SHA 不变。receipt 持久化后才删本次新建原页副本。读取时先验 archive SHA，再逐个
`extractfile(files.path)` 校验原页 SHA，不使用 extractall。只有完成独立打包和路径检查
后才能称为跨机交付；VA 旧原始根始终保留。

原始页和收据 SHA 验证之后才能规范化，逐证券日期使用同日未复权 close，校验筹码
总权重与价格档位，并调用既有研究函数。新增 summary 单独写入，合并不得覆盖旧表；
迁移特征采用 adj_factor 调整及精确 t-5，19:30 可用口径不构成历史 PIT 认证。

证据：`raw/basic-universe-nextweek-20261022-v1/chips/` 下 plan、probe、原始页、
receipt、collection.log、progress.json、最终 collection.json；以及 expanded 下缺口、
补充 summary、剩余缺口及哈希。通过标准为计划全范围终态、无未解释异常或截断，
每个缺口 key 均可定位成功分布或明确缺失证据，规范化与原函数一致。

桌面、390×844 手机与跨端 UI 一致性：not applicable，本节点不改变 UI 或生产入口。
跨端数据验收是冻结安装后同一输入 SHA、summary 行键和数值一致；未做 Mac 实测不得
宣称 Mac 实机通过。该阶段不训练、不发飞书、不发布、不交易、不注册调度。
