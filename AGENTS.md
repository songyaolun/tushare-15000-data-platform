# Project Agent Instructions

## Objective

Build a production-oriented Python platform that maximizes the useful coverage of every Tushare Pro points-based interface available to a 15,000-point account. Every discovered interface must have an explicit lifecycle state, collection policy, quality contract, and downstream information product or documented archival purpose.

## Engineering rules

- Use Python 3.11 or newer and `uv` for dependency management and command execution.
- Keep responsibilities separated. Put reusable logic in focused packages instead of large orchestration modules.
- Use complete, readable variable and method names. Avoid abbreviations that hide domain meaning.
- Every function must have a detailed docstring describing behavior and at least one input/output example.
- Every test case must document its scenario and expected result.
- Cover every function with tests and target 100% coverage.
- Tests must run without a real Tushare token or network access. Live probes require an explicit opt-in flag.
- Never persist, print, hash into cache keys, or include a Tushare token in exceptions, logs, manifests, or fixtures.
- Production Tushare transport must require HTTPS. Do not silently use an SDK path that sends credentials over plain HTTP.
- Preserve raw responses before normalization. Keep source units and fields auditable.
- Treat announcement time, effective time, observation time, and report period as separate concepts. Point-in-time behavior must fail closed.
- Separate points-based interfaces, independently purchased interfaces, deprecated interfaces, and empirically unavailable interfaces.
- Do not claim an interface is covered until its catalog policy, ingestion path, quality gate, tests, and information consumer are all present.

## 训练资源与进度硬规则

- 每次新训练前实时检查 CPU affinity、各级 cgroup CPU 配额、可用内存、已有负载，
  以及 GPU 型号/显存、框架和算法是否真正支持该 GPU；同时记录 OMP、BLAS、MKL
  线程设置。不能沿用旧机器的核数、额度或空闲状态。
- 用同一实际输入做有界短测，核对数值一致性后选择单模型线程数与模型并行数。
  禁止无依据单线程，也不得照搬其他机器的设置或为满载而制造资源争抢。
  本次16线程及测得加速比只适用于本次测量，不是所有机器的固定默认。
- 训练、原生预测、可移植推理、账户回放和核账必须及时输出真实阶段、已完成数量、
  总数量（可知时）、耗时、心跳和明确失败阶段。训练轮数使用框架支持的真实日志；
  CPU活跃、心跳、估算耗时都不能冒充完成百分比。
- 已运行任务变更并行或日志前，先评估重启损失和安全接入点。已加载函数不会随磁盘
  修改自动获得新日志；不得为补日志重启已完成拟合的任务。原执行源码/协议留快照，
  新执行版本单独冻结，不覆盖原证据。临时暂停调度必须恢复并回读确认。
- 资源不足、权限受限或算法不支持加速时，报告实测约束及可用方案；不虚构GPU支持、
  轮数进度或剩余时间。日志只记公开任务元数据，不输出凭据、样本内容或异常私密载荷。

## Delivery rules

- Implement changes on a feature branch and create a Merge Request targeting `main`.
- Do not merge, deploy, submit approvals, or configure production schedules unless the user explicitly requests it.
- Commit only source code, tests, documentation, and deterministic fixtures. Never commit credentials or live provider payloads containing sensitive data.
