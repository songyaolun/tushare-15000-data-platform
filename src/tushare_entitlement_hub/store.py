"""DuckDB 控制面和只增版本表，检查点与标准数据在同一事务提交。"""

from __future__ import annotations

import hashlib
import json
from calendar import monthrange
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from .models import CatalogEntry, entry_from_dict
from .raw import canonical_json


def encode(value: Any) -> str:
    """输出稳定 JSON；encode({'b': 1, 'a': 2}) 得到按键排序的字符串。

    参数为公开 JSON 兼容对象，返回排序后的紧凑字符串；凭据和非有限数值被拒绝。
    示例：encode({'b': 1, 'a': 2}) → '{"a":2,"b":1}'。
    """
    return canonical_json(value)


def digest(value: Any) -> str:
    """计算无凭据业务对象的 SHA256；digest({'a': 1}) 返回 64 位十六进制摘要。

    参数为公开的 JSON 对象，返回确定性摘要；相同字段不同顺序得到相同结果。
    示例：digest({'a': 1}) == digest({'a': 1}) → True。
    """
    return hashlib.sha256(encode(value).encode()).hexdigest()


def coverage_interval(task: dict[str, Any]) -> tuple[date, date]:
    """把任务范围映射成可连续比较的自然日区间，明确月、季、年和报告期语义。

    输入具有 start、end、date_field 和 params 的任务；返回闭区间起止日期。
    示例：month 的 2024-01-15 至 2024-01-20 查询覆盖整个一月，返回 01-01 至 01-31；
    period='20240331' 返回一季度范围；普通 end_date 任务仍保留原自然日范围。
    period 必须为有效季末，不能把不明报告频率扩大成已覆盖的年度。
    """
    first = date.fromisoformat(task["start"])
    last = date.fromisoformat(task["end"])
    if first > last:
        raise ValueError("task_range_invalid")
    field = task["date_field"]
    period = task.get("params", {}).get("period")
    if period is not None:
        text = str(period)
        parsed = date.fromisoformat(text)
        if parsed.month not in {3, 6, 9, 12} or parsed.day != monthrange(
            parsed.year, parsed.month
        )[1] or not first <= parsed <= last:
            raise ValueError("task_period_invalid")
        first = last = parsed
        if field != "year":
            field = "quarter"
    if field == "month":
        first = first.replace(day=1)
        last = last.replace(day=monthrange(last.year, last.month)[1])
    elif field == "quarter":
        first = first.replace(month=((first.month - 1) // 3) * 3 + 1, day=1)
        end_month = ((last.month - 1) // 3 + 1) * 3
        last = last.replace(month=end_month, day=monthrange(last.year, end_month)[1])
    elif field == "year":
        first = first.replace(month=1, day=1)
        last = last.replace(month=12, day=31)
    return first, last


def task_identity(task: dict[str, Any]) -> str:
    """仅对稳定执行语义生成任务 ID，规划时间只作为可变化的审计元数据。

    输入完整计划载荷；返回接口、日期域、范围、模式、参数、来源和政策摘要的哈希。
    示例：两个计划仅 planned_at 不同会返回同一 ID；params 改变则返回不同 ID。
    缺少执行字段时抛 ValueError，避免把不完整任务登记为有效计划。
    """
    fields = ("api_name", "date_field", "start", "end", "mode", "params", "source", "policy_hash")
    if any(field not in task for field in fields):
        raise ValueError("task_payload_incomplete")
    return digest({field: task[field] for field in fields})


class Store:
    """本地单写者控制面；Store(Path('hub.duckdb')) 创建必要表并限单线程。"""

    def __init__(self, path: Path) -> None:
        """打开本地数据库并初始化表；输入新路径，输出可用 Store 实例。

        参数 path 是数据库文件路径；返回实例，创建父目录与缺少的控制表。
        示例：store = Store(Path('/tmp/demo.duckdb'))；store.catalog() → []。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(path), config={"threads": 1, "memory_limit": "512MB"})
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS catalog (api_name VARCHAR PRIMARY KEY, payload VARCHAR);
            CREATE TABLE IF NOT EXISTS probes (
                api_name VARCHAR, observed_at VARCHAR, source VARCHAR, status VARCHAR,
                evidence VARCHAR, PRIMARY KEY(api_name, observed_at, source));
            CREATE TABLE IF NOT EXISTS tasks (
                task_id VARCHAR PRIMARY KEY, api_name VARCHAR, payload VARCHAR,
                state VARCHAR, observed_at VARCHAR, source VARCHAR, error_code VARCHAR);
            CREATE TABLE IF NOT EXISTS standard (
                api_name VARCHAR, business_key VARCHAR, version VARCHAR, observed_at VARCHAR,
                source VARCHAR, payload VARCHAR,
                PRIMARY KEY(api_name, business_key, version, source));
            CREATE TABLE IF NOT EXISTS checkpoints (
                api_name VARCHAR, date_field VARCHAR, source VARCHAR, value VARCHAR,
                observed_at VARCHAR, task_id VARCHAR, PRIMARY KEY(api_name, date_field, source));
            CREATE TABLE IF NOT EXISTS quality (
                task_id VARCHAR PRIMARY KEY, api_name VARCHAR, findings VARCHAR, fields VARCHAR);
            CREATE TABLE IF NOT EXISTS consumption (
                product VARCHAR, api_name VARCHAR, source VARCHAR, observed_at VARCHAR,
                row_count INTEGER, manifest_hash VARCHAR,
                PRIMARY KEY(product, api_name, source));
        """)

    def close(self) -> None:
        """释放连接；store.close() 后 DuckDB 文件可由下一进程打开。

        无额外参数，返回 None；关闭本实例持有的连接，不删除数据库或原始文件。
        示例：store.close() → None，随后可用新 Store 打开同一路径。
        """
        self.connection.close()

    def save_catalog(self, entries: list[CatalogEntry]) -> None:
        """按接口更新总账，保留未出现在新快照中的旧记录；[daily] → 登记 daily。

        参数为发现记录列表，返回 None；按名称合并证据并保护既有 blocked 状态。
        示例：save_catalog([CatalogEntry('daily')]) 后 catalog()[0].api_name → 'daily'。
        """
        previous = {entry.api_name: entry for entry in self.catalog()}
        for entry in entries:
            older = previous.get(entry.api_name)
            if older is not None:
                if older.lifecycle == "blocked" and entry.lifecycle in {
                    "production",
                    "research",
                    "archive",
                }:
                    entry.lifecycle = "blocked"
                known = {encode(item) for item in entry.evidence}
                entry.evidence.extend(item for item in older.evidence if encode(item) not in known)
                for name in (
                    "minimum_points",
                    "row_limit",
                    "frequency_per_minute",
                    "permission_mode",
                ):
                    old, new = getattr(older, name), getattr(entry, name)
                    if old is not None and new is not None and old != new:
                        entry.needs_review = True
            self.connection.execute(
                "INSERT OR REPLACE INTO catalog VALUES (?, ?)",
                [entry.api_name, encode(asdict(entry))],
            )

    def catalog(self) -> list[CatalogEntry]:
        """读取按名称排序的完整总账；空库返回 []，含政策的记录恢复为数据类。

        无额外参数，返回按 api_name 排序的 CatalogEntry 列表，不对记录隐式批准权限。
        示例：新数据库的 catalog() → []；嵌套政策从 JSON 恢复为 Policy。
        """
        return [
            entry_from_dict(json.loads(row[0]))
            for row in self.connection.execute(
                "SELECT payload FROM catalog ORDER BY api_name"
            ).fetchall()
        ]

    def record_probe(
        self, api_name: str, observed_at: str, source: str, status: str, evidence: dict[str, Any]
    ) -> None:
        """保存无凭据探活证据；fixture/available 只计离线成功，不计账户实测。

        参数为接口、观察时间、证据来源、状态和公开证据；返回 None 并保存观测。
        示例：record_probe('daily', '2024-01-03T00:00:00Z', 'fixture', 'empty', {})
        使 probes 表增加一条 empty 记录，不改变真实账户生命周期。
        """
        self.connection.execute(
            "INSERT OR REPLACE INTO probes VALUES (?, ?, ?, ?, ?)",
            [api_name, observed_at, source, status, encode(evidence)],
        )
        if source == "live" and status in {"unauthorized", "unsupported", "available", "empty"}:
            entries = [entry for entry in self.catalog() if entry.api_name == api_name]
            for entry in entries:
                if status in {"unauthorized", "unsupported"}:
                    entry.lifecycle = "blocked"
                elif entry.lifecycle == "blocked" and entry.policy is not None:
                    entry.lifecycle = entry.policy.lifecycle
            for entry in entries:
                self.connection.execute(
                    "INSERT OR REPLACE INTO catalog VALUES (?, ?)",
                    [entry.api_name, encode(asdict(entry))],
                )

    def save_plan(self, tasks: list[dict[str, Any]], observed_at: str) -> None:
        """登记计划中全部日期分片，已有完成记录不覆盖；两天计划形成两条 planned 任务。

        参数为任务列表与计划观察时间，返回 None；同范围旧失败计划可被新计划替代。
        示例：两个不同日期任务输入后 records('tasks') 的两条 state 均为 'planned'。
        """
        for task in tasks:
            if self._registered_state(task, task["source"]) is not None:
                continue
            for record in self.records("tasks"):
                older = json.loads(record["payload"])
                if (
                    record["state"] in {"planned", "failed"}
                    and record["task_id"] != task["task_id"]
                    and all(
                        older.get(field) == task.get(field)
                        for field in ("api_name", "date_field", "source", "start", "end", "params")
                    )
                ):
                    self.connection.execute(
                        "UPDATE tasks SET state='superseded' WHERE task_id=?", [record["task_id"]]
                    )
            self.connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, 'planned', ?, ?, '') ON CONFLICT DO NOTHING",
                [task["task_id"], task["api_name"], encode(task), observed_at, task["source"]],
            )

    def begin_task(self, task: dict[str, Any], observed_at: str, source: str) -> bool:
        """校验任务身份后启动新计划或重试失败任务，保留已登记的原始载荷。

        参数为完整任务、观察时间和证据来源；planned/failed 可进入 running，
        running/superseded 等状态拒绝重复启动，载荷被篡改时在写入前抛 ValueError。
        示例：新任务第一次 begin_task(...) → True；该任务已提交后再次调用 → False。
        """
        state = self._registered_state(task, source)
        if state == "committed":
            return False
        if state is None:
            self.connection.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, 'running', ?, ?, '')",
                [task["task_id"], task["api_name"], encode(task), observed_at, source],
            )
        elif state in {"planned", "failed"}:
            self.connection.execute(
                "UPDATE tasks SET state='running', observed_at=?, error_code='' WHERE task_id=?",
                [observed_at, task["task_id"]],
            )
        else:
            raise ValueError("task_state_not_runnable")
        return True

    def _registered_state(self, task: dict[str, Any], source: str) -> str | None:
        """校验执行身份、来源及持久化载荷，仅允许 planned_at 元数据发生变化。

        输入完整任务和来源；返回数据库状态，无记录时返回 None，冲突则抛 ValueError。
        示例：保留 task_id 却改变 params 会抛 task_identity_mismatch；同执行语义
        仅重新规划时间改变时通过校验，原持久载荷仍保留第一次登记的 planned_at。
        """
        if task.get("task_id") != task_identity(task):
            raise ValueError("task_identity_mismatch")
        if source not in {"fixture", "live"} or task["source"] != source:
            raise ValueError("task_source_mismatch")
        existing = self.connection.execute(
            "SELECT state, payload, api_name, source FROM tasks WHERE task_id=?", [task["task_id"]]
        ).fetchone()
        if existing is None:
            return None
        payload = json.loads(existing[1])
        supplied = {key: value for key, value in task.items() if key != "planned_at"}
        persisted = {key: value for key, value in payload.items() if key != "planned_at"}
        if (encode(supplied) != encode(persisted) or existing[2] != task["api_name"]
                or existing[3] != source):
            raise ValueError("task_payload_mismatch")
        return str(existing[0])

    def fail_task(self, task_id: str, code: str, findings: list[dict[str, str]]) -> None:
        """将任务标记失败并保存质量问题；输入 quality_failed，不推进水位。

        参数为任务ID、有限错误码与质量问题列表，返回 None；不写标准数据或检查点。
        示例：fail_task(task_id, 'quality_failed', []) 后该任务 state → 'failed'。
        """
        self.connection.execute(
            "UPDATE tasks SET state='failed', error_code=? WHERE task_id=?", [code, task_id]
        )
        self.connection.execute(
            """INSERT OR REPLACE INTO quality
            SELECT task_id, api_name, ?, '[]' FROM tasks WHERE task_id=?""",
            [encode(findings), task_id],
        )

    def commit_task(
        self,
        task: dict[str, Any],
        rows: list[dict[str, Any]],
        primary_key: list[str],
        observed_at: str,
        source: str,
        fields: list[str],
        findings: list[dict[str, str]] | None = None,
    ) -> None:
        """原子写版本、质量凭据与水位；任何 SQL 错误回滚，旧检查点保持不变。

        示例：daily 任务包含两条合格行，提交后 state=committed 且水位等于 task.end。
        """
        if self._registered_state(task, source) != "running":
            raise ValueError("task_not_running")
        self.connection.execute("BEGIN TRANSACTION")
        try:
            for row in rows:
                business_key = digest([row.get(key) for key in primary_key])
                version = digest(
                    [row["source_fields"], row["observed_at"], row.get("conversion_version")]
                )
                self.connection.execute(
                    """INSERT INTO standard VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING""",
                    [task["api_name"], business_key, version, observed_at, source, encode(row)],
                )
            self.connection.execute(
                "INSERT OR REPLACE INTO quality VALUES (?, ?, ?, ?)",
                [task["task_id"], task["api_name"], encode(findings or []), encode(fields)],
            )
            self.connection.execute(
                "UPDATE tasks SET state='committed', error_code='' WHERE task_id=?",
                [task["task_id"]],
            )
            siblings = [
                record
                for record in self.records("tasks")
                if record["api_name"] == task["api_name"]
                and record["source"] == source
                and json.loads(record["payload"])["date_field"] == task["date_field"]
            ]
            checkpoint = self._continuous_watermark(task, source, siblings)
            if checkpoint is not None:
                self.connection.execute(
                    """INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (api_name, date_field, source) DO UPDATE SET
                    value=greatest(checkpoints.value, excluded.value),
                    observed_at=greatest(checkpoints.observed_at, excluded.observed_at),
                    task_id=excluded.task_id""",
                    [
                        task["api_name"],
                        task["date_field"],
                        source,
                        checkpoint,
                        observed_at,
                        task["task_id"],
                    ],
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _continuous_watermark(
        self, task: dict[str, Any], source: str, siblings: list[dict[str, Any]]
    ) -> str | None:
        """仅从既有水位或最早有效计划起点推进连续覆盖，未规划日期同样形成缺口。

        输入当前任务、来源和同水位域任务；返回新 ISO 日期，没有推进则返回 None。
        示例：仅完成 2024-01-01 与 2024-01-03 时返回 2024-01-01；补齐 01-02 后
        可推进到 01-03。尚未完成的证券分片会阻断其覆盖期间，不混入其他日期字段。
        """
        intervals = [
            (*coverage_interval(json.loads(record["payload"])), record["state"])
            for record in siblings if record["state"] != "superseded"
        ]
        previous = self.watermark(task["api_name"], task["date_field"], source)
        candidate = date.fromisoformat(previous) if previous is not None else None
        if not intervals or candidate == date.max:
            return None
        next_day = candidate + timedelta(days=1) if candidate else min(
            start for start, _, _ in intervals
        )
        pending = [(start, end) for start, end, state in intervals if state != "committed"]
        completed = sorted((start, end) for start, end, state in intervals if state == "committed")
        for start, end in completed:
            if end < next_day:
                continue
            if start > next_day:
                break
            blockers = [
                max(start_pending, next_day)
                for start_pending, end_pending in pending
                if end_pending >= next_day and start_pending <= end
            ]
            if blockers and min(blockers) == next_day:
                break
            candidate = min(blockers) - timedelta(days=1) if blockers else end
            if blockers or candidate == date.max:
                break
            next_day = candidate + timedelta(days=1)
        if candidate is None or candidate.isoformat() == previous:
            return None
        return candidate.isoformat()

    def watermark(self, api_name: str, date_field: str, source: str = "fixture") -> str | None:
        """返回接口与日期维度的独立水位；没有 daily/trade_date 时返回 None。

        参数指定接口、日期维度与来源，返回 ISO 日期或 None，不混用其他接口水位。
        示例：空库 watermark('daily', 'trade_date') → None。
        """
        row = self.connection.execute(
            """SELECT value FROM checkpoints
            WHERE api_name=? AND date_field=? AND source=?""",
            [api_name, date_field, source],
        ).fetchone()
        return str(row[0]) if row else None

    def rows(self, api_name: str, source: str = "fixture") -> list[dict[str, Any]]:
        """读取所有历史标准版本；rows('income_vip') 保留观察到的每次财务修订。

        参数为接口及证据来源，返回标准记录字典列表，保留历史修订及来源字段。
        示例：空库 rows('income_vip') → []；fixture 数据不会进入 source='live' 的结果。
        """
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                """SELECT payload FROM standard
            WHERE api_name=? AND source=? ORDER BY observed_at, version""",
                [api_name, source],
            ).fetchall()
        ]

    def previous_fields(self, api_name: str, source: str = "fixture") -> list[str] | None:
        """返回最近成功任务的字段契约；首次采集返回 None。

        参数为接口及证据来源，返回最近成功批次的来源字段列表或 None。
        示例：首次 previous_fields('daily') → None；提交 ts_code/trade_date 后返回对应列表。
        """
        row = self.connection.execute(
            """SELECT q.fields FROM quality q JOIN tasks t USING(task_id)
            WHERE q.api_name=? AND t.source=? AND t.state='committed'
            ORDER BY t.observed_at DESC LIMIT 1""",
            [api_name, source],
        ).fetchone()
        return json.loads(row[0]) if row else None

    def records(self, table: str) -> list[dict[str, Any]]:
        """读取固定控制表的审计行；records('tasks') 返回字典列表，任意表名拒绝。

        参数只允许 tasks/probes/checkpoints/quality/consumption，返回行字典列表。
        示例：空库 records('tasks') → []；records('arbitrary') 抛 ValueError。
        """
        if table not in {"tasks", "probes", "checkpoints", "quality", "consumption"}:
            raise ValueError("未知控制表")
        result = self.connection.execute(f"SELECT * FROM {table}")
        columns = [column[0] for column in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    def consume(
        self,
        product: str,
        api_name: str,
        source: str,
        observed_at: str,
        row_count: int,
        manifest_hash: str,
    ) -> None:
        """登记实际消费的行数和清单哈希；空产品不产生消费证据。

        参数为产品、接口、来源、观察时间、正行数和清单哈希；返回 None，更新消费账本。
        示例：consume('market', 'daily', 'fixture', '2024-01-03', 1, 'example')
        记录 row_count=1；空产品或 row_count=0 抛 ValueError。
        """
        if not product.strip() or row_count <= 0:
            raise ValueError("消费证据需要产品及实际数据")
        self.connection.execute(
            "INSERT OR REPLACE INTO consumption VALUES (?, ?, ?, ?, ?, ?)",
            [product, api_name, source, observed_at, row_count, manifest_hash],
        )

    def export_parquet(self, destination: Path, source: str = "fixture") -> int:
        """把标准层可审计版本表导出 Parquet；返回导出的版本行数。

        参数为目标 Path 与 fixture/live 来源，返回实际导出的行数，创建目标父目录。
        示例：库内两条 fixture 版本时 export_parquet(Path('/tmp/demo.parquet')) → 2，
        且目标可作为 Parquet 文件读取；过滤参数不参与目标路径的 SQL 绑定。
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.connection.sql("SELECT * FROM standard WHERE source=?", params=[source]).write_parquet(
            str(destination)
        )
        rows = self.connection.execute(
            "SELECT count(*) FROM standard WHERE source=?", [source]
        ).fetchall()
        return int(rows[0][0])
