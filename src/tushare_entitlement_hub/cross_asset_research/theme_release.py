"""从通用参考库构建并发布可复用的 P02 本地数据产品。"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, cast

import duckdb

from ..product_readiness import P02_SCOPE_REQUIREMENTS, assess_product_readiness
from ..reference_storage import atomic_json, file_checksum, reference_fingerprint
from ..store import digest, encode
from .theme_ranking import (
    P02_FACTOR_WEIGHTS,
    ThemeObservation,
    average_tie_percentiles,
    calculate_theme_features,
    normalize_reason_tokens,
    parse_board_count,
    rank_theme_features,
)

SOURCE_THEME_TABLES = {
    "ths": ("ths_index", "ths_member", "name", "con_code"),
    "dc": ("dc_index", "dc_member", "name", "con_code"),
    "tdx": ("tdx_index", "tdx_member", "name", "con_code"),
    "kpl": (None, "kpl_concept_cons", "desc", "con_code"),
}


def _table_exists(connection: duckdb.DuckDBPyConnection, table: str) -> bool:
    """判断 accepted schema 是否存在指定表，表名只由内部固定列表传入。

    输入 ``daily`` 在表存在时返回 True；缺表返回 False，不捕获其他数据库错误。
    """
    row = connection.execute(
        "SELECT count(*)=1 FROM information_schema.tables "
        "WHERE table_schema='accepted' AND table_name=?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def load_scope_certificates(connection: duckdb.DuckDBPyConnection) -> tuple[dict[str, Any], ...]:
    """读取参考库范围证书；旧库无证书表时返回空元组并由 readiness 关闭发布。

    输入已打开的 reference 连接，输出按证书 ID 排序的字典元组；载荷损坏时抛异常，
    不把损坏证书当作缺失后继续发榜。
    """
    exists_row = connection.execute(
        "SELECT count(*)=1 FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name='scope_certificate'"
    ).fetchone()
    exists = bool(exists_row and exists_row[0])
    if not exists:
        return ()
    return tuple(
        cast(dict[str, Any], json.loads(payload))
        for (payload,) in connection.execute(
            "SELECT payload FROM scope_certificate ORDER BY certificate_id"
        ).fetchall()
    )


def resolve_theme_trade_date(
    connection: duckdb.DuckDBPyConnection,
    requested: str | None,
) -> str:
    """选择不晚于请求日的最新 A 股日行情日期，缺日行情时明确失败。

    输入 ``requested='2026-10-10'`` 且最新行情为 10 月 9 日，返回 ``2026-10-09``；
    requested 为 None 时返回库内最新日期。不会使用系统今天替代业务日期。
    """
    if not _table_exists(connection, "daily"):
        raise ValueError("p02_daily_missing")
    row = connection.execute(
        "SELECT max(trade_date)::VARCHAR FROM accepted.daily "
        "WHERE (?::DATE IS NULL OR trade_date<=?::DATE)",
        [requested, requested],
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("p02_trade_date_unavailable")
    return str(row[0])


def _read_records(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    fields: tuple[str, ...],
    target_date: str | None = None,
) -> list[dict[str, Any]]:
    """从固定 accepted 表投影有界字段，可选按 trade_date 精确过滤。

    输入 daily、字段和目标日，返回字典列表；缺表返回空列表。字段及表来自模块常量，
    调用方不得把用户文本传入。输出日期统一为字符串，数值保持 DuckDB 类型。
    """
    if not _table_exists(connection, table):
        return []
    available = {row[0] for row in connection.execute(f'DESCRIBE accepted."{table}"').fetchall()}
    selected = tuple(field for field in fields if field in available)
    if not selected:
        return []
    where = (
        " WHERE trade_date=?::DATE" if target_date is not None and "trade_date" in available else ""
    )
    projections = [
        f'CAST("{field}" AS VARCHAR) AS "{field}"' if field == "observed_at" else f'"{field}"'
        for field in selected
    ]
    rows = connection.execute(
        f'SELECT {",".join(projections)} FROM accepted."{table}"{where}',
        [target_date] if where else [],
    ).fetchall()
    return [
        {
            field: value.isoformat() if hasattr(value, "isoformat") else value
            for field, value in zip(selected, row, strict=True)
        }
        for row in rows
    ]


def load_theme_observations(
    connection: duckdb.DuckDBPyConnection,
    target_date: str,
) -> tuple[ThemeObservation, ...]:
    """把同花顺目录、当前成员和目标日市场记录连接成主题观察。

    成员按每个主题最新 ``observed_at`` 快照读取；行情、涨跌停、热度、资金和开盘啦
    事件严格限定 target_date。输出不含综合分，可被纯评分函数和其他产品复用。
    """
    indexes = _read_records(
        connection, "ths_index", ("ts_code", "name", "observed_at", "source_record_id")
    )
    members = _read_records(
        connection,
        "ths_member",
        ("ts_code", "con_code", "observed_at", "source_record_id"),
    )
    daily = _read_records(
        connection,
        "daily",
        ("ts_code", "pct_chg", "close", "trade_date", "source_record_id"),
        target_date,
    )
    limits = _read_records(
        connection,
        "stk_limit",
        ("ts_code", "up_limit", "trade_date", "source_record_id"),
        target_date,
    )
    hot = _read_records(
        connection,
        "ths_hot",
        ("ts_code", "rank", "trade_date", "source_record_id"),
        target_date,
    )
    flows = _read_records(
        connection,
        "moneyflow_cnt_ths",
        ("ts_code", "net_amount", "company_num", "trade_date", "source_record_id"),
        target_date,
    )
    events = _read_records(
        connection,
        "kpl_list",
        ("ts_code", "status", "theme", "lu_desc", "trade_date", "source_record_id"),
        target_date,
    )
    latest_index: dict[str, dict[str, Any]] = {}
    for row in indexes:
        code = str(row.get("ts_code") or "")
        if code and str(row.get("observed_at") or "") >= str(
            latest_index.get(code, {}).get("observed_at") or ""
        ):
            latest_index[code] = row
    latest_member_observation: dict[str, str] = {}
    for row in members:
        code = str(row.get("ts_code") or "")
        latest_member_observation[code] = max(
            latest_member_observation.get(code, ""), str(row.get("observed_at") or "")
        )
    members_by_theme: dict[str, dict[str, dict[str, Any]]] = {}
    for row in members:
        theme_code = str(row.get("ts_code") or "")
        member_code = str(row.get("con_code") or "")
        if (
            theme_code
            and member_code
            and str(row.get("observed_at") or "") == latest_member_observation.get(theme_code)
        ):
            members_by_theme.setdefault(theme_code, {})[member_code] = row
    daily_by_member = {
        str(row.get("ts_code")): row for row in daily if row.get("ts_code") is not None
    }
    limits_by_member = {
        str(row.get("ts_code")): row for row in limits if row.get("ts_code") is not None
    }
    heat_by_member: dict[str, tuple[float, str]] = {}
    for row in hot:
        rank = row.get("rank")
        if isinstance(rank, (int, float)) and math.isfinite(rank) and rank > 0:
            code = str(row.get("ts_code"))
            score = 1 / float(rank)
            if code not in heat_by_member or score > heat_by_member[code][0]:
                heat_by_member[code] = (score, str(row.get("source_record_id") or ""))
    flow_by_theme = {
        str(row.get("ts_code")): row for row in flows if row.get("ts_code") is not None
    }
    event_by_member: dict[str, dict[str, Any]] = {}
    for row in events:
        code = str(row.get("ts_code") or "")
        if code and parse_board_count(str(row.get("status") or "")) >= parse_board_count(
            str(event_by_member.get(code, {}).get("status") or "")
        ):
            event_by_member[code] = row
    observations = []
    for theme_code, member_rows in sorted(members_by_theme.items()):
        directory = latest_index.get(theme_code, {})
        member_codes = tuple(sorted(member_rows))
        returns: dict[str, float] = {}
        limit_known = set()
        limit_up = set()
        boards: dict[str, int] = {}
        reasons: dict[str, tuple[str, ...]] = {}
        for member_code in member_codes:
            quote, boundary = daily_by_member.get(member_code), limits_by_member.get(member_code)
            if quote is not None:
                value = quote.get("pct_chg")
                if isinstance(value, (int, float)) and math.isfinite(value):
                    returns[member_code] = float(value)
            if boundary is not None:
                limit_known.add(member_code)
                close, up_limit = (
                    None if quote is None else quote.get("close"),
                    boundary.get("up_limit"),
                )
                if (
                    isinstance(close, (int, float))
                    and isinstance(up_limit, (int, float))
                    and math.isclose(float(close), float(up_limit), rel_tol=1e-6, abs_tol=0.001)
                ):
                    limit_up.add(member_code)
            event = event_by_member.get(member_code)
            if event is not None:
                boards[member_code] = parse_board_count(str(event.get("status") or ""))
                reasons[member_code] = normalize_reason_tokens(
                    event.get("theme"), event.get("lu_desc")
                )
        flow = flow_by_theme.get(theme_code)
        net_amount = None if flow is None else flow.get("net_amount")
        company_count = None if flow is None else flow.get("company_num")
        flow_strength = (
            float(net_amount) / float(company_count)
            if isinstance(net_amount, (int, float))
            and isinstance(company_count, (int, float))
            and math.isfinite(net_amount)
            and math.isfinite(company_count)
            and company_count > 0
            else None
        )
        observations.append(
            ThemeObservation(
                theme_code=theme_code,
                theme_name=str(directory.get("name") or theme_code),
                member_codes=member_codes,
                returns_by_member=returns,
                flow_strength=flow_strength,
                heat_by_member={
                    code: heat_by_member[code][0] for code in member_codes if code in heat_by_member
                },
                limit_known_members=frozenset(limit_known),
                limit_up_members=frozenset(limit_up),
                board_counts=boards,
                reason_tokens_by_member=reasons,
            )
        )
    return tuple(observations)


def normalize_theme_name(value: str) -> str:
    """规范主题名称用于跨来源确定性匹配，不扩展同义词或调用模型。

    输入 ``'AI 传媒（同花顺）'`` 返回 ``ai传媒同花顺``；仅去空白和标点并统一大小写，
    不擅自删除行业层级、地区或业务限定词。
    """
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).casefold()


def load_source_theme_rankings(
    connection: duckdb.DuckDBPyConnection,
    target_date: str,
    ths_observations: tuple[ThemeObservation, ...] = (),
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], frozenset[str]]]:
    """生成同花顺、东财、通达信和开盘啦的独立价格扩散分榜。

    每个来源只用自身成员定义，成员行情统一取 Tushare daily 目标日；至少五个有效成员且
    覆盖率达到 90% 才入榜。返回分榜行和成员集合，后者只用于严格跨来源一致性匹配。
    """
    daily = _read_records(
        connection, "daily", ("ts_code", "pct_chg", "source_record_id"), target_date
    )
    returns = {
        str(row["ts_code"]): float(row["pct_chg"])
        for row in daily
        if isinstance(row.get("pct_chg"), (int, float)) and math.isfinite(row["pct_chg"])
    }
    rankings: list[dict[str, Any]] = []
    member_sets: dict[tuple[str, str], frozenset[str]] = {}
    for source, (
        catalog_table,
        member_table,
        name_field,
        member_field,
    ) in SOURCE_THEME_TABLES.items():
        grouped: dict[str, set[str]]
        fallback_names: dict[str, str]
        if source == "ths" and ths_observations:
            names = {
                observation.theme_code: observation.theme_name for observation in ths_observations
            }
            grouped = {
                observation.theme_code: set(observation.member_codes)
                for observation in ths_observations
            }
            fallback_names = {}
        else:
            member_rows = _read_records(
                connection,
                member_table,
                (
                    "ts_code",
                    member_field,
                    "trade_date",
                    name_field,
                    "observed_at",
                    "source_record_id",
                ),
                target_date,
            )
            catalog_rows = (
                _read_records(
                    connection,
                    catalog_table,
                    ("ts_code", name_field, "observed_at", "source_record_id"),
                )
                if catalog_table is not None
                else []
            )
            names = {
                str(row.get("ts_code")): str(row.get(name_field) or row.get("ts_code"))
                for row in catalog_rows
                if row.get("ts_code") is not None
            }
            if source == "ths":
                latest_observation: dict[str, str] = {}
                for row in member_rows:
                    code = str(row.get("ts_code") or "")
                    latest_observation[code] = max(
                        latest_observation.get(code, ""), str(row.get("observed_at") or "")
                    )
                member_rows = [
                    row
                    for row in member_rows
                    if str(row.get("observed_at") or "")
                    == latest_observation.get(str(row.get("ts_code") or ""))
                ]
            grouped = {}
            fallback_names = {}
            for row in member_rows:
                theme_code = str(row.get("ts_code") or "")
                member_code = str(row.get(member_field) or "")
                if theme_code and member_code:
                    grouped.setdefault(theme_code, set()).add(member_code)
                    if row.get(name_field):
                        fallback_names.setdefault(theme_code, str(row[name_field]))
        candidates = []
        for theme_code, members in grouped.items():
            observed = {member: returns[member] for member in members if member in returns}
            coverage = len(observed) / len(members) if members else 0.0
            if len(observed) < 5 or coverage < 0.90:
                continue
            row = {
                "source": source,
                "theme_code": theme_code,
                "theme_name": names.get(theme_code, fallback_names.get(theme_code, theme_code)),
                "member_count": len(members),
                "effective_member_count": len(observed),
                "price_coverage": coverage,
                "member_up_fraction": sum(value > 0 for value in observed.values()) / len(observed),
                "equal_weight_return": math.fsum(observed.values()) / len(observed),
            }
            candidates.append(row)
            member_sets[(source, theme_code)] = frozenset(members)
        breadth = (
            average_tie_percentiles(
                {row["theme_code"]: row["member_up_fraction"] for row in candidates}
            )
            if candidates
            else {}
        )
        returns_rank = (
            average_tie_percentiles(
                {row["theme_code"]: row["equal_weight_return"] for row in candidates}
            )
            if candidates
            else {}
        )
        for row in candidates:
            row["price_diffusion_score"] = 50 * (
                breadth[row["theme_code"]] + returns_rank[row["theme_code"]]
            )
        candidates.sort(key=lambda row: (-row["price_diffusion_score"], row["theme_code"]))
        for rank, row in enumerate(candidates, 1):
            row["source_rank"] = rank
        rankings.extend(candidates)
    return rankings, member_sets


def build_cross_source_consensus(
    rankings: list[dict[str, Any]],
    member_sets: dict[tuple[str, str], frozenset[str]],
) -> list[dict[str, Any]]:
    """把名称一致且成员 Jaccard 不低于 30% 的来源结果连到同花顺主题。

    输入各来源独立分榜，输出至少两个来源确认的主题。每个外部来源对一个同花顺主题只取
    重合度最高项；不按模糊名称或模型语义合并，避免把相近题材强行当作同一主题。
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rankings:
        by_source.setdefault(str(row["source"]), []).append(row)
    consensus = []
    for base in by_source.get("ths", []):
        base_members = member_sets.get(("ths", base["theme_code"]), frozenset())
        matches = [base]
        overlaps = {"ths": 1.0}
        normalized = normalize_theme_name(base["theme_name"])
        for source, rows in sorted(by_source.items()):
            if source == "ths":
                continue
            candidates = []
            for row in rows:
                other = member_sets.get((source, row["theme_code"]), frozenset())
                union = base_members | other
                overlap = len(base_members & other) / len(union) if union else 0.0
                if normalize_theme_name(row["theme_name"]) == normalized and overlap >= 0.30:
                    candidates.append((overlap, row))
            if candidates:
                overlap, selected = max(
                    candidates, key=lambda item: (item[0], item[1]["price_diffusion_score"])
                )
                matches.append(selected)
                overlaps[source] = overlap
        if len(matches) >= 2:
            consensus.append(
                {
                    "ths_theme_code": base["theme_code"],
                    "theme_name": base["theme_name"],
                    "source_count": len(matches),
                    "sources": sorted(row["source"] for row in matches),
                    "matched_theme_codes": {row["source"]: row["theme_code"] for row in matches},
                    "member_jaccard": overlaps,
                    "consensus_score": math.fsum(row["price_diffusion_score"] for row in matches)
                    / len(matches),
                }
            )
    consensus.sort(
        key=lambda row: (-row["source_count"], -row["consensus_score"], row["ths_theme_code"])
    )
    for rank, row in enumerate(consensus, 1):
        row["consensus_rank"] = rank
    return consensus


def build_theme_release(reference: Path, requested_date: str | None = None) -> dict[str, Any]:
    """从只读参考库构建 P02 内存 release，不写文件也不改变来源库。

    返回目标日、readiness、正式主榜、价格扩散榜、事件摘要和证据链接。全局范围证书
    不完整时主榜为空，价格扩散榜仍按逐主题行情门禁生成。
    """
    before = reference_fingerprint(reference)
    with duckdb.connect(
        str(reference), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        target_date = resolve_theme_trade_date(connection, requested_date)
        certificates = load_scope_certificates(connection)
        readiness = assess_product_readiness(
            "P02", certificates, P02_SCOPE_REQUIREMENTS, target_date=target_date
        )
        observations = load_theme_observations(connection, target_date)
        source_rankings, source_member_sets = load_source_theme_rankings(
            connection, target_date, observations
        )
    if reference_fingerprint(reference) != before:
        raise ValueError("reference_changed_during_p02_read")
    features = [calculate_theme_features(observation) for observation in observations]
    main, price = rank_theme_features(features, release_ready=readiness.ready)
    events = [
        {
            "theme_code": observation.theme_code,
            "limit_up_members": sorted(observation.limit_up_members),
            "board_counts": dict(sorted(observation.board_counts.items())),
            "reason_tokens_by_member": {
                key: list(value)
                for key, value in sorted(observation.reason_tokens_by_member.items())
            },
        }
        for observation in observations
    ]
    consensus = build_cross_source_consensus(source_rankings, source_member_sets)
    release_identity = digest(
        [
            before,
            target_date,
            P02_FACTOR_WEIGHTS,
            readiness.reason_codes,
            main,
            price,
            source_rankings,
            consensus,
        ]
    )
    return {
        "release_id": release_identity,
        "reference_fingerprint": before,
        "reference_path": str(reference.resolve()),
        "target_date": target_date,
        "readiness": {
            "status": readiness.status,
            "reason_codes": list(readiness.reason_codes),
            "certified_day_counts": readiness.certified_day_counts,
        },
        "main_ranking": main,
        "price_diffusion": price,
        "event_evidence": events,
        "source_rankings": source_rankings,
        "cross_source_consensus": consensus,
        "theme_count": len(observations),
        "historical_pit_certified": False,
    }


def _write_parquet(
    path: Path,
    columns: tuple[tuple[str, str], ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """用单线程 512MB DuckDB 写 ZSTD Parquet，嵌套值编码为规范 JSON。

    输入列定义和字典行，输出路径、行数、字节和 SHA256；空结果仍写带类型的空 Parquet，
    使下游视图结构稳定。未知字段不会进入文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(config={"threads": 1, "memory_limit": "512MB"}) as connection:
        definition = ",".join(f'"{name}" {kind}' for name, kind in columns)
        connection.execute(f"CREATE TABLE result({definition})")
        names = [name for name, _ in columns]
        projections = [
            "CAST(unnest(?::VARCHAR[]) AS JSON)" if kind == "JSON" else f"unnest(?::{kind}[])"
            for _, kind in columns
        ]
        for start in range(0, len(rows), 10_000):
            batch = rows[start : start + 10_000]
            values = [
                [
                    encode(value) if isinstance(value, (dict, list, tuple)) else value
                    for row in batch
                    for value in (row.get(name),)
                ]
                for name in names
            ]
            connection.execute("INSERT INTO result SELECT " + ",".join(projections), values)
        connection.execute(
            "COPY result TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)",
            [str(path)],
        )
    return {
        "path": str(path),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": file_checksum(path),
    }


def _write_evidence_links(reference: Path, target_date: str, path: Path) -> dict[str, Any]:
    """用 DuckDB 直接联接成员与来源记录并写证据 ID，不把百万级关系装入 Python。

    输入统一参考库、目标日和输出路径，返回 Parquet 路径、行数、字节和 SHA256。
    成员按主题最新观察快照选取，目录、成员、行情、涨跌停、热度、资金和事件记录均保留。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped_reference = str(reference.resolve()).replace("'", "''")
    escaped_output = str(path.resolve()).replace("'", "''")
    with duckdb.connect(config={"threads": 1, "memory_limit": "512MB"}) as connection:
        connection.execute(f"ATTACH '{escaped_reference}' AS reference (READ_ONLY)")
        query = """
            WITH member_versions AS (
                SELECT ts_code AS theme_code, con_code, source_record_id, observed_at,
                       max(observed_at) OVER (PARTITION BY ts_code) AS latest_observed_at
                FROM reference.accepted.ths_member
            ), members AS (
                SELECT theme_code, con_code, source_record_id
                FROM member_versions WHERE observed_at=latest_observed_at
            ), links AS (
                SELECT theme_code, source_record_id FROM members
                UNION
                SELECT index_rows.ts_code, index_rows.source_record_id
                FROM reference.accepted.ths_index index_rows
                JOIN (SELECT DISTINCT theme_code FROM members) themes
                  ON themes.theme_code=index_rows.ts_code
                UNION
                SELECT members.theme_code, daily.source_record_id
                FROM members JOIN reference.accepted.daily daily
                  ON daily.ts_code=members.con_code AND daily.trade_date=?::DATE
                UNION
                SELECT members.theme_code, limits.source_record_id
                FROM members JOIN reference.accepted.stk_limit limits
                  ON limits.ts_code=members.con_code AND limits.trade_date=?::DATE
                UNION
                SELECT members.theme_code, hot.source_record_id
                FROM members JOIN reference.accepted.ths_hot hot
                  ON hot.ts_code=members.con_code AND hot.trade_date=?::DATE
                UNION
                SELECT members.theme_code, events.source_record_id
                FROM members JOIN reference.accepted.kpl_list events
                  ON events.ts_code=members.con_code AND events.trade_date=?::DATE
                UNION
                SELECT themes.theme_code, flow.source_record_id
                FROM (SELECT DISTINCT theme_code FROM members) themes
                JOIN reference.accepted.moneyflow_cnt_ths flow
                  ON flow.ts_code=themes.theme_code AND flow.trade_date=?::DATE
            )
            SELECT theme_code, source_record_id FROM links
            WHERE nullif(source_record_id, '') IS NOT NULL
        """
        connection.execute(
            "COPY (" + query + f") TO '{escaped_output}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)",
            [target_date] * 5,
        )
        count_row = connection.execute(
            "SELECT count(*) FROM read_parquet(?)", [str(path)]
        ).fetchone()
        rows = int(count_row[0]) if count_row else 0
    return {
        "path": str(path),
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": file_checksum(path),
    }


def publish_theme_release(destination: Path, release: dict[str, Any]) -> dict[str, Any]:
    """不可变发布 P02 Parquet、轻量 DuckDB、CSV 摘要和 current 指针。

    相同 release_id 重跑会校验并复用；新 release 完整写入并通过 SQL 行数核对后才原子
    更新 ``current.json``。产品表只保存结果和证据 ID，不复制来源整行。
    """
    release_id = str(release["release_id"])
    release_directory = destination / "releases" / release_id[:20]
    manifest_path = release_directory / "release.json"
    if manifest_path.exists():
        existing = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
        if existing.get("release_id") != release_id:
            raise ValueError("p02_release_identity_collision")
        atomic_json(destination / "current.json", existing)
        return existing
    release_directory.mkdir(parents=True, exist_ok=False)
    ranking_columns = (
        ("theme_code", "VARCHAR"),
        ("theme_name", "VARCHAR"),
        ("rank", "BIGINT"),
        ("score", "DOUBLE"),
        ("raw_score", "DOUBLE"),
        ("sample_shrinkage", "DOUBLE"),
        ("member_count", "BIGINT"),
        ("effective_member_count", "BIGINT"),
        ("price_coverage", "DOUBLE"),
        ("limit_coverage", "DOUBLE"),
        ("member_up_fraction", "DOUBLE"),
        ("equal_weight_return", "DOUBLE"),
        ("flow_strength", "DOUBLE"),
        ("heat_strength", "DOUBLE"),
        ("limit_up_breadth", "DOUBLE"),
        ("board_strength", "DOUBLE"),
        ("reason_consistency", "DOUBLE"),
        ("factor_percentiles", "JSON"),
        ("reason_codes", "JSON"),
    )
    price_columns = tuple(
        (name, kind)
        for name, kind in ranking_columns
        if name not in {"rank", "score", "raw_score", "sample_shrinkage", "factor_percentiles"}
    ) + (("price_diffusion_rank", "BIGINT"), ("price_diffusion_score", "DOUBLE"))
    table_specs = {
        "main_ranking": (ranking_columns, release["main_ranking"]),
        "price_diffusion": (price_columns, release["price_diffusion"]),
        "event_evidence": (
            (
                ("theme_code", "VARCHAR"),
                ("limit_up_members", "JSON"),
                ("board_counts", "JSON"),
                ("reason_tokens_by_member", "JSON"),
            ),
            release["event_evidence"],
        ),
        "source_ranking": (
            (
                ("source", "VARCHAR"),
                ("theme_code", "VARCHAR"),
                ("theme_name", "VARCHAR"),
                ("source_rank", "BIGINT"),
                ("price_diffusion_score", "DOUBLE"),
                ("member_count", "BIGINT"),
                ("effective_member_count", "BIGINT"),
                ("price_coverage", "DOUBLE"),
                ("member_up_fraction", "DOUBLE"),
                ("equal_weight_return", "DOUBLE"),
            ),
            release["source_rankings"],
        ),
        "cross_source_consensus": (
            (
                ("ths_theme_code", "VARCHAR"),
                ("theme_name", "VARCHAR"),
                ("consensus_rank", "BIGINT"),
                ("consensus_score", "DOUBLE"),
                ("source_count", "BIGINT"),
                ("sources", "JSON"),
                ("matched_theme_codes", "JSON"),
                ("member_jaccard", "JSON"),
            ),
            release["cross_source_consensus"],
        ),
    }
    files = {
        name: _write_parquet(release_directory / f"{name}.parquet", columns, rows)
        for name, (columns, rows) in table_specs.items()
    }
    files["evidence_link"] = _write_evidence_links(
        Path(release["reference_path"]),
        str(release["target_date"]),
        release_directory / "evidence_link.parquet",
    )
    database = release_directory / "products.duckdb"
    with duckdb.connect(
        str(database), config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        connection.execute("CREATE SCHEMA p02")
        for name, item in files.items():
            escaped = str(Path(item["path"]).resolve()).replace("'", "''")
            connection.execute(
                f"CREATE VIEW p02.\"{name}\" AS SELECT * FROM read_parquet('{escaped}')"
            )
            count_row = connection.execute(f'SELECT count(*) FROM p02."{name}"').fetchone()
            count = int(count_row[0]) if count_row else -1
            if count != item["rows"]:
                raise ValueError("p02_release_row_count_mismatch")
        release_metadata = {
            "release_id": release["release_id"],
            "reference_fingerprint": release["reference_fingerprint"],
            "target_date": release["target_date"],
            "readiness": release["readiness"],
            "theme_count": release["theme_count"],
            "main_ranking_rows": len(release["main_ranking"]),
            "price_diffusion_rows": len(release["price_diffusion"]),
            "source_ranking_rows": len(release["source_rankings"]),
            "cross_source_consensus_rows": len(release["cross_source_consensus"]),
            "evidence_link_rows": files["evidence_link"]["rows"],
            "factor_weights": P02_FACTOR_WEIGHTS,
            "historical_pit_certified": False,
        }
        connection.execute("CREATE TABLE release(payload JSON)")
        connection.execute("INSERT INTO release VALUES (?)", [encode(release_metadata)])
        connection.execute("CHECKPOINT")
    csv_path = release_directory / "main_ranking.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        names = [
            name
            for name, _ in ranking_columns
            if name not in {"factor_percentiles", "reason_codes"}
        ]
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(release["main_ranking"])
    manifest = {
        "release_id": release_id,
        "target_date": release["target_date"],
        "readiness": release["readiness"],
        "theme_count": release["theme_count"],
        "main_ranking_rows": len(release["main_ranking"]),
        "price_diffusion_rows": len(release["price_diffusion"]),
        "reference_fingerprint": release["reference_fingerprint"],
        "database": str(database),
        "database_sha256": file_checksum(database),
        "csv": str(csv_path),
        "files": files,
        "storage_bytes": sum(item["bytes"] for item in files.values()) + database.stat().st_size,
        "historical_pit_certified": False,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(destination / "current.json", manifest)
    return manifest


def validate_theme_release(reference: Path, destination: Path) -> dict[str, Any]:
    """校验当前 P02 release 的来源指纹、文件、视图、门禁、排名和证据关系。

    输入统一参考库及 P02 根目录，合法时返回 ``valid=True`` 和各表行数；任何缺文件、
    哈希变化、排名断裂、主榜缺因子、覆盖不足或证据孤儿均返回固定失败原因，不抛来源载荷。
    """
    try:
        manifest = cast(
            dict[str, Any],
            json.loads((destination / "current.json").read_text(encoding="utf-8")),
        )
        if reference_fingerprint(reference) != manifest["reference_fingerprint"]:
            raise ValueError("p02_reference_fingerprint_mismatch")
        database = Path(manifest["database"])
        if file_checksum(database) != manifest["database_sha256"]:
            raise ValueError("p02_database_checksum_mismatch")
        for item in manifest["files"].values():
            path = Path(item["path"])
            if file_checksum(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("p02_file_checksum_mismatch")
        with duckdb.connect(
            str(database), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        ) as connection:
            counts = {}
            for name in manifest["files"]:
                count_row = connection.execute(f'SELECT count(*) FROM p02."{name}"').fetchone()
                counts[name] = int(count_row[0]) if count_row else -1
            if any(counts[name] != item["rows"] for name, item in manifest["files"].items()):
                raise ValueError("p02_row_count_mismatch")
            release_row = connection.execute("SELECT payload FROM release").fetchone()
            release = json.loads(release_row[0]) if release_row else {}
            if release.get("release_id") != manifest["release_id"]:
                raise ValueError("p02_release_identity_mismatch")
            main_violation_row = connection.execute(
                "SELECT count(*) FROM p02.main_ranking WHERE "
                "effective_member_count<5 OR price_coverage<0.9 OR limit_coverage<0.9 OR "
                "score IS NULL OR raw_score IS NULL OR member_up_fraction IS NULL OR "
                "equal_weight_return IS NULL OR flow_strength IS NULL OR heat_strength IS NULL OR "
                "limit_up_breadth IS NULL OR board_strength IS NULL OR reason_consistency IS NULL"
            ).fetchone()
            main_violations = int(main_violation_row[0]) if main_violation_row else -1
            rank_state = connection.execute(
                "SELECT count(*),count(DISTINCT rank),coalesce(min(rank),0),coalesce(max(rank),0) "
                "FROM p02.main_ranking"
            ).fetchone()
            if main_violations or rank_state != (
                counts["main_ranking"],
                counts["main_ranking"],
                1 if counts["main_ranking"] else 0,
                counts["main_ranking"],
            ):
                raise ValueError("p02_main_ranking_gate_mismatch")
            missing_evidence_row = connection.execute(
                "SELECT count(*) FROM p02.main_ranking ranking WHERE NOT EXISTS ("
                "SELECT 1 FROM p02.evidence_link evidence "
                "WHERE evidence.theme_code=ranking.theme_code)"
            ).fetchone()
            missing_evidence = int(missing_evidence_row[0]) if missing_evidence_row else -1
            evidence_orphan_row = connection.execute(
                "SELECT count(*) FROM p02.evidence_link evidence WHERE NOT EXISTS ("
                "SELECT 1 FROM p02.event_evidence themes "
                "WHERE themes.theme_code=evidence.theme_code)"
            ).fetchone()
            evidence_orphans = int(evidence_orphan_row[0]) if evidence_orphan_row else -1
            if missing_evidence or evidence_orphans:
                raise ValueError("p02_evidence_link_mismatch")
            if manifest["readiness"]["status"] == "ready" and not counts["main_ranking"]:
                raise ValueError("p02_ready_release_empty")
        return {
            "valid": True,
            "release_id": manifest["release_id"],
            "target_date": manifest["target_date"],
            "readiness": manifest["readiness"],
            "counts": counts,
            "reason_codes": [],
        }
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError, duckdb.Error):
        return {"valid": False, "reason_codes": ["p02_validation_failed"]}
