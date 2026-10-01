"""公开文档发现、离线重放与可追溯权益合并；从不访问数据接口。"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
import markdown
from bs4 import BeautifulSoup, Tag

from tushare_entitlement_hub.models import CatalogEntry, Policy, policy_from_dict

BASE_URL = "https://tushare.pro"
DIRECTORY_URL = f"{BASE_URL}/document/2"
ENTITLEMENT_URL = f"{BASE_URL}/document/1?doc_id=290"
API_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*\b")
NON_APIS = {"api", "api_name", "pro", "tushare", "ts", "query", "pro_api", "set_token"}


def load_policies() -> dict[str, Policy]:
    """读取包内显式政策，不给未知接口套用默认策略。

    输入：无。输出：以真实接口名为键的政策对象；JSON 重名立即报错。
    示例：``load_policies()['daily'].primary_key`` 返回 ``['ts_code', 'trade_date']``。
    """
    policy_path = Path(__file__).parent / "data" / "policies.json"
    values: list[dict[str, Any]] = json.loads(policy_path.read_text(encoding="utf-8"))
    result: dict[str, Policy] = {}
    for value in values:
        policy = policy_from_dict(value)
        if policy.api_name in result:
            raise ValueError(f"政策接口重复：{policy.api_name}")
        result[policy.api_name] = policy
    return result


def _document_url(value: str, base: str = DIRECTORY_URL) -> str | None:
    """规范化白名单文档 URL，拒绝凭据、非 HTTPS、外域及数据接口。

    输入：相对或绝对链接。输出：合法完整 URL，否则 None。
    示例：``_document_url('/document/2?doc_id=27')`` 返回官方 daily 文档 URL。
    """
    try:
        parsed = urlsplit(urljoin(base, value))
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "tushare.pro"
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.path not in {"/document/1", "/document/2"}
    ):
        return None
    document_id = parse_qs(parsed.query).get("doc_id", [""])[0]
    if document_id and not document_id.isdigit():
        return None
    return f"{BASE_URL}{parsed.path}" + (f"?doc_id={document_id}" if document_id else "")


def _body(source: str, suffix: str) -> BeautifulSoup:
    """把 HTML 或 Markdown 转为仅含正文的树，避免导航栏污染接口识别。

    输入：文件内容及扩展名。输出：正文 BeautifulSoup。
    示例：``_body('# 日线\n接口：daily', '.md').get_text()`` 包含 daily。
    """
    html = (
        markdown.markdown(source, extensions=["tables", "fenced_code"])
        if suffix in {".md", ".markdown"}
        else source
    )
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one(".content.col-md-9") or soup.select_one("article") or soup
    for unwanted in content.select("script, style, nav, footer, .search-panel"):
        unwanted.decompose()
    return BeautifulSoup(str(content), "html.parser")


def _api_names(body: BeautifulSoup) -> list[str]:
    """从接口声明及调用示例提取真实名称，保留同页 VIP 与多接口声明。

    输入：正文树。输出：去重排序后的 API 名；不会把侧栏中文标题当 API。
    示例：``_api_names(_body('接口：income\npro.income_vip()', '.md'))`` 返回两项。
    """
    text = body.get_text(" ", strip=True)
    declared: set[str] = set()
    for match in re.finditer(r"(?:接口(?:名称|名)?|api_name)\s*[：:=]\s*", text):
        # 声明第一段英文即接口；逗号后可能是自然语言说明，调用示例另行补全。
        token = API_PATTERN.match(text[match.end() :].strip().strip("`'\""))
        if token:
            declared.add(token.group())
    references: set[str] = set()
    for match in re.finditer(r"\b(?:pro|ts)\s*\.\s*([a-z][a-z0-9_]+)\s*\(", text):
        references.add(match.group(1))
    for match in re.finditer(r"\bpro\s*\.\s*query\s*\(\s*['\"]([a-z][a-z0-9_]+)", text):
        references.add(match.group(1))
    if declared:
        declared.update(
            name
            for name in references
            if any(name.startswith(base + "_") for base in declared.copy())
        )
    elif "输入参数" in text or "输出参数" in text:
        declared.update(references)
    declared.update(re.findall(r"(?<![a-z0-9_])[a-z][a-z0-9_]*_vip(?![a-z0-9_])", text))
    return sorted(declared - NON_APIS)


def _number(text: str, patterns: tuple[str, ...]) -> int | None:
    r"""只提取显式数值，不把价格、年份或账户等级推断为接口配额。

    输入：原文和有捕获组的表达式。输出：整数或 None。
    示例：``_number('每分钟500次', (r'每分钟(\d+)次',))`` 返回 500。
    """
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def _paragraph_sentences(body: BeautifulSoup) -> list[str]:
    """按显式换行拆分正文权益段落，避免同一段中的普通版与 VIP 串项。

    输入：正文树。输出：保留内联强调文字、去重后的非空行。
    示例：``<p>积分：2000<br>VIP需5000积分</p>`` 返回两行。
    """
    sentences: list[str] = []
    for paragraph in body.select("p, li"):
        if paragraph.find_parent("table"):
            continue
        copied = BeautifulSoup(str(paragraph), "html.parser")
        for line_break in copied.select("br"):
            line_break.replace_with("\n")
        sentences.extend(line.strip() for line in copied.get_text().splitlines() if line.strip())
    return list(dict.fromkeys(sentences))


def _update_description(body: BeautifulSoup) -> str:
    """保存官方明确的数据更新时段，未披露时返回空串而不推测日期。

    输入：正文树。输出：原文中的更新时间说明。
    示例：“交易日每天15点～16点之间入库”原样保留，不能转成文档修改日期。
    """
    for sentence in _paragraph_sentences(body):
        if re.search(
            r"(?:每天|每日|交易日|每周|每月|每季度|盘后|盘中|实时|定期|不定期)"
            r".{0,60}(?:更新|入库|发布)|(?:更新|入库)(?:时间|频率)\s*[：:]"
            r"|\d+日.{0,12}更新当月|每(?:周|月|季度).{0,20}更新",
            sentence,
        ):
            return sentence
    return ""


def _claims(body: BeautifulSoup, api_name: str) -> list[dict[str, Any]]:
    """从独立段落抽取权益断言，区分行数、频次、积分及独立购买。

    输入：正文及当前接口。输出：保留原句的断言列表；VIP 段不污染普通接口。
    示例：正文“积分：2000积分，每次最多6000行”产生积分与行数断言。
    """
    claims: list[dict[str, Any]] = []
    for sentence in _paragraph_sentences(body):
        mentioned = set(re.findall(r"(?<![a-z0-9_])[a-z]+(?:_[a-z0-9]+)+(?![a-z0-9_])", sentence))
        if any(name.endswith("_vip") for name in mentioned) and api_name not in mentioned:
            continue
        claim: dict[str, Any] = {"quote": sentence}
        independent = bool(
            re.search(r"单独(?:开通|开|权限|购买|的权限|申请)|独立权限|不在积分|不受积分", sentence)
        )
        points = _number(
            sentence,
            (
                r"(?:至少|需要|需有|需|达到|满|积累|积攒|累积|用户|积分[：:])\s*([\d,]+)\s*积分",
                r"([\d,]+)\s*积分(?:以上|即可|可|起)",
                r"积分[：:]\s*([\d,]+)\s*分",
                r"积分(?:达到|不少于|至少|要求为?)\s*([\d,]+)",
            ),
        )
        if api_name.endswith("_vip") and api_name not in mentioned:
            points = None
        if independent:
            claim["permission_mode"] = "independent"
        elif points is not None:
            claim["permission_mode"] = "points"
            claim["minimum_points"] = points
        row_limit = _number(
            sentence,
            (
                r"(?:单次|每次|一次)[^。；]{0,30}?(?:最多|最大|提取|返回|获取|请求|限量)\s*([\d,]+)\s*(?:行|条)",
                r"(?:单次|每次|一次)\s*([\d,]+)\s*(?:行|条)",
                r"最多\s*([\d,]+)\s*(?:行|条)",
                r"单次最大\s*([\d,]+)(?:[,，。]|$)",
            ),
        )
        frequency = _number(sentence, (r"每分钟(?:内|请求|可调取|可以请求)*\s*([\d,]+)\s*次",))
        tiers = [
            {"minimum_points": int(points_text), "frequency_per_minute": int(frequency_text)}
            for points_text, frequency_text in re.findall(
                r"(\d+)\s*积分(?:以上|起)?[^，。；]{0,12}?每分钟"
                r"(?:内|请求|可调取|可以请求)*\s*(\d+)\s*次",
                sentence,
            )
        ]
        eligible_tiers = [tier for tier in tiers if tier["minimum_points"] <= 15000]
        if eligible_tiers:
            frequency = max(eligible_tiers, key=lambda tier: tier["minimum_points"])[
                "frequency_per_minute"
            ]
            claim["frequency_tiers"] = tiers
            claim["frequency_for_points"] = 15000
        if row_limit is not None:
            claim["row_limit"] = row_limit
        if frequency is not None:
            claim["frequency_per_minute"] = frequency
        if re.search(r"(?:停止更新|停止维护|已停用|已废弃|不再更新)", sentence):
            claim["deprecated"] = True
        if len(claim) > 1:
            claims.append(claim)
    return claims


def _merge(entry: CatalogEntry, evidence: dict[str, Any]) -> None:
    """合并断言并保留冲突；数值冲突不覆盖原值并标记人工复核。

    输入：目录对象和一条证据。输出：原地更新，无返回值。
    示例：2000 与 5000 积分冲突后 ``entry.needs_review`` 为 True。
    """
    entry.evidence.append(evidence)
    for field in ("permission_mode", "minimum_points", "row_limit", "frequency_per_minute"):
        if field not in evidence:
            continue
        previous = getattr(entry, field)
        value = evidence[field]
        if previous not in (None, "unknown") and previous != value:
            entry.needs_review = True
            # 独立购买证据优先隔离；数值不猜测、不平均。
            if field == "permission_mode" and value == "independent":
                entry.permission_mode = "independent"
        else:
            setattr(entry, field, value)
    if evidence.get("deprecated"):
        entry.lifecycle = "deprecated"


def _table_evidence(
    documents: list[tuple[Path, BeautifulSoup, str]], entries: dict[str, CatalogEntry]
) -> None:
    """把权益表的明确链接和分类映射到已发现接口，保留总表来源。

    输入：文档正文三元组、已发现记录。输出：原地合并独立权限证据。
    示例：权益表“新闻资讯”会隔离 news、major_news、cctv_news。
    """
    aliases: dict[str, set[str]] = {
        "历史分钟": {"stk_mins"},
        "实时分钟": {"rt_min", "rt_min_daily"},
        "实时日线": {"rt_k"},
        "指数实时日线": {"rt_idx_k"},
        "申万指数实时行情": {"rt_sw_k"},
        "ETF实时日线": {"rt_etf_k"},
        "ETF实时参考": {"etf_iopv"},
        "期货历史分钟": {"ft_mins"},
        "期货实时分钟": {"rt_fut_min"},
        "期权历史分钟": {"opt_mins"},
        "申万分钟": {"sw_mins"},
        "新闻资讯": {"news", "major_news", "cctv_news"},
        "公告信息": {"anns_d", "fund_anns", "index_anns"},
        "可转债价格变动": {"cb_price_chg"},
        "盘前股本": {"stk_premarket"},
        "上证e互动、深证互动易": {"irm_qa_sh", "irm_qa_sz"},
        "集合竞价": {"stk_auction", "stk_auction_o", "stk_auction_c", "etf_auction"},
        "政策法规库": {"npr"},
        "券商研报库": {"broker_research"},
        "央行货币政策执行报告": {"pbc_policy"},
    }
    for path, body, source_url in documents:
        if "独立权限接口" not in body.get_text():
            continue
        full_text = body.get_text(" ", strip=True)
        if "分钟和港美股数据权限不在积分范畴" in full_text:
            macro_us = {"us_tycr", "us_trycr", "us_tbr", "us_tltr", "us_trltr"}
            for name, item in entries.items():
                if (name.startswith(("hk_", "us_")) and name not in macro_us) or (
                    name.endswith("_mins") or (name.startswith("rt_") and "_min" in name)
                ):
                    _merge(
                        item,
                        {
                            "source": str(path),
                            "doc_url": source_url,
                            "kind": "entitlement_scope",
                            "permission_mode": "independent",
                            "quote": "分钟和港美股数据权限不在积分范畴内，各类分钟单独分别开权限",
                        },
                    )
        for table in body.select("table"):
            headers = " ".join(cell.get_text(strip=True) for cell in table.select("th"))
            if "积分数" in headers and "每分钟频次" in headers:
                for row in table.select("tr"):
                    cells = row.select("td")
                    if cells and cells[0].get_text(strip=True) == "120" and "daily" in entries:
                        _merge(
                            entries["daily"],
                            {
                                "source": str(path),
                                "doc_url": source_url,
                                "kind": "entitlement_table",
                                "permission_mode": "points",
                                "minimum_points": 120,
                                "quote": row.get_text(" ", strip=True),
                            },
                        )
            headers = " ".join(cell.get_text(strip=True) for cell in table.select("th"))
            if "类型" not in headers or "包含数据" not in headers:
                continue
            for row in table.select("tr"):
                cells = row.select("td")
                if not cells:
                    continue
                kind = cells[0].get_text(strip=True)
                names = set(aliases.get(kind, set()))
                linked = {_document_url(str(link.get("href", ""))) for link in row.select("a")}
                names.update(name for name, item in entries.items() if item.doc_url in linked)
                if kind == "港股财报":
                    names.update(
                        name
                        for name in entries
                        if name.startswith("hk_")
                        and any(
                            value in name
                            for value in ("income", "balancesheet", "cashflow", "fina")
                        )
                    )
                if kind.startswith("美股"):
                    names.update(
                        name
                        for name in entries
                        if name.startswith("us_")
                        and (
                            (kind == "美股日线" and name in {"us_daily", "us_daily_adj"})
                            or (
                                kind == "美股财报"
                                and any(
                                    value in name
                                    for value in ("income", "balancesheet", "cashflow", "fina")
                                )
                            )
                        )
                    )
                for name in names & entries.keys():
                    _merge(
                        entries[name],
                        {
                            "source": str(path),
                            "doc_url": source_url,
                            "kind": "entitlement_table",
                            "permission_mode": "independent",
                            "quote": row.get_text(" ", strip=True),
                        },
                    )


def _source_url(path: Path, source: str) -> str:
    """从保存的来源标记或确定的文件名恢复文档 URL。

    输入：文档路径及内容。输出：白名单来源 URL，无法确定时为空。
    示例：``_source_url(Path('doc-27.html'), '')`` 返回 daily 文档地址。
    """
    match = re.search(
        r"source_url\s*[:=]\s*[\"']?(https://tushare\.pro/document/[12][^\s<>\"']*)", source
    )
    if match:
        return _document_url(match.group(1)) or ""
    match = re.fullmatch(r"doc-(\d+)", path.stem)
    if match:
        return f"{DIRECTORY_URL}?doc_id={match.group(1)}"
    if "entitlement" in path.stem:
        return ENTITLEMENT_URL
    if "directory" in path.stem:
        return DIRECTORY_URL
    return ""


def _merge_fetch_failures(
    paths: list[Path], entries: dict[str, CatalogEntry], verified_at: str
) -> None:
    """把抓取清单中的失败保留在目录债务，防止旧文件掩盖本轮失败。

    输入：文档路径、目录对象映射和核验时间。输出：原地追加失败证据或未解析记录。
    示例：doc-27 抓取超时后 daily 标记 needs_review；目录页失败也单独登记。
    """
    manifests = {
        manifest
        for path in paths
        for manifest in (path.rglob("index.json") if path.is_dir() else [path])
        if manifest.name == "index.json"
    }
    for manifest in sorted(manifests):
        values = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("抓取清单必须为列表")
        for value in values:
            if not isinstance(value, dict) or value.get("status") != "failed":
                continue
            url = _document_url(str(value.get("doc_url", "")))
            if url is None:
                continue
            matches = [entry for entry in entries.values() if entry.doc_url == url]
            if not matches:
                identifier = parse_qs(urlsplit(url).query).get("doc_id", ["directory"])[0]
                name = f"unresolved_doc_{identifier}"
                entries[name] = CatalogEntry(
                    api_name=name, title="公开文档抓取失败", doc_url=url, last_verified=verified_at
                )
                matches = [entries[name]]
            for entry in matches:
                entry.needs_review = True
                entry.evidence.append(
                    {
                        "kind": "fetch_failure",
                        "doc_url": url,
                        "source": str(manifest),
                        "error_type": str(value.get("error", "UnknownError")),
                    }
                )


def sync_catalog(
    paths: list[Path], verified_at: str, policies: dict[str, Policy] | None = None
) -> list[CatalogEntry]:
    """离线发现与合并 HTML/Markdown 目录，不执行网络请求或账户探测。

    输入：文件或目录列表、核验日期、可选政策。输出：按 api_name 排序的目录。
    缺失详细页会保留 unresolved_doc 债务；权益未知或冲突不会自动批准采集。
    示例：``sync_catalog([Path('fixtures/catalog')], '2026-10-01')`` 重放固化证据。
    """
    active_policies = load_policies() if policies is None else policies
    files = sorted(
        {
            file
            for path in paths
            for file in (path.rglob("*") if path.is_dir() else [path])
            if file.suffix.lower() in {".html", ".htm", ".md", ".markdown"}
        }
    )
    documents: list[tuple[Path, BeautifulSoup, str]] = []
    entries: dict[str, CatalogEntry] = {}
    links: dict[str, str] = {}
    resolved: set[str] = set()
    non_api: dict[str, str] = {}
    for path in files:
        source = path.read_text(encoding="utf-8")
        body = _body(source, path.suffix.lower())
        source_url = _source_url(path, source)
        documents.append((path, body, source_url))
        full = BeautifulSoup(source, "html.parser")
        for link in full.select("#jstree li a"):
            parent = link.parent
            if isinstance(parent, Tag) and parent.find("ul", recursive=False):
                continue
            url = _document_url(str(link.get("href", "")))
            if url and "/document/2?" in url:
                links[url] = link.get_text(" ", strip=True)
        names = _api_names(body)
        if "不提供API方式" in body.get_text():
            non_api[source_url] = body.get_text(" ", strip=True).split("数据字段")[0]
        if names:
            resolved.add(source_url)
        heading = body.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True) if heading else path.stem
        update_description = _update_description(body)
        for name in names:
            policy = active_policies.get(name)
            entry = entries.setdefault(
                name,
                CatalogEntry(
                    api_name=name,
                    title=title,
                    doc_url=source_url,
                    last_verified=verified_at,
                    policy=policy,
                    category=policy.category if policy else "unclassified",
                    lifecycle=policy.lifecycle if policy else "unclassified",
                ),
            )
            if update_description:
                if not entry.updated_at:
                    entry.updated_at = update_description
                entry.evidence.append(
                    {"kind": "update_schedule", "doc_url": source_url, "quote": update_description}
                )
            _merge(
                entry,
                {
                    "kind": "discovery",
                    "source": str(path),
                    "doc_url": source_url,
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "title": title,
                },
            )
            for claim in _claims(body, name):
                _merge(
                    entry, {"kind": "detail", "source": str(path), "doc_url": source_url, **claim}
                )
    _table_evidence(documents, entries)
    for url, title in links.items():
        if url in resolved:
            continue
        identifier = parse_qs(urlsplit(url).query)["doc_id"][0]
        name = f"unresolved_doc_{identifier}"
        entries[name] = CatalogEntry(
            api_name=name,
            title=title,
            doc_url=url,
            last_verified=verified_at,
            needs_review=True,
            evidence=[
                {
                    "kind": "unresolved_document",
                    "doc_url": url,
                    "quote": "目录已列出，但未取得可解析的接口声明；不能计入已覆盖接口。",
                }
            ],
        )
    for item in entries.values():
        if item.doc_url in non_api:
            item.permission_mode = "independent"
            item.lifecycle = "independent_permission"
            item.needs_review = False
            item.evidence = [
                {
                    "kind": "non_api_delivery",
                    "doc_url": item.doc_url,
                    "quote": non_api[item.doc_url],
                }
            ]
    _merge_fetch_failures(paths, entries, verified_at)
    for entry in entries.values():
        if entry.policy is not None and entry.row_limit is None:
            entry.needs_review = True
            entry.evidence.append(
                {
                    "kind": "quota_unknown",
                    "doc_url": entry.doc_url,
                    "quote": "文档未确认单次行上限；政策页大小仅为本地保守值，须复核。",
                }
            )
        if entry.permission_mode == "independent":
            entry.lifecycle = "independent_permission"
        if entry.policy is None and entry.doc_url not in non_api:
            entry.needs_review = True
        if entry.policy is not None and entry.policy.lifecycle == "blocked":
            entry.evidence.append(
                {
                    "kind": "collection_scope_exclusion",
                    "doc_url": entry.doc_url,
                    "quote": entry.policy.purpose,
                    "source": "package:data/policies.json",
                    "empirical_unavailable": False,
                }
            )
        missing_fields = [
            field
            for field in ("updated_at", "minimum_points", "row_limit", "frequency_per_minute")
            if getattr(entry, field) in (None, "")
        ]
        if missing_fields:
            entry.evidence.append(
                {
                    "kind": "missing_metadata",
                    "doc_url": entry.doc_url,
                    "missing_fields": missing_fields,
                    "quote": "官方页面未披露或未能确认这些元数据，不使用核验日期代替数据更新说明。",
                }
            )
    return sorted(entries.values(), key=lambda item: item.api_name)


def _sanitize_document(source: str) -> str:
    """仅保存公开目录与正文，去除登录界面、脚本和凭据示例实参。

    输入：公开文档 HTML。输出：可离线解析的净化 HTML，不包含追踪或登录脚本。
    示例：含 ``ts.set_token('example')`` 的代码块会保存为脱敏占位符。
    """
    original = BeautifulSoup(source, "html.parser")
    content = original.select_one(".content.col-md-9")
    navigation = original.select_one("#jstree")
    if content and re.search(r"接口(?:名称|名)?\s*[：:]", content.get_text()):
        navigation = None
    selected = [navigation, content]
    if not any(selected):
        selected = [original]
    clean = BeautifulSoup("<html><body></body></html>", "html.parser")
    for section in selected:
        if section is not None and clean.body is not None:
            clean.body.append(section)
    for unwanted in clean.select("script, style, nav, footer, form, input, iframe, .search-panel"):
        unwanted.decompose()
    for block in clean.select("pre"):
        code = block.get_text()
        code = re.sub(r"(set_token\s*\()([^)]*)(\))", r"\1'<REDACTED>'\3", code)
        code = re.sub(
            r"(pro_api\s*\(\s*(?:token\s*=\s*)?)(['\"])[^'\"]+\2",
            r"\1'<REDACTED>'",
            code,
        )
        block.clear()
        block.append(code)
    return str(clean)


def _fetch_one(url: str) -> tuple[str, str, str]:
    """抓取单个公开文档，不跟随重定向、不接收凭据、不访问 API 域。

    输入：白名单 HTTPS 文档 URL。输出：(URL, HTML, 错误类型)。
    示例：有效公开页成功返回非空 HTML；超时返回空 HTML 和 TimeoutException。
    """
    if _document_url(url) != url:
        return url, "", "InvalidDocumentURL"
    try:
        response = httpx.get(url, timeout=30, follow_redirects=False, trust_env=False)
        response.raise_for_status()
        return url, _sanitize_document(response.text), ""
    except httpx.HTTPError as error:
        return url, "", type(error).__name__


def fetch_documentation(destination: Path) -> list[Path]:
    """显式联网保存目录、权益表和可达详细页，最多四个公开 HTTP 请求并发。

    输入：本地目标目录。输出：本轮成功保存的 HTML 与 index.json 路径，清单保留所有失败。
    仅跟随 /document/2 详细链接及固定权益表，拒绝重定向和 api.tushare.pro。
    示例：``fetch_documentation(Path('/tmp/public-docs'))`` 生成可离线复核快照。
    """
    destination.mkdir(parents=True, exist_ok=True)
    queue = [DIRECTORY_URL, ENTITLEMENT_URL]
    visited: set[str] = set()
    index: list[dict[str, str]] = []
    paths: list[Path] = []
    while queue:
        batch = sorted(set(queue) - visited)
        queue = []
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_fetch_one, batch))
        for url, source, error in results:
            visited.add(url)
            identifier = parse_qs(urlsplit(url).query).get("doc_id", [""])[0]
            filename = (
                "entitlements.html"
                if url == ENTITLEMENT_URL
                else f"doc-{identifier}.html"
                if identifier
                else "directory.html"
            )
            item = {
                "doc_url": url,
                "file": filename,
                "status": "failed" if error else "ok",
                "error": error,
                "snapshot_kind": "sanitized_documentation",
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
            index.append(item)
            if error:
                continue
            path = destination / filename
            path.write_text(f"<!-- source_url: {url} -->\n{source}", encoding="utf-8")
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            paths.append(path)
            soup = BeautifulSoup(source, "html.parser")
            for link in soup.select("a[href]"):
                target = _document_url(str(link.get("href", "")), url)
                if target and target.startswith(f"{DIRECTORY_URL}?") and target not in visited:
                    queue.append(target)
        (destination / "index.json").write_text(
            json.dumps(
                sorted(index, key=lambda item: item["doc_url"]), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    return sorted([*paths, destination / "index.json"])


def catalog_summary(entries: list[CatalogEntry]) -> dict[str, Any]:
    """生成不声称全量覆盖的离线发现摘要，供维护文档与 CLI 展示。

    输入：目录对象列表。输出：接口数、未解析页、生命周期和待复核名称。
    示例：空列表返回 ``discovered_interfaces=0``，不会报告“全部完成”。
    """
    states = {
        state: sum(entry.lifecycle == state for entry in entries)
        for state in sorted({entry.lifecycle for entry in entries})
    }
    return {
        "discovered_interfaces": sum(
            not item.api_name.startswith("unresolved_doc_") for item in entries
        ),
        "unresolved_documents": [
            asdict(item) for item in entries if item.api_name.startswith("unresolved_doc_")
        ],
        "lifecycle_counts": states,
        "needs_review": [item.api_name for item in entries if item.needs_review],
        "coverage_claim": "仅为本次公开文档可达集合，不代表官方全部接口或实测账户权限。",
    }
